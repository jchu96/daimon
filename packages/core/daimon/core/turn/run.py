"""D-08/D-09/D-10: `run_prepared_turn` owns the driver call and the one-shot
dead-session recovery cycle.

`_is_dead_session` is ported verbatim from
`daimon.adapters.discord.bot._is_dead_session` (D-10) -- applies uniformly to
fresh and reused sessions; a fresh-session 404 costs one harmless retry
rather than adding a reused-only guard (an unnamed behaviour change that was
considered and rejected).
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime

import anthropic as _anthropic
import structlog
from anthropic.types import RawMessageStreamEvent
from anthropic.types.beta.sessions import BetaManagedAgentsImageBlockParam
from daimon.core.stores.thread_sessions import mark_dead
from daimon.core.turn.deps import TurnDeps
from daimon.core.turn.driver import run_turn
from daimon.core.turn.lifecycle import InterruptSource, ReconnectReason, TurnLifecycle
from daimon.core.turn.posture import Billed
from daimon.core.turn.prepare import PreparedTurn, bind_recorder, create_fresh_session
from daimon.core.turn.state import TurnState

log = structlog.get_logger(__name__)

__all__ = ["RunOutcome", "run_prepared_turn"]


@dataclass(frozen=True)
class RunOutcome:
    """The final `TurnState` plus the session/mapping ids the FINAL attempt
    ran against -- needed because the adapter still owns the watermark
    write, which must target the post-recovery mapping_id.
    """

    state: TurnState
    ma_session_id: str
    mapping_id: uuid.UUID | None
    recovered: bool


# MA's rejection when events.send targets a session it has terminated. The id
# is well-formed and the session exists — it is simply closed to new events, so
# this is a 400 rather than the 404 a deleted session gives.
_ARCHIVED_SESSION_MARKER = "cannot send events to archived session"


@dataclass
class _DeferredFailureLifecycle:
    """First-attempt wrapper that holds the terminal-failure hook until we know
    whether recovery will run.

    A dead-session 400 is recoverable and heals in about three seconds, but the
    Discord adapter renders its red error embed from ``on_terminal_failure``
    (despite the protocol calling that hook bookkeeping-only), so the user sees
    a scary ``upstream: Error code: 400`` that is retracted a moment later. The
    adopted message ref means the error does not *persist*; holding the call is
    what stops it being *shown*. Replayed verbatim when we do not recover, so a
    genuinely failed turn is unaffected.
    """

    inner: TurnLifecycle
    _held: tuple[TurnState, Exception] | None = None

    async def on_render(self, state: TurnState) -> None:
        await self.inner.on_render(state)

    async def on_terminal_success(self, state: TurnState) -> None:
        await self.inner.on_terminal_success(state)

    async def on_terminal_failure(self, state: TurnState, err: Exception) -> None:
        self._held = (state, err)

    async def on_sse_event(self, event: RawMessageStreamEvent) -> None:
        await self.inner.on_sse_event(event)

    async def on_reconnect(self, reason: ReconnectReason) -> None:
        await self.inner.on_reconnect(reason)

    async def on_rate_limited(self, until: datetime | None) -> None:
        await self.inner.on_rate_limited(until)

    async def on_interrupt_sent(self, source: InterruptSource) -> None:
        await self.inner.on_interrupt_sent(source)

    async def flush_held_failure(self) -> None:
        """Replay the withheld failure. Call on every path that does not recover."""
        if self._held is None:
            return
        state, err = self._held
        self._held = None
        await self.inner.on_terminal_failure(state, err)


def _is_dead_session(state: TurnState) -> bool:
    """Return True if state.error signals a gone or closed MA session.

    Two distinct signatures, both meaning "this session can never accept
    another event, so recreate rather than surfacing a dead end":

    - **404** from events.send: the session existed but is gone (deleted /
      expired / GC'd).
    - **400 whose message is `Cannot send events to archived session`**: MA
      terminated the session (e.g. a turn hit a terminal model error) and
      closed it to further events.

    Every OTHER 400 still surfaces as a normal turn error. That distinction is
    the point: a bare 400 means a malformed session id, which is not reachable
    with well-formed stored ids and must not trigger a recreate. Matching on
    the message rather than the status alone keeps that case excluded.

    The 400 limb is why this function exists in its current shape. Without it a
    single terminal error bricked the thread PERMANENTLY: MA terminated the
    session, the mapping row still pointed at it, and every later message in
    that thread 400'd here forever with zero tokens billed and no path back.
    Observed on staging thread 1535185295245582356 / session
    sesn_01TBcsjhyD4KMEc6wasC3vyg (2026-08-07), where an oversized image ended
    the session and the next "hello" — and every message after it — failed.

    Note this deliberately does NOT fire on the terminating turn itself (whose
    error is `session terminated by MA`, carrying no APIStatusError cause).
    That turn really did fail, and re-running it against a fresh session would
    just replay whatever killed it. Recovery instead happens on the NEXT
    message, which is the first to see the archived-session 400 — so the thread
    heals on its own without retrying poison.
    """
    err = state.error
    if err is None or err.kind != "upstream":
        return False
    cause = err.cause
    if not isinstance(cause, _anthropic.APIStatusError):
        return False
    if cause.status_code == 404:
        return True
    return cause.status_code == 400 and _ARCHIVED_SESSION_MARKER in str(cause).lower()


async def run_prepared_turn(
    deps: TurnDeps,
    prepared: PreparedTurn,
    *,
    tenant_id: uuid.UUID,
    platform: str,
    thread_id: str,
    external_user_id: str,
    user_message: str,
    lifecycle: TurnLifecycle,
    cancel: asyncio.Event,
    reseed_user_message: Callable[[], Awaitable[str]],
    recovery_lifecycle: Callable[[asyncio.Event], TurnLifecycle],
    image_blocks: Sequence[BetaManagedAgentsImageBlockParam] | None = None,
    render_interval_s: float = 2.0,
) -> RunOutcome:
    """Run one turn against `prepared`'s session; on a dead-session (404)
    signature, recover exactly once: mark the stale mapping dead, create a
    fresh session + mapping row, rebind the recorder to the NEW session id,
    reseed the user message, and re-run once. A second consecutive dead
    signature is returned as-is -- no further retry.

    `external_user_id` is required here (not carried on `PreparedTurn`)
    because recovery must rebuild the usage recorder from scratch against
    the new session id via `prepare.py`'s binding helper, which needs the
    platform user id explicitly, exactly as `bind_session` did to build the
    original recorder.
    """
    ma_session_id = prepared.ma_session_id
    mapping_id = prepared.mapping_id

    first_attempt = _DeferredFailureLifecycle(inner=lifecycle)
    state = await run_turn(
        anthropic=deps.anthropic,
        session_id=ma_session_id,
        user_message=user_message,
        lifecycle=first_attempt,
        cancel=cancel,
        render_interval_s=render_interval_s,
        billing=Billed(record=prepared._record),  # pyright: ignore[reportPrivateUsage]
        image_blocks=image_blocks,
    )

    if not (_is_dead_session(state) and mapping_id is not None):
        await first_attempt.flush_held_failure()
        return RunOutcome(
            state=state,
            ma_session_id=ma_session_id,
            mapping_id=mapping_id,
            recovered=False,
        )

    # If recovery itself blows up, the withheld failure is the only thing the
    # user would ever see -- without this the embed sits on "thinking" forever,
    # which is the exact failure mode this module exists to prevent. Re-raise:
    # the caller still needs to know recovery broke.
    try:
        async with deps.sessionmaker() as session:
            await mark_dead(session, id=mapping_id)
            await session.commit()

        new_session_id, new_mapping_id = await create_fresh_session(
            deps,
            prepared.admission,
            tenant_id=tenant_id,
            platform=platform,
            thread_id=thread_id,
            session_account_id=prepared.session_account_id,
        )

        new_record = bind_recorder(
            deps,
            prepared.admission,
            tenant_id=tenant_id,
            external_user_id=external_user_id,
            ma_session_id=new_session_id,
        )

        log.info(
            "turn.session_recovered",
            old_session_id=ma_session_id,
            new_session_id=new_session_id,
            old_mapping_id=str(mapping_id),
            new_mapping_id=str(new_mapping_id),
            thread_id=thread_id,
        )

        reseeded_message = await reseed_user_message()
        fresh_cancel = asyncio.Event()
        new_lifecycle = recovery_lifecycle(fresh_cancel)

        recovered_state = await run_turn(
            anthropic=deps.anthropic,
            session_id=new_session_id,
            user_message=reseeded_message,
            lifecycle=new_lifecycle,
            cancel=fresh_cancel,
            render_interval_s=render_interval_s,
            billing=Billed(record=new_record),
            image_blocks=image_blocks,
        )
    except Exception:
        await first_attempt.flush_held_failure()
        raise

    return RunOutcome(
        state=recovered_state,
        ma_session_id=new_session_id,
        mapping_id=new_mapping_id,
        recovered=True,
    )
