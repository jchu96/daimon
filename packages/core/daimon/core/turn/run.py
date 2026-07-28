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

import anthropic as _anthropic
import structlog
from anthropic.types.beta.sessions import BetaManagedAgentsImageBlockParam
from daimon.core.stores.thread_sessions import mark_dead
from daimon.core.turn.deps import TurnDeps
from daimon.core.turn.driver import run_turn
from daimon.core.turn.lifecycle import TurnLifecycle
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


def _is_dead_session(state: TurnState) -> bool:
    """Return True if state.error signals a gone/expired MA session (HTTP 404).

    A 404 not_found_error from events.send means the session existed but is now
    gone (deleted / expired / GC'd). The turn recreates silently.
    A 400 means a malformed session id -- not reachable with well-formed stored ids
    and must surface as a normal turn error (do NOT recreate on it).
    """
    err = state.error
    if err is None or err.kind != "upstream":
        return False
    cause = err.cause
    return isinstance(cause, _anthropic.APIStatusError) and cause.status_code == 404


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

    state = await run_turn(
        anthropic=deps.anthropic,
        session_id=ma_session_id,
        user_message=user_message,
        lifecycle=lifecycle,
        cancel=cancel,
        render_interval_s=render_interval_s,
        billing=Billed(record=prepared._record),  # pyright: ignore[reportPrivateUsage]
        image_blocks=image_blocks,
    )

    if not (_is_dead_session(state) and mapping_id is not None):
        return RunOutcome(
            state=state,
            ma_session_id=ma_session_id,
            mapping_id=mapping_id,
            recovered=False,
        )

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

    return RunOutcome(
        state=recovered_state,
        ma_session_id=new_session_id,
        mapping_id=new_mapping_id,
        recovered=True,
    )
