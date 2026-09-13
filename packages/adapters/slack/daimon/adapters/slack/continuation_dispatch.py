"""Dispatch queued task-continuations at Slack turn completion.

A handoff that carried work to continue queues a `task_continuations` row
(`daimon.core.continuity.continuation`) rather than running a second turn
inline. This module is the Slack-side caller of that contract: after every
turn completes, it lists whatever is still pending for that thread, claims
each row (at-most-once across restarts), decides whether it should still run,
and either runs the receiving agent's first turn or posts the decision's skip
copy.

`run_follow_up` is injected rather than built here so tests can assert
claim/skip/decide behaviour without running a second real turn — the actual
Slack turn (admit -> bind_session -> run_prepared_turn) is wired by the
caller in `app.py`, which already holds the web client, channel and thread
this dispatch runs against.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, cast

import anthropic as anthropic_pkg
import structlog
from anthropic import AsyncAnthropic
from daimon.adapters.slack.context import THREAD_PAGE_LIMIT
from daimon.core.continuity.continuation import (
    ContinuationRequest,
    claim_continuation,
    decide_continuation,
    settle_continuation,
)
from daimon.core.errors import DaimonError
from daimon.core.stores.domain import TaskContinuationRow
from daimon.core.stores.task_continuations import list_pending_continuations
from daimon.core.turn.errors import SessionPreparationFailed
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

log = structlog.get_logger(__name__)

__all__ = ["RunFollowUp", "dispatch_pending_continuations"]

#: Runs the receiving agent's first turn, seeded with the continuation's
#: `requested_work`. Raises on failure; the dispatcher settles the claimed row
#: as skipped (`blocked_preparation_failed` or `dispatch_failed`) so nothing is
#: left claimed by a process that will not deliver it.
RunFollowUp = Callable[[TaskContinuationRow, str], Awaitable[None]]


async def _latest_human_message_at(
    web_client: AsyncWebClient, *, channel: str, thread_ts: str, after: datetime
) -> datetime | None:
    """The newest human (non-bot) message timestamp after `after`, one page.

    Mirrors `context.build_delta_xml`'s `conversations.replies` call exactly
    (`oldest`, `inclusive=False`, `limit=THREAD_PAGE_LIMIT`): the dispatch
    decision must never depend on Slack history beyond what an ordinary turn
    would ever read.
    """
    response = await web_client.conversations_replies(  # pyright: ignore[reportUnknownMemberType]
        channel=channel,
        ts=thread_ts,
        oldest=f"{after.timestamp():.6f}",
        inclusive=False,
        limit=THREAD_PAGE_LIMIT,
    )
    messages = cast(list[dict[str, Any]], response["messages"])  # pyright: ignore[reportUnknownVariableType]
    latest: datetime | None = None
    for msg in messages:
        if "bot_id" in msg or not msg.get("user"):
            continue
        ts_value = cast(str | None, msg.get("ts"))
        if not ts_value:
            continue
        candidate = datetime.fromtimestamp(float(ts_value), tz=UTC)
        if latest is None or candidate > latest:
            latest = candidate
    return latest


async def dispatch_pending_continuations(
    sessionmaker: async_sessionmaker[AsyncSession],
    anthropic: AsyncAnthropic,
    web_client: AsyncWebClient,
    *,
    tenant_id: uuid.UUID,
    channel: str,
    thread_id: str,
    active_turn: bool,
    run_follow_up: RunFollowUp,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> None:
    """Claim and settle every still-pending continuation queued for this thread.

    Called from the Slack turn-completion path (once a turn's own answer has
    been posted and its marker cleared), so the common case is an empty list
    and this is one cheap store read. `active_turn` is the caller's own
    marker check for THIS thread, passed through unchanged to
    `decide_continuation` -- this function does not re-derive it.

    Each row is claimed before it is decided, so a decision to skip is still
    only ever made by the one process that will also settle it. A dispatch
    settles `delivered` only after `run_follow_up` returns without raising;
    a preparation failure settles `skipped`/`blocked_preparation_failed` and any
    other boundary error settles `skipped`/`dispatch_failed`, so a claimed row is
    never left behind by the process that claimed it.
    """
    async with sessionmaker() as session:
        rows = await list_pending_continuations(
            session, tenant_id=tenant_id, platform="slack", thread_id=thread_id
        )
    for row in rows:
        claimed = await claim_continuation(
            sessionmaker, idempotency_key=row.idempotency_key, now=now()
        )
        if not claimed:
            continue

        request = ContinuationRequest(
            tenant_id=row.tenant_id,
            platform="slack",
            parent_channel_id=row.parent_channel_id,
            thread_id=row.thread_id,
            requester_account_id=row.requester_account_id,
            requester_external_user_id=row.requester_external_user_id,
            target_ma_agent_id=row.target_ma_agent_id,
            target_name=row.target_name,
            requested_work=row.requested_work,
            reason=row.reason,
            idempotency_key=row.idempotency_key,
        )
        latest_user_message_at = await _latest_human_message_at(
            web_client, channel=channel, thread_ts=thread_id, after=row.created_at
        )
        decision = await decide_continuation(
            sessionmaker,
            anthropic,
            request=request,
            now=now(),
            latest_user_message_at=latest_user_message_at,
            active_turn=active_turn,
        )

        if decision.action == "dispatch":
            seed = decision.seed_user_message
            if seed is None:
                # decide_continuation never returns "dispatch" without one;
                # guards a caller-side contract break rather than a real path.
                log.error("slack.continuation.dispatch_missing_seed", row_id=str(row.id))
                await settle_continuation(
                    sessionmaker,
                    idempotency_key=row.idempotency_key,
                    status="skipped",
                    now=now(),
                    skip_reason="missing_seed",
                )
                continue
            try:
                await run_follow_up(row, seed)
            except SessionPreparationFailed:
                log.warning("slack.continuation.dispatch_preparation_failed", row_id=str(row.id))
                await settle_continuation(
                    sessionmaker,
                    idempotency_key=row.idempotency_key,
                    status="skipped",
                    now=now(),
                    skip_reason="blocked_preparation_failed",
                )
                continue
            except (DaimonError, anthropic_pkg.APIError, SlackApiError) as exc:
                log.warning(
                    "slack.continuation.dispatch_failed", row_id=str(row.id), error=str(exc)
                )
                await settle_continuation(
                    sessionmaker,
                    idempotency_key=row.idempotency_key,
                    status="skipped",
                    now=now(),
                    skip_reason="dispatch_failed",
                )
                continue
            await settle_continuation(
                sessionmaker, idempotency_key=row.idempotency_key, status="delivered", now=now()
            )
            continue

        if decision.message is not None:
            await web_client.chat_postMessage(  # pyright: ignore[reportUnknownMemberType]
                channel=channel, thread_ts=thread_id, text=decision.message
            )
        await settle_continuation(
            sessionmaker,
            idempotency_key=row.idempotency_key,
            status="skipped",
            now=now(),
            skip_reason=decision.action,
        )
