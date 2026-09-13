"""Queued first turns for an agent a task was just handed to.

Handing a task over may carry work to continue. That continuation is a turn
somebody will be billed for, posted into a thread people are reading, so it
must happen at most once — across process restarts, across two adapter
processes, across a retry loop. `claim_continuation` is the whole guarantee: a
single conditional UPDATE, so the database decides the winner and everyone else
gets False and does nothing.

`requested_work is None` means the handoff carried no work: the switch is
recorded for the audit trail and nothing is ever dispatched.
"""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime
from typing import Literal

from daimon.core._models import TaskContinuation
from daimon.core.stores.domain import ContinuationReason, TaskContinuationRow
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession


async def record_continuation(
    session: AsyncSession,
    *,
    tenant_id: _uuid.UUID,
    platform: str,
    parent_channel_id: str,
    thread_id: str,
    requester_account_id: _uuid.UUID,
    requester_external_user_id: str,
    target_ma_agent_id: str,
    target_name: str,
    reason: ContinuationReason,
    idempotency_key: _uuid.UUID,
    requested_work: str | None = None,
) -> TaskContinuationRow:
    """Queue a continuation as `pending`. Writing it dispatches nothing.

    `target_ma_agent_id` is stored concrete, never a name: a name resolves
    differently later, and a continuation must reach the agent the requester
    actually chose. `idempotency_key` is minted by the caller so a retried tool
    call recognises its own row instead of queueing a second one.
    """
    orm = TaskContinuation(
        tenant_id=tenant_id,
        platform=platform,
        parent_channel_id=parent_channel_id,
        thread_id=thread_id,
        requester_account_id=requester_account_id,
        requester_external_user_id=requester_external_user_id,
        target_ma_agent_id=target_ma_agent_id,
        target_name=target_name,
        requested_work=requested_work,
        reason=reason,
        idempotency_key=idempotency_key,
    )
    session.add(orm)
    await session.flush()
    await session.refresh(orm)
    return TaskContinuationRow.model_validate(orm)


async def claim_continuation(
    session: AsyncSession,
    *,
    idempotency_key: _uuid.UUID,
    now: datetime,
) -> bool:
    """Take exclusive responsibility for dispatching this continuation.

    The at-most-once primitive. One `UPDATE … WHERE status = 'pending'
    RETURNING`: Postgres serializes two racing transactions on the row lock,
    and the loser re-evaluates the predicate against the winner's committed
    `'claimed'` and matches nothing. Exactly one caller sees True, so exactly
    one turn is ever dispatched — a restart mid-dispatch cannot double-post.

    A True return means the caller now owns the row and must settle it.
    """
    claimed = (
        await session.execute(
            update(TaskContinuation)
            .where(
                TaskContinuation.idempotency_key == idempotency_key,
                TaskContinuation.status == "pending",
            )
            .values(status="claimed", claimed_at=now)
            .returning(TaskContinuation.id)
        )
    ).scalar_one_or_none()
    await session.flush()
    return claimed is not None


async def settle_continuation(
    session: AsyncSession,
    *,
    idempotency_key: _uuid.UUID,
    status: Literal["delivered", "skipped"],
    now: datetime,
    skip_reason: str | None = None,
) -> None:
    """Close out a claimed continuation, delivered or deliberately skipped.

    `delivered_at` is stamped only on delivery, so a skipped row never reads as
    though a turn ran for it.
    """
    values: dict[str, object] = {"status": status, "skip_reason": skip_reason}
    if status == "delivered":
        values["delivered_at"] = now
    await session.execute(
        update(TaskContinuation)
        .where(TaskContinuation.idempotency_key == idempotency_key)
        .values(**values)
    )
    await session.flush()


async def get_continuation(
    session: AsyncSession,
    *,
    idempotency_key: _uuid.UUID,
) -> TaskContinuationRow | None:
    orm = (
        await session.execute(
            select(TaskContinuation).where(TaskContinuation.idempotency_key == idempotency_key)
        )
    ).scalar_one_or_none()
    return None if orm is None else TaskContinuationRow.model_validate(orm)


async def list_pending_continuations(
    session: AsyncSession,
    *,
    tenant_id: _uuid.UUID,
    platform: str,
    thread_id: str,
) -> list[TaskContinuationRow]:
    """Undispatched continuations for one thread, oldest first."""
    rows = (
        await session.execute(
            select(TaskContinuation)
            .where(
                TaskContinuation.tenant_id == tenant_id,
                TaskContinuation.platform == platform,
                TaskContinuation.thread_id == thread_id,
                TaskContinuation.status == "pending",
            )
            .order_by(TaskContinuation.created_at, TaskContinuation.id)
        )
    ).scalars()
    return [TaskContinuationRow.model_validate(row) for row in rows]
