"""Tenant and location scoped setup conversations, shared by their participants."""

from __future__ import annotations

import uuid
from typing import Literal

from daimon.core._models import ThreadAgentBinding
from daimon.core.stores.domain import ThreadAgentBindingRow
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession


async def create_binding(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    platform: str,
    parent_channel_id: str,
    thread_id: str,
    responder_ma_agent_id: str,
    responder_name: str,
    configuration_target_ma_agent_id: str | None = None,
    configuration_target_name: str | None = None,
    creator_account_id: uuid.UUID | None = None,
    kind: Literal["setup", "handoff"] = "setup",
) -> ThreadAgentBindingRow:
    """Bind a thread to a responder.

    `kind` says why the binding exists: `'setup'` is a setup conversation,
    `'handoff'` records that a task in an ordinary thread was handed to a
    different agent. They share the table because both answer the same
    question — who replies in this thread — but only setup bindings are
    setup conversations, which is why `list_active_bindings` filters on it.
    """
    binding = ThreadAgentBinding(
        tenant_id=tenant_id,
        platform=platform,
        parent_channel_id=parent_channel_id,
        thread_id=thread_id,
        kind=kind,
        responder_ma_agent_id=responder_ma_agent_id,
        responder_name=responder_name,
        configuration_target_ma_agent_id=configuration_target_ma_agent_id,
        configuration_target_name=configuration_target_name,
        creator_account_id=creator_account_id,
    )
    session.add(binding)
    await session.flush()
    return ThreadAgentBindingRow.model_validate(binding)


async def get_binding(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    platform: str,
    parent_channel_id: str,
    thread_id: str,
) -> ThreadAgentBindingRow | None:
    binding = (
        await session.execute(
            select(ThreadAgentBinding).where(
                ThreadAgentBinding.tenant_id == tenant_id,
                ThreadAgentBinding.platform == platform,
                ThreadAgentBinding.parent_channel_id == parent_channel_id,
                ThreadAgentBinding.thread_id == thread_id,
            )
        )
    ).scalar_one_or_none()
    return ThreadAgentBindingRow.model_validate(binding) if binding is not None else None


async def list_active_bindings(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    platform: str,
    parent_channel_id: str,
    limit: int = 10,
) -> list[ThreadAgentBindingRow]:
    """Live setup conversations under a channel.

    Setup only, by design: every caller of this lists setup conversations to a
    user. A handoff binding is a fact about one ordinary thread, not a setup
    conversation, and surfacing it here would misdescribe the thread.
    """
    bindings = (
        await session.execute(
            select(ThreadAgentBinding)
            .where(
                ThreadAgentBinding.tenant_id == tenant_id,
                ThreadAgentBinding.platform == platform,
                ThreadAgentBinding.parent_channel_id == parent_channel_id,
                ThreadAgentBinding.kind == "setup",
                ThreadAgentBinding.archived.is_(False),
                ThreadAgentBinding.locked.is_(False),
                ThreadAgentBinding.deleted.is_(False),
            )
            .order_by(ThreadAgentBinding.updated_at.desc(), ThreadAgentBinding.id)
            .limit(min(limit, 10))
        )
    ).scalars()
    return [ThreadAgentBindingRow.model_validate(binding) for binding in bindings]


async def update_target(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    platform: str,
    parent_channel_id: str,
    thread_id: str,
    configuration_target_ma_agent_id: str,
    configuration_target_name: str,
) -> ThreadAgentBindingRow | None:
    binding = (
        await session.execute(
            update(ThreadAgentBinding)
            .where(
                ThreadAgentBinding.tenant_id == tenant_id,
                ThreadAgentBinding.platform == platform,
                ThreadAgentBinding.parent_channel_id == parent_channel_id,
                ThreadAgentBinding.thread_id == thread_id,
                ThreadAgentBinding.deleted.is_(False),
            )
            .values(
                configuration_target_ma_agent_id=configuration_target_ma_agent_id,
                configuration_target_name=configuration_target_name,
                updated_at=func.now(),
            )
            .returning(ThreadAgentBinding)
        )
    ).scalar_one_or_none()
    await session.flush()
    return ThreadAgentBindingRow.model_validate(binding) if binding is not None else None


async def update_lifecycle(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    platform: str,
    parent_channel_id: str,
    thread_id: str,
    archived: bool | None = None,
    locked: bool | None = None,
    deleted: bool | None = None,
) -> None:
    values: dict[str, object] = {"updated_at": func.now()}
    for name, value in (("archived", archived), ("locked", locked), ("deleted", deleted)):
        if value is not None:
            values[name] = value
    await session.execute(
        update(ThreadAgentBinding)
        .where(
            ThreadAgentBinding.tenant_id == tenant_id,
            ThreadAgentBinding.platform == platform,
            ThreadAgentBinding.parent_channel_id == parent_channel_id,
            ThreadAgentBinding.thread_id == thread_id,
            ThreadAgentBinding.deleted.is_(False),
        )
        .values(**values)
    )
    await session.flush()


async def update_channel_lifecycle(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    platform: str,
    parent_channel_id: str,
    archived: bool | None = None,
    deleted: bool | None = None,
) -> None:
    values: dict[str, object] = {"updated_at": func.now()}
    if archived is not None:
        values["archived"] = archived
    if deleted is not None:
        values["deleted"] = deleted
    await session.execute(
        update(ThreadAgentBinding)
        .where(
            ThreadAgentBinding.tenant_id == tenant_id,
            ThreadAgentBinding.platform == platform,
            ThreadAgentBinding.parent_channel_id == parent_channel_id,
            ThreadAgentBinding.deleted.is_(False),
        )
        .values(**values)
    )
    await session.flush()
