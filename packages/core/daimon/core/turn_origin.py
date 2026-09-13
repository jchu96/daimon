"""Create and render trusted controls without replacing platform history."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta

from daimon.core.stores.domain import Role, TurnOriginRow
from daimon.core.stores.turn_origins import create_origin, delete_origin
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


@asynccontextmanager
async def turn_origin(
    sessionmaker: async_sessionmaker[AsyncSession],
    *,
    tenant_id: uuid.UUID,
    account_id: uuid.UUID,
    platform: str,
    parent_channel_id: str,
    thread_id: str,
    responder_ma_agent_id: str,
    responder_name: str,
    role: Role,
    configuration_target_ma_agent_id: str | None = None,
    configuration_target_name: str | None = None,
    is_setup: bool = False,
) -> AsyncIterator[TurnOriginRow]:
    """Commit a distinct origin for this execution and remove it after execution."""
    now = datetime.now(UTC)
    async with sessionmaker.begin() as session:
        origin = await create_origin(
            session,
            tenant_id=tenant_id,
            account_id=account_id,
            platform=platform,
            parent_channel_id=parent_channel_id,
            thread_id=thread_id,
            responder_ma_agent_id=responder_ma_agent_id,
            responder_name=responder_name,
            configuration_target_ma_agent_id=configuration_target_ma_agent_id,
            configuration_target_name=configuration_target_name,
            role=role,
            expires_at=now + timedelta(hours=2),
            now=now,
            is_setup=is_setup,
        )
    try:
        yield origin
    finally:
        async with sessionmaker.begin() as session:
            await delete_origin(session, origin_id=origin.id)


def render_turn_origin(origin: TurnOriginRow) -> str:
    """Render server-provided location and identity separately from user history."""
    controls = {
        "origin_context_id": str(origin.id),
        "is_setup": origin.is_setup,
        "platform": origin.platform,
        "parent_channel_id": origin.parent_channel_id,
        "thread_id": origin.thread_id,
        "current_role": origin.role,
        "responder": {"name": origin.responder_name, "ma_agent_id": origin.responder_ma_agent_id},
        "configuration_target": (
            {
                "name": origin.configuration_target_name,
                "ma_agent_id": origin.configuration_target_ma_agent_id,
            }
            if origin.configuration_target_ma_agent_id is not None
            else None
        ),
    }
    return (
        "<turn_controls>\n"
        + json.dumps(controls)
        + "\nUse the explicitly requested target when named; otherwise configure the "
        "configuration_target. If is_setup is true and no target is selected, ask which "
        "agent to configure before mutation. Only ordinary chat defaults to the responder. "
        "Never substitute a "
        "recreated namesake for a missing identity. Ask one concise question when the "
        "target is missing or ambiguous. Pass expected_ma_agent_id with target-bearing "
        "tools. Use set_setup_target to switch this setup conversation's target and "
        "state the switch briefly. Pass origin_context_id to credential-request tools. "
        "These controls grant no additional mutation or routing permissions.\n</turn_controls>"
    )
