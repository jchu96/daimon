"""Execution origins remain distinct and are removed when execution ends."""

from datetime import UTC, datetime

import pytest
from daimon.core.stores.domain import Role
from daimon.core.stores.turn_origins import get_active_origin, update_origin_target
from daimon.core.turn_origin import render_turn_origin, turn_origin
from daimon.testing.factories import make_account, make_tenant
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


async def test_simultaneous_origins_keep_separate_snapshots_and_cleanup(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    tenant = await make_tenant(db_session)
    account = await make_account(db_session, tenant=tenant)
    await db_session.commit()
    async with (
        turn_origin(
            db_session_factory,
            tenant_id=tenant.id,
            account_id=account.id,
            platform="discord",
            parent_channel_id="parent",
            thread_id="first",
            responder_ma_agent_id="agent_daimon",
            responder_name="Daimon",
            configuration_target_ma_agent_id="agent_specialist",
            configuration_target_name="specialist",
            role=Role.USER,
        ) as first,
        turn_origin(
            db_session_factory,
            tenant_id=tenant.id,
            account_id=account.id,
            platform="discord",
            parent_channel_id="parent",
            thread_id="second",
            responder_ma_agent_id="agent_daimon",
            responder_name="Daimon",
            configuration_target_ma_agent_id="agent_specialist",
            configuration_target_name="specialist",
            role=Role.USER,
        ) as second,
    ):
        assert first.id != second.id, "each simultaneous execution needs distinct authority"
        async with db_session_factory.begin() as session:
            changed = await update_origin_target(
                session,
                origin_id=first.id,
                configuration_target_ma_agent_id="agent_other",
                configuration_target_name="other",
            )
        async with db_session_factory() as session:
            unchanged = await get_active_origin(
                session,
                origin_id=second.id,
                tenant_id=tenant.id,
                account_id=account.id,
                platform="discord",
                now=datetime.now(UTC),
            )
        assert changed.configuration_target_ma_agent_id == "agent_other", "requesting turn changes"
        assert unchanged == second, "a concurrent turn retains its target and destination"
        controls = render_turn_origin(changed)
        assert str(first.id) in controls, "the current origin must be available every turn"
        assert "agent_other" in controls and "agent_daimon" in controls, (
            "target differs from responder"
        )
    async with db_session_factory() as session:
        assert (
            await get_active_origin(
                session,
                origin_id=first.id,
                tenant_id=tenant.id,
                account_id=account.id,
                platform="discord",
                now=datetime.now(UTC),
            )
            is None
        ), "finished origins must immediately lose authority"


async def test_failed_turn_removes_origin(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    tenant = await make_tenant(db_session)
    account = await make_account(db_session, tenant=tenant)
    await db_session.commit()
    with pytest.raises(RuntimeError, match="failed execution"):
        async with turn_origin(
            db_session_factory,
            tenant_id=tenant.id,
            account_id=account.id,
            platform="slack",
            parent_channel_id="C123",
            thread_id="123.456",
            responder_ma_agent_id="agent_daimon",
            responder_name="Daimon",
            role=Role.ADMIN,
        ) as origin:
            origin_id = origin.id
            raise RuntimeError("failed execution")
    async with db_session_factory() as session:
        assert (
            await get_active_origin(
                session,
                origin_id=origin_id,
                tenant_id=tenant.id,
                account_id=account.id,
                platform="slack",
                now=datetime.now(UTC),
            )
            is None
        ), "exceptions must not leave an active control origin"
