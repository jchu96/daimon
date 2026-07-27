"""Scenario (b): over-balance turn blocked pre-spend.

A tenant with a depleted balance (the default for a freshly-provisioned
tenant -- an empty ledger sums to `Decimal("0")`, and `is_over_balance`
treats `balance <= 0` as depleted) is blocked BEFORE `create_session` /
`run_turn` on both platforms, via the REAL platform entry point (D-02).
Asserts the per-driver expected copy (divergence principle, 02-CONTEXT
D-10 -- Discord and Slack copy are allowed to differ) and zero
`usage_events` / `tenant_ledger` rows.
"""

from __future__ import annotations

from typing import cast

from daimon.core._models import TenantLedger, UsageEvent
from daimon.core.stores.domain import Platform
from daimon.testing.factories import make_tenant
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .conftest import build_turn_router
from .drivers.protocol import PlatformDriver


async def test_turn_blocked_when_over_balance_writes_no_usage_and_no_ledger_row(
    driver: PlatformDriver,
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    workspace_id = "900002001"
    user_id = "555000112"

    tenant = await make_tenant(
        db_session, platform=cast(Platform, driver.param_id), workspace_id=workspace_id
    )
    await db_session.commit()

    router = build_turn_router(str(tenant.id))
    posted = await driver.dispatch_turn(
        sessionmaker=db_session_factory,
        router=router,
        tenant_id=tenant.id,
        workspace_id=workspace_id,
        channel_id="100001",
        user_id=user_id,
        text="hello",
    )

    expected = driver.expected_blocked_text("balance")
    assert expected in posted, f"expected the over-balance copy {expected!r}, got: {posted}"

    usage_rows = (
        (await db_session.execute(select(UsageEvent).where(UsageEvent.tenant_id == tenant.id)))
        .scalars()
        .all()
    )
    assert usage_rows == [], "over-balance turn must write zero usage_events rows"

    ledger_rows = (
        (await db_session.execute(select(TenantLedger).where(TenantLedger.tenant_id == tenant.id)))
        .scalars()
        .all()
    )
    assert ledger_rows == [], "over-balance turn must write zero tenant_ledger rows"
