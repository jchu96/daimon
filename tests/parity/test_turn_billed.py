"""Scenario (a): turn -> billed.

A non-blocked turn dispatched via the REAL platform entry point (D-02) --
`bot.on_message` for Discord, `SlackApp._handle_app_mention` for Slack --
driving the REAL `run_turn` SSE turn driver (D-01, never patched) writes a
`usage_events` row AND a `tenant_ledger` debit, on BOTH platforms. Reverting
Phase-02's `usage_record` wiring into `run_turn` (bot.py:1107-1116 /
app.py:1148-1157) would make this test fail on the affected platform -- the
exact regression class this suite exists to catch.
"""

from __future__ import annotations

from decimal import Decimal
from typing import cast

from daimon.core._models import TenantLedger, UsageEvent
from daimon.core.stores import tenant_ledger
from daimon.core.stores.domain import Platform
from daimon.testing.factories import make_tenant
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .conftest import build_turn_router
from .drivers.protocol import PlatformDriver


async def test_turn_billed_when_unblocked_writes_usage_event_and_ledger_debit(
    driver: PlatformDriver,
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    # Numeric-looking: Discord's driver treats it as a real guild id (int());
    # Slack accepts any string team_id, so one value works for both drivers.
    workspace_id = "900001001"
    user_id = "555000111"

    tenant = await make_tenant(
        db_session, platform=cast(Platform, driver.param_id), workspace_id=workspace_id
    )
    await tenant_ledger.insert_entry(
        db_session,
        tenant_id=tenant.id,
        delta_usd=Decimal("100.00"),
        reason="trial",
        idempotency_key=f"trial:{tenant.id}",
    )
    await db_session.commit()

    router = build_turn_router(str(tenant.id))
    posted = await driver.dispatch_turn(
        sessionmaker=db_session_factory,
        router=router,
        tenant_id=tenant.id,
        workspace_id=workspace_id,
        channel_id="100000",
        user_id=user_id,
        text="hello",
    )
    assert posted, f"expected the agent's reply to be posted somewhere, got: {posted}"

    usage_rows = (
        (await db_session.execute(select(UsageEvent).where(UsageEvent.tenant_id == tenant.id)))
        .scalars()
        .all()
    )
    assert len(usage_rows) == 1, "unblocked turn must write exactly one usage_events row"
    assert usage_rows[0].managed_session_id == "sess_parity_test"
    assert usage_rows[0].model == "claude-sonnet-4-6"

    debit_rows = (
        (
            await db_session.execute(
                select(TenantLedger).where(
                    TenantLedger.tenant_id == tenant.id,
                    TenantLedger.delta_usd < 0,
                )
            )
        )
        .scalars()
        .all()
    )
    assert len(debit_rows) == 1, "unblocked turn must write exactly one tenant_ledger debit"
