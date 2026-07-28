"""Scenario: dead-session recovery driven through the platform entry point.

A turn against a live `thread_sessions` row whose MA session has expired/been
GC'd (confirmed signal: 404 `not_found_error` from `events.send`) must mark
the stale row dead, create a fresh MA session + a new live row, re-run once
with full history, and bill the NEW session id -- never the dead one. This is
`run_prepared_turn`'s one-shot recovery cycle (D-08/D-09/D-10), proven here at
the real platform entry point (`DaimonBot.on_message`) rather than only at
the core unit-test level (`packages/core/tests/turn/test_run_prepared_turn.py`).

Discord only for this plan -- the Slack half lands in a follow-up plan once
its adapter is cut over to the orchestrator; the fixtures below are written
so a Slack scenario can be added without reshaping this file.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import httpx
from anthropic.types.beta import BetaEnvironment, BetaManagedAgentsAgent
from anthropic.types.beta.beta_managed_agents_model_config import BetaManagedAgentsModelConfig
from anthropic.types.beta.sessions.beta_managed_agents_agent_message_event import (
    BetaManagedAgentsAgentMessageEvent,
)
from anthropic.types.beta.sessions.beta_managed_agents_session_end_turn import (
    BetaManagedAgentsSessionEndTurn,
)
from anthropic.types.beta.sessions.beta_managed_agents_session_status_idle_event import (
    BetaManagedAgentsSessionStatusIdleEvent,
)
from anthropic.types.beta.sessions.beta_managed_agents_span_model_request_end_event import (
    BetaManagedAgentsSpanModelRequestEndEvent,
)
from anthropic.types.beta.sessions.beta_managed_agents_span_model_usage import (
    BetaManagedAgentsSpanModelUsage,
)
from anthropic.types.beta.sessions.beta_managed_agents_text_block import BetaManagedAgentsTextBlock
from daimon.core.defaults.metadata import MA_METADATA_KEY_NAME, MA_METADATA_KEY_TENANT
from daimon.core.stores import tenant_ledger, usage_events
from daimon.core.stores.identity import get_or_create_platform_principal
from daimon.core.stores.thread_sessions import (
    create_thread_session,
    get_live_thread_session,
    get_thread_session_by_id,
)
from daimon.testing.factories import make_tenant
from daimon.testing.ma import (
    EMPTY_CLOUD_CONFIG,
    MARouter,
    list_response,
    not_found_response,
    send_events_response,
    sse_response,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .conftest import AGENT_ID, AGENT_TEXT, ENV_ID, MODEL_ID
from .drivers.discord_driver import DiscordDriver

# The Discord driver's `create_session` stub always returns this fixed session
# id for the recreate call (see DiscordDriver._make_fake_session usage in
# dispatch_turn) -- the recovered-turn's NEW session id in this scenario.
_RECOVERED_SESSION_ID = "sess_parity_test"
_DEAD_SESSION_ID = "sess_dead_before_recovery"


def _build_dead_session_router(tenant_id_str: str) -> MARouter:
    """Agent/environment resolution identical to `build_turn_router`, but with
    session-id-scoped event routes: the OLD (already-dead) session's SSE
    stream open 404s (the confirmed dead-session signal -- `run_turn` opens
    the stream before posting the initial user message, so the 404 surfaces
    there), the recreated session's stream open + `events.send` succeed.
    """
    agent_item = BetaManagedAgentsAgent(
        id=AGENT_ID,
        type="agent",
        name="test-agent",
        model=BetaManagedAgentsModelConfig(id=MODEL_ID),
        metadata={
            MA_METADATA_KEY_TENANT: tenant_id_str,
            MA_METADATA_KEY_NAME: "test-agent",
        },
        description=None,
        created_at="2026-06-14T00:00:00Z",  # pyright: ignore[reportArgumentType]
        updated_at="2026-06-14T00:00:00Z",  # pyright: ignore[reportArgumentType]
        version=1,
        mcp_servers=[],
        skills=[],
        tools=[],
        system=None,
    ).model_dump(mode="json")

    env_item = BetaEnvironment(
        id=ENV_ID,
        type="environment",
        name="test-env",
        config=EMPTY_CLOUD_CONFIG,
        metadata={
            MA_METADATA_KEY_TENANT: tenant_id_str,
            MA_METADATA_KEY_NAME: "test-env",
        },
        description="",
        created_at="2026-06-14T00:00:00Z",
        updated_at="2026-06-14T00:00:00Z",
    ).model_dump(mode="json")

    now = datetime.now(UTC)
    agent_message_event = BetaManagedAgentsAgentMessageEvent(
        id="evt_parity_recover_msg",
        type="agent.message",
        processed_at=now,
        content=[BetaManagedAgentsTextBlock(type="text", text=AGENT_TEXT)],
    ).model_dump(mode="json")
    model_request_end_event = BetaManagedAgentsSpanModelRequestEndEvent(
        id="evt_parity_recover_usage",
        is_error=False,
        model_request_start_id="start_parity_recover",
        model_usage=BetaManagedAgentsSpanModelUsage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        processed_at=now,
        type="span.model_request_end",
    ).model_dump(mode="json")
    idle_event = BetaManagedAgentsSessionStatusIdleEvent(
        id="evt_parity_recover_idle",
        type="session.status_idle",
        processed_at=now,
        stop_reason=BetaManagedAgentsSessionEndTurn(type="end_turn"),
    ).model_dump(mode="json")

    router = MARouter()
    router.add("GET", r"/v1/agents", lambda req, _m: list_response([agent_item]))
    router.add("GET", r"/v1/agents/[^/]+", lambda req, _m: httpx.Response(200, json=agent_item))
    router.add("GET", r"/v1/environments", lambda req, _m: list_response([env_item]))
    router.add("GET", r"/v1/environments/[^/]+", lambda req, _m: httpx.Response(200, json=env_item))
    # OLD session: opening the SSE stream 404s -- the dead-session signal
    # (`run_turn` opens the stream before `send_initial`, so this is the
    # first request the driver makes against a gone session).
    router.add(
        "GET",
        rf"/v1/sessions/{_DEAD_SESSION_ID}/events/stream",
        lambda req, _m: not_found_response("session gone"),
    )
    # Recreated session: events.send + SSE stream succeed.
    router.add(
        "POST",
        rf"/v1/sessions/{_RECOVERED_SESSION_ID}/events",
        lambda req, _m: send_events_response(),
    )
    router.add(
        "GET",
        rf"/v1/sessions/{_RECOVERED_SESSION_ID}/events/stream",
        lambda req, _m: sse_response([agent_message_event, model_request_end_event, idle_event]),
    )
    return router


async def test_dead_session_recreates_marks_old_row_dead_and_bills_new_session(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    workspace_id = "900002001"
    user_id = "555000222"
    thread_id = "200000"

    tenant = await make_tenant(db_session, platform="discord", workspace_id=workspace_id)
    await tenant_ledger.insert_entry(
        db_session,
        tenant_id=tenant.id,
        delta_usd=Decimal("100.00"),
        reason="trial",
        idempotency_key=f"trial:{tenant.id}",
    )
    # Identity resolution inside admit() is idempotent get-or-create -- pre-creating
    # here just lets the test learn the account_id up front to seed the live row.
    principal = await get_or_create_platform_principal(
        db_session, tenant_id=tenant.id, platform="discord", external_id=user_id
    )
    old_row = await create_thread_session(
        db_session,
        tenant_id=tenant.id,
        platform="discord",
        thread_id=thread_id,
        account_id=principal.account_id,
        ma_session_id=_DEAD_SESSION_ID,
        watermark_message_id="100",
    )
    await db_session.commit()

    router = _build_dead_session_router(str(tenant.id))
    driver = DiscordDriver()
    posted = await driver.dispatch_turn(
        sessionmaker=db_session_factory,
        router=router,
        tenant_id=tenant.id,
        workspace_id=workspace_id,
        channel_id=thread_id,
        user_id=user_id,
        text="hello after recovery",
    )
    assert posted, f"expected the recovered turn's reply to be posted, got: {posted}"

    # The stale mapping is marked dead.
    dead_row = await get_thread_session_by_id(db_session, id=old_row.id)
    assert dead_row is not None, "the pre-existing mapping row must still exist"
    assert dead_row.status == "dead", "the pre-existing mapping must be marked dead"

    # A new live row exists with a different ma_session_id.
    live_row = await get_live_thread_session(
        db_session,
        tenant_id=tenant.id,
        platform="discord",
        thread_id=thread_id,
        account_id=principal.account_id,
    )
    assert live_row is not None, "a new live row must exist after recovery"
    assert live_row.id != old_row.id, "the new live row must be a distinct row"
    assert live_row.ma_session_id == _RECOVERED_SESSION_ID, (
        "the new live row must store the recreated session id"
    )
    assert live_row.ma_session_id != _DEAD_SESSION_ID, (
        "the new live row must NOT reuse the dead session id"
    )

    # The turn produced a final assistant message (already asserted via `posted`
    # above, since the driver only records message.channel.send / thread.send /
    # message_ref.edit text calls).

    # usage_events must bill the NEW session id, never the dead one (the exact
    # attribution bug D-08/D-09/D-10 fixes -- RESEARCH § "Known attribution bug").
    usage_rows = await usage_events.list_for_tenant(db_session, tenant_id=tenant.id)
    assert len(usage_rows) == 1, "the recovered turn must write exactly one usage_events row"
    assert usage_rows[0].managed_session_id == _RECOVERED_SESSION_ID, (
        "usage_events must attribute the recovered turn to the NEW session id"
    )
    assert usage_rows[0].managed_session_id != _DEAD_SESSION_ID, (
        "usage_events must never attribute the recovered turn to the dead session id"
    )

    ledger_rows = await tenant_ledger.list_for_tenant(db_session, tenant_id=tenant.id)
    debit_rows = [row for row in ledger_rows if row.delta_usd < 0]
    assert len(debit_rows) == 1, "the recovered turn must write exactly one tenant_ledger debit"
