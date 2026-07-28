"""Unit tests for daimon.core.turn.prepare.bind_session -- the D-01 stage-two
chokepoint: thread-session find-or-create, the single shared `create_session`
call, the `thread_sessions` mapping write, and recorder binding.
"""

from __future__ import annotations

import dataclasses
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import httpx
from anthropic.types.beta import BetaEnvironment, BetaManagedAgentsAgent
from anthropic.types.beta.beta_managed_agents_model_config import BetaManagedAgentsModelConfig
from anthropic.types.beta.sessions.beta_managed_agents_span_model_request_end_event import (
    BetaManagedAgentsSpanModelRequestEndEvent,
)
from anthropic.types.beta.sessions.beta_managed_agents_span_model_usage import (
    BetaManagedAgentsSpanModelUsage,
)
from daimon.core.config import McpSettings
from daimon.core.defaults.metadata import MA_METADATA_KEY_NAME, MA_METADATA_KEY_TENANT
from daimon.core.ma_resolver import new_resolver_cache
from daimon.core.scope import DeploymentDefault, ResolvedConfig
from daimon.core.stores import usage_events
from daimon.core.turn.admission import Admission
from daimon.core.turn.deps import TurnDeps
from daimon.core.turn.prepare import PreparedTurn, bind_session
from daimon.testing.ma import (
    EMPTY_CLOUD_CONFIG,
    MARouter,
    build_fake_anthropic,
    make_fake_memory_store_handler,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from daimon.testing.factories import (  # isort: skip
    make_account,
    make_tenant,
    make_thread_session,
)

_NOW = datetime(2026, 7, 28, tzinfo=UTC)


def _agent(*, agent_id: str, tenant_id: uuid.UUID, name: str = "daimon") -> BetaManagedAgentsAgent:
    now = datetime.now(UTC)
    return BetaManagedAgentsAgent(
        id=agent_id,
        type="agent",
        name=name,
        version=1,
        model=BetaManagedAgentsModelConfig(id="claude-sonnet-4-6", speed="standard"),
        system=None,
        description=None,
        metadata={MA_METADATA_KEY_TENANT: str(tenant_id), MA_METADATA_KEY_NAME: name},
        mcp_servers=[],
        tools=[],
        skills=[],
        created_at=now,
        updated_at=now,
        archived_at=None,
    )


def _env(*, env_id: str, tenant_id: uuid.UUID, name: str = "default") -> BetaEnvironment:
    now_iso = datetime.now(UTC).isoformat()
    return BetaEnvironment(
        id=env_id,
        type="environment",
        name=name,
        description="",
        config=EMPTY_CLOUD_CONFIG,
        metadata={MA_METADATA_KEY_TENANT: str(tenant_id), MA_METADATA_KEY_NAME: name},
        created_at=now_iso,
        updated_at=now_iso,
        archived_at=None,
    )


def _admission(
    *, account_id: uuid.UUID, agent: BetaManagedAgentsAgent, env: BetaEnvironment
) -> Admission:
    return Admission(
        account_id=account_id,
        agent=agent,
        environment=env,
        config=ResolvedConfig(agent_name="daimon", environment_name="default"),
    )


def _router_with_session_create(*, session_bodies: list[dict[str, object]]) -> MARouter:
    """A router serving the memory-store cold-provision path + session-create,
    which is all `create_session` needs when no vault/env-secrets/repo-binding
    are configured. Captures each session-create request body."""
    router = MARouter()
    memory_handler = make_fake_memory_store_handler()

    def _memory(request: httpx.Request, _match: object) -> httpx.Response:
        return memory_handler(request)

    router.add("POST", r"/v1/memory_stores", _memory)

    def _session_create(request: httpx.Request, _match: object) -> httpx.Response:
        import json

        body = json.loads(request.content)
        session_bodies.append(body)
        return httpx.Response(
            200,
            json={
                "id": f"sess_{len(session_bodies)}",
                "type": "session",
                "agent": {
                    "id": body["agent"],
                    "mcp_servers": [],
                    "model": {"id": "claude-sonnet-4-6"},
                    "name": "daimon",
                    "skills": [],
                    "tools": [],
                    "type": "agent",
                    "version": 1,
                },
                "created_at": "2026-07-28T00:00:00Z",
                "outcome_evaluations": [],
                "environment_id": body["environment_id"],
                "metadata": {},
                "resources": [],
                "stats": {},
                "status": "idle",
                "updated_at": "2026-07-28T00:00:00Z",
                "usage": {},
                "vault_ids": [],
            },
        )

    router.add("POST", r"/v1/sessions", _session_create)
    return router


def _deps(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    router: MARouter,
) -> TurnDeps:
    return TurnDeps(
        anthropic=build_fake_anthropic(router.dispatch),
        sessionmaker=sessionmaker,
        deployment_default=DeploymentDefault(),
        resolver_cache=new_resolver_cache(),
        defaults_root=Path("/nonexistent"),
        mcp=McpSettings(),
        billing_config=None,
        markup=Decimal("1.0"),
        fernet=None,
        github_fallback_pat=None,
        github_app_id=None,
        github_app_private_key=None,
        public_url=None,
    )


async def test_bind_session_reuses_live_row_when_reuse_existing_true_and_row_exists(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    tenant = await make_tenant(db_session)
    account = await make_account(db_session, tenant=tenant)
    row = await make_thread_session(
        db_session,
        tenant=tenant,
        account=account,
        platform="discord",
        thread_id="thread-1",
        ma_session_id="sess_existing",
        watermark_message_id="msg-99",
    )
    await db_session.commit()

    def _explode(_request: httpx.Request, _match: object) -> httpx.Response:
        raise AssertionError("create_session must not be called on a reuse hit")

    router = MARouter()
    router.add("POST", r"/v1/sessions", _explode)
    deps = _deps(sessionmaker=db_session_factory, router=router)
    agent = _agent(agent_id="ag_1", tenant_id=tenant.id)
    env = _env(env_id="env_1", tenant_id=tenant.id)
    admission = _admission(account_id=account.id, agent=agent, env=env)

    prepared = await bind_session(
        deps,
        admission,
        tenant_id=tenant.id,
        platform="discord",
        external_user_id="user-1",
        thread_id="thread-1",
        session_account_id=account.id,
        reuse_existing=True,
    )

    assert prepared.reused is True, "a live row hit must set reused=True"
    assert prepared.ma_session_id == "sess_existing", "must return the row's ma_session_id"
    assert prepared.mapping_id == row.id, "must return the row's id as mapping_id"
    assert prepared.watermark == "msg-99", "must return the row's watermark_message_id"


async def test_bind_session_creates_session_and_writes_mapping_when_no_live_row(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    tenant = await make_tenant(db_session)
    account = await make_account(db_session, tenant=tenant)
    await db_session.commit()

    session_bodies: list[dict[str, object]] = []
    router = _router_with_session_create(session_bodies=session_bodies)
    deps = _deps(sessionmaker=db_session_factory, router=router)
    agent = _agent(agent_id="ag_1", tenant_id=tenant.id)
    env = _env(env_id="env_1", tenant_id=tenant.id)
    admission = _admission(account_id=account.id, agent=agent, env=env)

    prepared = await bind_session(
        deps,
        admission,
        tenant_id=tenant.id,
        platform="discord",
        external_user_id="user-1",
        thread_id="thread-2",
        session_account_id=account.id,
        reuse_existing=True,
    )

    assert len(session_bodies) == 1, "no live row must trigger exactly one create_session call"
    assert prepared.reused is False, "no live row must set reused=False"
    assert prepared.watermark is None, "a fresh session has no watermark yet"
    assert prepared.mapping_id is not None, "a new thread_sessions row must be written"

    async with db_session_factory() as s:
        from daimon.core.stores.thread_sessions import get_live_thread_session

        live = await get_live_thread_session(
            s,
            tenant_id=tenant.id,
            platform="discord",
            thread_id="thread-2",
            account_id=account.id,
        )
    assert live is not None, "the mapping write must be visible via get_live_thread_session"
    assert live.ma_session_id == prepared.ma_session_id, (
        "the persisted row's ma_session_id must match the returned PreparedTurn"
    )


async def test_bind_session_always_creates_fresh_session_when_reuse_existing_false(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    tenant = await make_tenant(db_session)
    account = await make_account(db_session, tenant=tenant)
    # A live row DOES exist for this thread -- but reuse_existing=False must
    # ignore it (Discord's channel-mention path).
    await make_thread_session(
        db_session,
        tenant=tenant,
        account=account,
        platform="discord",
        thread_id="thread-3",
        ma_session_id="sess_should_not_be_reused",
    )
    await db_session.commit()

    session_bodies: list[dict[str, object]] = []
    router = _router_with_session_create(session_bodies=session_bodies)
    deps = _deps(sessionmaker=db_session_factory, router=router)
    agent = _agent(agent_id="ag_1", tenant_id=tenant.id)
    env = _env(env_id="env_1", tenant_id=tenant.id)
    admission = _admission(account_id=account.id, agent=agent, env=env)

    prepared = await bind_session(
        deps,
        admission,
        tenant_id=tenant.id,
        platform="discord",
        external_user_id="user-1",
        thread_id="thread-3",
        session_account_id=account.id,
        reuse_existing=False,
    )

    assert len(session_bodies) == 1, "reuse_existing=False must always create a fresh session"
    assert prepared.reused is False, "reuse_existing=False must report reused=False"
    assert prepared.ma_session_id != "sess_should_not_be_reused", (
        "the existing live row must be ignored, not reused"
    )


async def test_bind_session_recorder_writes_usage_event_matching_ma_session_id(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    tenant = await make_tenant(db_session)
    account = await make_account(db_session, tenant=tenant)
    await db_session.commit()

    session_bodies: list[dict[str, object]] = []
    router = _router_with_session_create(session_bodies=session_bodies)
    deps = _deps(sessionmaker=db_session_factory, router=router)
    agent = _agent(agent_id="ag_1", tenant_id=tenant.id)
    env = _env(env_id="env_1", tenant_id=tenant.id)
    admission = _admission(account_id=account.id, agent=agent, env=env)

    prepared = await bind_session(
        deps,
        admission,
        tenant_id=tenant.id,
        platform="discord",
        external_user_id="user-1",
        thread_id="thread-4",
        session_account_id=account.id,
        reuse_existing=True,
    )

    event = BetaManagedAgentsSpanModelRequestEndEvent(
        id="evt_1",
        is_error=False,
        model_request_start_id="start_1",
        model_usage=BetaManagedAgentsSpanModelUsage(
            input_tokens=10,
            output_tokens=20,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        processed_at=datetime.now(UTC),
        type="span.model_request_end",
    )
    await prepared._record(event=event)  # pyright: ignore[reportPrivateUsage]

    async with db_session_factory() as s:
        rows = await usage_events.list_for_tenant(s, tenant_id=tenant.id)
    assert len(rows) == 1, "the bound recorder must write exactly one usage_events row"
    assert rows[0].managed_session_id == prepared.ma_session_id, (
        "the recorded managed_session_id must equal bind_session's returned ma_session_id"
    )


def test_prepared_turn_public_fields_exclude_recorder() -> None:
    field_names = {f.name for f in dataclasses.fields(PreparedTurn)}
    public_field_names = {name for name in field_names if not name.startswith("_")}
    assert "record" not in public_field_names, (
        "PreparedTurn's public field set must not expose the recorder"
    )
    assert public_field_names == {
        "admission",
        "ma_session_id",
        "mapping_id",
        "watermark",
        "reused",
        "session_account_id",
    }, "PreparedTurn must expose exactly the documented public fields"
