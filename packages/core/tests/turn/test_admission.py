"""Unit tests for daimon.core.turn.admission.admit -- the D-01 stage-one
chokepoint: identity -> config cascade -> missing-config bail -> MA
resolve+retrieve -> balance gate -> cap gate.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from anthropic.types.beta import BetaEnvironment, BetaManagedAgentsAgent
from anthropic.types.beta.beta_managed_agents_model_config import BetaManagedAgentsModelConfig
from daimon.core.billing import BillingConfig
from daimon.core.config import McpSettings
from daimon.core.defaults.metadata import MA_METADATA_KEY_NAME, MA_METADATA_KEY_TENANT
from daimon.core.ma_resolver import MAResolverMissError, new_resolver_cache
from daimon.core.scope import DeploymentDefault
from daimon.core.turn.admission import Admission, AdmissionDenied, MissingTurnConfigError, admit
from daimon.core.turn.deps import TurnDeps
from daimon.testing.ma import EMPTY_CLOUD_CONFIG, MARouter, build_fake_anthropic, list_response
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from daimon.testing.factories import (  # isort: skip
    make_ledger_entry,
    make_tenant,
    make_tenant_config,
    make_tenant_user_cap,
)

_NOW = datetime(2026, 7, 28, tzinfo=UTC)


def _agent(*, agent_id: str, name: str, tenant_id: uuid.UUID) -> BetaManagedAgentsAgent:
    now = datetime.now(UTC)
    return BetaManagedAgentsAgent(
        id=agent_id,
        type="agent",
        name=name,
        version=1,
        model=BetaManagedAgentsModelConfig(id="claude-sonnet-4-6", speed="standard"),
        system=None,
        description=None,
        metadata={
            MA_METADATA_KEY_TENANT: str(tenant_id),
            MA_METADATA_KEY_NAME: name,
        },
        mcp_servers=[],
        tools=[],
        skills=[],
        created_at=now,
        updated_at=now,
        archived_at=None,
    )


def _env(*, env_id: str, name: str, tenant_id: uuid.UUID) -> BetaEnvironment:
    now_iso = datetime.now(UTC).isoformat()
    return BetaEnvironment(
        id=env_id,
        type="environment",
        name=name,
        description="",
        config=EMPTY_CLOUD_CONFIG,
        metadata={
            MA_METADATA_KEY_TENANT: str(tenant_id),
            MA_METADATA_KEY_NAME: name,
        },
        created_at=now_iso,
        updated_at=now_iso,
        archived_at=None,
    )


def _agent_payload(agent: BetaManagedAgentsAgent) -> dict[str, object]:
    return agent.model_dump(mode="json")


def _env_payload(env: BetaEnvironment) -> dict[str, object]:
    return env.model_dump(mode="json")


def _resolved_router(*, tenant_id: uuid.UUID) -> MARouter:
    """Router whose LIST routes resolve a live agent/environment for tenant_id."""
    agent = _agent(agent_id="ag_1", name="daimon", tenant_id=tenant_id)
    env = _env(env_id="env_1", name="default", tenant_id=tenant_id)
    router = MARouter()
    router.add("GET", r"/v1/agents", lambda req, _m: list_response([_agent_payload(agent)]))
    router.add(
        "GET", r"/v1/agents/ag_1", lambda req, _m: httpx.Response(200, json=_agent_payload(agent))
    )
    router.add("GET", r"/v1/environments", lambda req, _m: list_response([_env_payload(env)]))
    router.add(
        "GET",
        r"/v1/environments/env_1",
        lambda req, _m: httpx.Response(200, json=_env_payload(env)),
    )
    return router


def _billing_config() -> BillingConfig:
    return BillingConfig(
        secret_key=SecretStr("sk_test"),
        webhook_secret=SecretStr("whsec_test"),
        prices={10: "price_10", 25: "price_25", 50: "price_50", 100: "price_100"},
        success_url="https://example.com/billing/success",
        cancel_url="https://example.com/billing/cancel",
    )


def _deps(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    defaults_root: Path,
    router: MARouter,
    billing_config: BillingConfig | None = None,
) -> TurnDeps:
    return TurnDeps(
        anthropic=build_fake_anthropic(router.dispatch),
        sessionmaker=sessionmaker,
        deployment_default=DeploymentDefault(),
        resolver_cache=new_resolver_cache(),
        defaults_root=defaults_root,
        mcp=McpSettings(),
        billing_config=billing_config,
        markup=Decimal("1.0"),
        fernet=None,
        github_fallback_pat=None,
        github_app_id=None,
        github_app_private_key=None,
        public_url=None,
    )


async def test_admit_over_balance_tenant_raises_admission_denied_balance_depleted(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    tenant = await make_tenant(db_session)
    await make_tenant_config(
        db_session, tenant=tenant, agent_name="daimon", environment_name="default"
    )
    await db_session.commit()
    # No ledger entry seeded -> balance is 0 -> over-balance.

    router = _resolved_router(tenant_id=tenant.id)
    deps = _deps(
        sessionmaker=db_session_factory,
        defaults_root=tmp_path,
        router=router,
    )

    with pytest.raises(AdmissionDenied) as exc_info:
        await admit(
            deps,
            tenant_id=tenant.id,
            platform="discord",
            external_user_id="user-1",
            channel_id="chan-1",
            now=_NOW,
        )

    assert exc_info.value.reason == "balance_depleted", (
        "an over-balance tenant must raise AdmissionDenied(reason='balance_depleted')"
    )


async def test_admit_over_cap_user_raises_admission_denied_cap_exceeded(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    tenant = await make_tenant(db_session)
    await make_tenant_config(
        db_session, tenant=tenant, agent_name="daimon", environment_name="default"
    )
    await make_ledger_entry(db_session, tenant=tenant, delta_usd=Decimal("10"))
    await make_tenant_user_cap(db_session, tenant=tenant, amount=Decimal("0"))
    await db_session.commit()

    router = _resolved_router(tenant_id=tenant.id)
    deps = _deps(
        sessionmaker=db_session_factory,
        defaults_root=tmp_path,
        router=router,
        billing_config=_billing_config(),
    )

    with pytest.raises(AdmissionDenied) as exc_info:
        await admit(
            deps,
            tenant_id=tenant.id,
            platform="discord",
            external_user_id="user-1",
            channel_id="chan-1",
            now=_NOW,
        )

    assert exc_info.value.reason == "cap_exceeded", (
        "an over-cap user must raise AdmissionDenied(reason='cap_exceeded')"
    )


async def test_admit_missing_agent_only_raises_missing_turn_config_error(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    tenant = await make_tenant(db_session)
    await make_tenant_config(db_session, tenant=tenant, agent_name=None, environment_name="default")
    await db_session.commit()

    deps = _deps(
        sessionmaker=db_session_factory,
        defaults_root=tmp_path,
        router=MARouter(),
    )

    with pytest.raises(MissingTurnConfigError) as exc_info:
        await admit(
            deps,
            tenant_id=tenant.id,
            platform="discord",
            external_user_id="user-1",
            channel_id="chan-1",
            now=_NOW,
        )

    assert exc_info.value.missing == ("agent",), (
        "channel with only agent unresolved must report missing == ('agent',)"
    )


async def test_admit_missing_agent_and_environment_raises_missing_turn_config_error(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    tenant = await make_tenant(db_session)
    await db_session.commit()  # no tenant config row at all -> both unresolved

    deps = _deps(
        sessionmaker=db_session_factory,
        defaults_root=tmp_path,
        router=MARouter(),
    )

    with pytest.raises(MissingTurnConfigError) as exc_info:
        await admit(
            deps,
            tenant_id=tenant.id,
            platform="discord",
            external_user_id="user-1",
            channel_id="chan-1",
            now=_NOW,
        )

    assert exc_info.value.missing == ("agent", "environment"), (
        "channel with both unresolved must report missing == ('agent', 'environment')"
    )


async def test_admit_unresolvable_tag_propagates_ma_resolver_miss_error(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """A configured tag with no live MA resource and an empty defaults tree
    (apply_callable is a no-op) must raise MAResolverMissError unwrapped."""
    tenant = await make_tenant(db_session)
    await make_tenant_config(
        db_session, tenant=tenant, agent_name="ghost-agent", environment_name="ghost-env"
    )
    await db_session.commit()

    router = MARouter()
    router.add("GET", r"/v1/agents", lambda req, _m: list_response([]))
    router.add("GET", r"/v1/environments", lambda req, _m: list_response([]))

    deps = _deps(
        sessionmaker=db_session_factory,
        defaults_root=tmp_path,  # empty tree: apply_callable is a real no-op
        router=router,
    )

    with pytest.raises(MAResolverMissError) as exc_info:
        await admit(
            deps,
            tenant_id=tenant.id,
            platform="discord",
            external_user_id="user-1",
            channel_id="chan-1",
            now=_NOW,
        )

    assert exc_info.value.kind == "agent", (
        "agent is resolved before environment; the miss must surface for 'agent' first"
    )


async def test_admit_happy_path_returns_admission_with_retrieved_agent_and_account(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    tenant = await make_tenant(db_session)
    await make_tenant_config(
        db_session, tenant=tenant, agent_name="daimon", environment_name="default"
    )
    await make_ledger_entry(db_session, tenant=tenant, delta_usd=Decimal("10"))
    await db_session.commit()

    router = _resolved_router(tenant_id=tenant.id)
    deps = _deps(
        sessionmaker=db_session_factory,
        defaults_root=tmp_path,
        router=router,
    )

    admission = await admit(
        deps,
        tenant_id=tenant.id,
        platform="discord",
        external_user_id="user-1",
        channel_id="chan-1",
        now=_NOW,
    )

    assert isinstance(admission, Admission), "admit() must return an Admission on the happy path"
    assert admission.agent.name == "daimon", "agent.name must come from a real agents.retrieve"
    assert admission.agent.model.id == "claude-sonnet-4-6", (
        "agent.model.id must come from a real agents.retrieve"
    )
    async with db_session_factory() as s:
        from daimon.core.stores.identity import get_or_create_platform_principal

        principal = await get_or_create_platform_principal(
            s, tenant_id=tenant.id, platform="discord", external_id="user-1"
        )
    assert admission.account_id == principal.account_id, (
        "account_id must match the principal resolved for this (platform, external_id)"
    )


async def test_admit_gate_order_config_bail_wins_over_over_balance(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    """A tenant that is both over-balance AND mis-configured must see the
    config error -- config bail runs before the balance gate."""
    tenant = await make_tenant(db_session)
    # No tenant config row at all -> both agent/environment unresolved.
    # No ledger entry -> also over-balance.
    await db_session.commit()

    deps = _deps(
        sessionmaker=db_session_factory,
        defaults_root=tmp_path,
        router=MARouter(),
    )

    with pytest.raises(MissingTurnConfigError):
        await admit(
            deps,
            tenant_id=tenant.id,
            platform="discord",
            external_user_id="user-1",
            channel_id="chan-1",
            now=_NOW,
        )
