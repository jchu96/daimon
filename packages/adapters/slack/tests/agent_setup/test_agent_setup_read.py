"""Real-Postgres tests for agent_setup/read.py.

Three behaviors (security invariant + roster cap + scope-hint copy):
(a) Names-only secret hygiene: load_section_data(section="secrets") returns key
    NAMES only — the secret value never appears in the result.
(b) Roster cap-25 over_cap: load_tenant_roster caps at 25 entries when more
    agents exist in the tenant.
(c) Scope-hint copy: load_scope_hint returns the UI-SPEC copy per scope state
    (workspace-scope hit and unset case).
"""

from __future__ import annotations

import json
import secrets
import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from anthropic import AsyncAnthropic
from daimon.adapters.slack.agent_setup.read import (
    coding_tools_available,
    github_facts,
    load_panel_details,
    load_panel_roster,
    load_scope_hint,
    load_section_data,
    load_tenant_roster,
    resolve_attributions,
)
from daimon.adapters.slack.runtime import SlackRuntime
from daimon.core.defaults.metadata import MA_METADATA_KEY_NAME, MA_METADATA_KEY_TENANT
from daimon.core.defaults.provisioning import derive_guild_account_uuid
from daimon.core.ma_identity import derive_agent_uuid
from daimon.core.scope import ChannelScopeRef, DeploymentDefault, TenantScopeRef
from daimon.core.stores.agent_files import put_agent_file
from daimon.core.stores.scoped_config_write import set_fields
from daimon.core.stores.tenants import get_tenant
from daimon.testing import ma_agent
from daimon.testing.factories import make_account, make_platform_principal, make_tenant
from daimon.testing.ma import MARouter, build_fake_anthropic
from pydantic import HttpUrl, SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TEAM_ID = "T_READ_TESTS"
_AGENT_NAME = "read-test-agent"
_MA_AGENT_ID = f"agent_{'x' * 24}"


async def _seed_tenant(session: AsyncSession, team_id: str = _TEAM_ID) -> uuid.UUID:
    """Create a Tenant row and return the derived tenant_id."""
    tenant = await make_tenant(session, platform="slack", workspace_id=team_id)
    return tenant.id


async def _seed_account(session: AsyncSession, tenant_id: uuid.UUID) -> uuid.UUID:
    """Create an Account row and return its id."""
    tenant_row = await get_tenant(session, tenant_id)
    assert tenant_row is not None, "_seed_account requires a tenant seeded via _seed_tenant"
    account = await make_account(session, tenant=tenant_row)
    return account.id


def _make_agent_payload(
    *,
    tenant_id: uuid.UUID,
    name: str,
    ma_agent_id: str | None = None,
    model: str = "claude-sonnet-4-6",
) -> dict[str, Any]:
    """Build an MA agent payload for use in httpx.MockTransport responses."""
    now = datetime.now(UTC).isoformat()
    agent_id = (
        ma_agent_id or f"agent_{secrets.token_urlsafe(18).replace('-', '').replace('_', '')[:24]}"
    )
    return {
        "id": agent_id,
        "type": "agent",
        "name": name,
        "version": 1,
        "model": {"id": model, "speed": "standard"},
        "system": None,
        "metadata": {
            MA_METADATA_KEY_TENANT: str(tenant_id),
            MA_METADATA_KEY_NAME: name,
        },
        "mcp_servers": [],
        "tools": [],
        "skills": [],
        "created_at": now,
        "updated_at": now,
        "archived_at": None,
        "description": None,
    }


# ---------------------------------------------------------------------------
# (a) Names-only secret hygiene (the security invariant pyright cannot check)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_section_data_secrets_returns_key_names_only_and_value_is_absent(
    db_session: AsyncSession,
) -> None:
    """secrets section must return key names only — values never leave the read layer."""
    tenant_id = await _seed_tenant(db_session)
    _account_id = await _seed_account(db_session, tenant_id)

    # Derive the agent_uuid the read layer uses for agent_files lookup
    agent_uuid = derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=_MA_AGENT_ID)

    # Seed a secret with a known VALUE that must not appear in the result
    secret_key = "API_TOKEN"
    secret_value = "s3cr3t-value-should-never-appear"
    await put_agent_file(
        db_session,
        tenant_id=tenant_id,
        agent_id=agent_uuid,
        key=secret_key,
        content=secret_value,
        set_by_account_id=None,
    )

    # Build a fake MA handler that returns exactly one agent for this tenant
    agent_payload = _make_agent_payload(
        tenant_id=tenant_id, name=_AGENT_NAME, ma_agent_id=_MA_AGENT_ID
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/agents":
            return httpx.Response(
                200, json={"data": [agent_payload], "has_more": False, "next_page": None}
            )
        return httpx.Response(404, json={"error": "unhandled"})

    anthropic = build_fake_anthropic(handler)

    result = await load_section_data(
        db_session,
        anthropic,
        tenant_id=tenant_id,
        agent_name=_AGENT_NAME,
        section="secrets",
    )

    # The result must be a list of strings (key names only)
    assert isinstance(result, list), (
        "secrets section must return list of key names only — values never leave the read layer"
    )
    secret_names: list[str] = result  # type: ignore[assignment]
    assert secret_key in secret_names, "secrets section must include the key name 'API_TOKEN'"

    # The secret VALUE must never appear anywhere in the result
    serialized = repr(result)
    assert secret_value not in serialized, (
        "secrets section must return key names only — values never leave the read layer"
    )

    # Verify there is no 'value' or 'val' field on the result items (result is list[str])
    for item in secret_names:
        assert isinstance(item, str), (
            "secrets section items must be plain strings (key names), not objects with a value field"
        )

    # Negative check: the result should not contain the value even after json serialization
    json_serialized = json.dumps(secret_names)
    assert secret_value not in json_serialized, (
        "secrets section values must not appear in any serialized form of the result"
    )


# ---------------------------------------------------------------------------
# (b) Roster cap-25 over_cap count
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_tenant_roster_caps_at_25_and_over_cap_count_is_correct(
    db_session: AsyncSession,
) -> None:
    """static_select caps at 25 options; the surplus surfaces as over_cap."""
    tenant_id = await _seed_tenant(db_session)

    # Build 28 agents tagged with this tenant
    num_agents = 28
    agents = [
        _make_agent_payload(tenant_id=tenant_id, name=f"agent-{i:03d}") for i in range(num_agents)
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/agents":
            return httpx.Response(200, json={"data": agents, "has_more": False, "next_page": None})
        return httpx.Response(404, json={"error": "unhandled"})

    anthropic = build_fake_anthropic(handler)

    entries, over_cap = await load_tenant_roster(db_session, anthropic, tenant_id=tenant_id)

    assert len(entries) == 25, "static_select caps at 25 options; the surplus surfaces as over_cap"
    assert over_cap == 3, "over_cap must be the exact number of agents beyond the 25-option cap"


# ---------------------------------------------------------------------------
# (c) Scope-hint copy per UI-SPEC Copywriting Contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_load_scope_hint_returns_workspace_copy_when_tenant_scope_propagated(
    db_session: AsyncSession,
) -> None:
    """scope hint renders the UI-SPEC copy for each scope state."""
    tenant_id = await _seed_tenant(db_session)
    account_id = await _seed_account(db_session, tenant_id)
    channel_id = "C_HINT_TEST"

    # Seed a workspace-scope propagation
    scope = TenantScopeRef(tenant_id=tenant_id)
    await set_fields(
        db_session,
        scope=scope,
        tenant_id=tenant_id,
        agent_name=_AGENT_NAME,
        mode="agent",
        actor_account_id=account_id,
    )

    result = await load_scope_hint(
        db_session,
        tenant_id=tenant_id,
        agent_name=_AGENT_NAME,
        channel_id=channel_id,
    )

    assert result == ":globe_with_meridians: Set for *Whole workspace*", (
        "scope hint renders the UI-SPEC copy for each scope state"
    )


@pytest.mark.asyncio
async def test_load_scope_hint_returns_unset_copy_when_no_propagation_seeded(
    db_session: AsyncSession,
) -> None:
    """scope hint renders the unset copy when no propagation has been seeded."""
    tenant_id = await _seed_tenant(db_session)
    channel_id = "C_HINT_UNSET"

    result = await load_scope_hint(
        db_session,
        tenant_id=tenant_id,
        agent_name=_AGENT_NAME,
        channel_id=channel_id,
    )

    assert result == "_(no default set for this agent)_", (
        "scope hint renders the UI-SPEC copy for each scope state"
    )


# ---------------------------------------------------------------------------
# (d) The read-only panel's wrappers over the core reads
# ---------------------------------------------------------------------------

_PANEL_TEAM_ID = "T_PANEL_READ"
_PANEL_CHANNEL_ID = "C0PANEL1234"


def _panel_runtime(
    anthropic: AsyncAnthropic,
    sessionmaker: async_sessionmaker[AsyncSession],
    *,
    fallback_pat: str | None = "ghp_fallback",
    app_id: str | None = "12345",
    app_private_key: str | None = "-----BEGIN KEY-----",
    mcp_public_url: str | None = "https://mcp.example.com/mcp",
    mcp_jwt_secret: str | None = "signing-secret",
) -> SlackRuntime:
    """A runtime carrying only the settings the panel reads."""
    settings = MagicMock()
    settings.crypto.keys = ()
    settings.github.fallback_pat = SecretStr(fallback_pat) if fallback_pat else None
    settings.github.app_id = app_id
    settings.github.app_private_key = SecretStr(app_private_key) if app_private_key else None
    settings.mcp.public_url = HttpUrl(mcp_public_url) if mcp_public_url else None
    settings.mcp.jwt_secret = SecretStr(mcp_jwt_secret) if mcp_jwt_secret else None
    return SlackRuntime(
        settings=settings,
        anthropic=anthropic,
        sessionmaker=sessionmaker,
        billing_config=None,
        http_client=MagicMock(spec=httpx.AsyncClient),
        resolver_cache=MagicMock(),  # pyright: ignore[reportArgumentType]  # turn path not exercised
        turn_deps=MagicMock(),  # pyright: ignore[reportArgumentType]  # turn path not exercised
        deployment_default=DeploymentDefault(agent_name="daimon", environment_name="default"),
    )


def test_github_facts_reads_both_halves_of_the_app_identity() -> None:
    sessionmaker = MagicMock(spec=async_sessionmaker)
    anthropic = build_fake_anthropic(MARouter().dispatch)
    complete = github_facts(_panel_runtime(anthropic, sessionmaker))
    assert complete.has_fallback_pat is True, "a configured operator token is reported"
    assert complete.app_configured is True, "an app id plus a private key is a configured App"

    half = github_facts(_panel_runtime(anthropic, sessionmaker, app_private_key=None))
    assert half.app_configured is False, "an app id with no private key mints no token"

    bare = github_facts(_panel_runtime(anthropic, sessionmaker, fallback_pat=None))
    assert bare.has_fallback_pat is False, "no operator token is reported as none"


def test_coding_tools_available_needs_both_the_url_and_the_signing_secret() -> None:
    sessionmaker = MagicMock(spec=async_sessionmaker)
    anthropic = build_fake_anthropic(MARouter().dispatch)
    assert coding_tools_available(_panel_runtime(anthropic, sessionmaker)) is True, (
        "a URL and a signing secret together make a usable token"
    )
    assert (
        coding_tools_available(_panel_runtime(anthropic, sessionmaker, mcp_jwt_secret=None))
        is False
    ), "a token that cannot be signed is not available"
    assert (
        coding_tools_available(_panel_runtime(anthropic, sessionmaker, mcp_public_url=None))
        is False
    ), "a token with nowhere to connect is not available"


@pytest.mark.asyncio
async def test_load_panel_roster_marks_the_agent_answering_in_this_channel(
    db_session: AsyncSession,
) -> None:
    tenant_id = await _seed_tenant(db_session, team_id=_PANEL_TEAM_ID)
    here = ma_agent(id="ag_here", name="research-bot", tenant_id=tenant_id)
    elsewhere = ma_agent(id="ag_elsewhere", name="churn-explorer", tenant_id=tenant_id)
    await set_fields(
        db_session,
        scope=ChannelScopeRef(tenant_id=tenant_id, channel_id=_PANEL_CHANNEL_ID),
        tenant_id=tenant_id,
        agent_name="research-bot",
    )
    router = MARouter()
    router.add_agent_list(here, elsewhere)
    roster = await load_panel_roster(
        db_session,
        build_fake_anthropic(router.dispatch),
        tenant_id=tenant_id,
        channel_id=_PANEL_CHANNEL_ID,
        thread_id=None,
        default=DeploymentDefault(),
    )
    assert roster.answering is not None, "a channel with its own setting has an answering agent"
    assert roster.answering.name == "research-bot", "the channel's own setting decides"
    assert roster.answering.answering_tier == "channel", "and the tier says where it came from"
    assert [row.name for row in roster.rows] == ["research-bot", "churn-explorer"], (
        "the agent answering here sorts ahead of the rest"
    )


@pytest.mark.asyncio
async def test_load_panel_details_renders_the_channel_as_a_mention(
    db_session: AsyncSession, db_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    tenant_id = await _seed_tenant(db_session, team_id=f"{_PANEL_TEAM_ID}_DETAILS")
    agent = ma_agent(id="ag_details", name="research-bot", tenant_id=tenant_id)
    router = MARouter()
    router.add_agent(agent)
    router.add_agent_list(agent)
    anthropic = build_fake_anthropic(router.dispatch)
    runtime = _panel_runtime(anthropic, db_session_factory)
    roster = await load_panel_roster(
        db_session,
        anthropic,
        tenant_id=tenant_id,
        channel_id=_PANEL_CHANNEL_ID,
        thread_id=None,
        default=runtime.deployment_default,
    )
    details = await load_panel_details(
        db_session,
        anthropic,
        runtime,
        tenant_id=tenant_id,
        roster=roster,
        agent_name="research-bot",
        channel_id=_PANEL_CHANNEL_ID,
        thread_id=None,
        is_admin=False,
    )
    assert details is not None, "an agent in the roster resolves to its details"
    assert details.name == "research-bot", "the details describe the agent that was asked for"
    assert details.unrouted_note is not None, "nothing routes to this agent yet"
    assert f"<#{_PANEL_CHANNEL_ID}>" in details.unrouted_note, (
        "the routing request names the channel as a Slack mention, not a raw id"
    )


@pytest.mark.asyncio
async def test_load_panel_details_when_name_is_not_in_the_roster_returns_none(
    db_session: AsyncSession, db_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    tenant_id = await _seed_tenant(db_session, team_id=f"{_PANEL_TEAM_ID}_MISS")
    agent = ma_agent(id="ag_only", name="research-bot", tenant_id=tenant_id)
    router = MARouter()
    router.add_agent(agent)
    router.add_agent_list(agent)
    anthropic = build_fake_anthropic(router.dispatch)
    runtime = _panel_runtime(anthropic, db_session_factory)
    roster = await load_panel_roster(
        db_session,
        anthropic,
        tenant_id=tenant_id,
        channel_id=_PANEL_CHANNEL_ID,
        thread_id=None,
        default=runtime.deployment_default,
    )
    details = await load_panel_details(
        db_session,
        anthropic,
        runtime,
        tenant_id=tenant_id,
        roster=roster,
        agent_name="ghost-bot",
        channel_id=_PANEL_CHANNEL_ID,
        thread_id=None,
        is_admin=True,
    )
    assert details is None, (
        "a name this tenant's roster does not hold resolves to None, never to a fetch by id"
    )


@pytest.mark.asyncio
async def test_resolve_attributions_omits_non_slack_principals_and_the_workspace_stamp(
    db_session: AsyncSession,
) -> None:
    tenant = await make_tenant(db_session, platform="slack", workspace_id=f"{_PANEL_TEAM_ID}_ATTR")
    slack_account = await make_account(db_session, tenant=tenant)
    discord_account = await make_account(db_session, tenant=tenant)
    cli_account = await make_account(db_session, tenant=tenant)
    await make_platform_principal(
        db_session,
        platform="slack",
        external_id="U0SLACK123",
        tenant=tenant,
        account=slack_account,
    )
    await make_platform_principal(
        db_session,
        platform="discord",
        external_id="99887766",
        tenant=tenant,
        account=discord_account,
    )
    stamp_account_id = derive_guild_account_uuid(tenant.id)
    resolved = await resolve_attributions(
        db_session,
        tenant_id=tenant.id,
        account_ids=[slack_account.id, discord_account.id, cli_account.id, stamp_account_id],
    )
    assert resolved == {slack_account.id: "<@U0SLACK123>"}, (
        "only an account with a Slack identity here gets a mention; "
        "a Discord-only actor, an account with no principal, and the workspace stamp are omitted"
    )
