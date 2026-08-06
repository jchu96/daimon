"""Tests for the chat-initiated repo bind's click-time authorization gate.

`_admin_interaction` / `_member_interaction` are copied verbatim from
`tests/agent_setup/test_authz.py` rather than imported: that module's package
(`agent_setup/`, which carries an `__init__.py`) sits one level below
`tests/`, which carries none, so pytest's `--import-mode=importlib` only
resolves `agent_setup.test_authz` as an importable dotted name once
`agent_setup/test_authz.py` has itself already been collected in the same
session. Running this file alone (as the plan's own verify command does)
raises `ModuleNotFoundError` for that import, so a copy — not an import — is
what actually works standalone. Both builders build a `MagicMock(spec=discord.Member)`
so `isinstance(user, discord.Member)` holds and the admin/member split is real,
never a bare `MagicMock()`.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import discord
import httpx
import pytest
from anthropic.types.beta import BetaManagedAgentsAgent
from daimon.adapters.discord.agent_setup import authz as panel_authz
from daimon.adapters.discord.agent_setup.state import RosterEntry
from daimon.adapters.discord.credential_repo_bind import (
    _AGENT_GONE_MESSAGE,
    _SHARED_AGENT_MESSAGE,
    _WRONG_GUILD_MESSAGE,
    refuse_if_shared_and_not_admin_for_request,
)
from daimon.adapters.discord.runtime import DiscordRuntime
from daimon.core.ma_identity import derive_agent_uuid
from daimon.core.ma_resolver import new_resolver_cache
from daimon.core.notebooks._rate_limit import RateLimiter
from daimon.core.scope import DeploymentDefault, TenantScopeRef
from daimon.core.specs import AgentSpec
from daimon.core.stores import scoped_config_write
from daimon.testing.factories import make_tenant
from daimon.testing.ma import build_fake_anthropic, list_response
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

pytestmark = pytest.mark.asyncio

# `make_tenant` derives the tenant id from `workspace_id` the exact same way
# `derive_tenant_uuid(platform="discord", workspace_id=str(interaction.guild_id))`
# does — see `daimon.testing.factories.make_tenant` and
# `daimon.core.ma_identity.derive_tenant_uuid`. Every tenant seeded below and
# every interaction built below must agree on this one id, or a refusal test
# lands on the wrong-guild branch and passes for the wrong reason. The one
# deliberate exception is the wrong-guild test itself, which seeds a distinct
# workspace_id on purpose.
_GUILD_ID = 111


def _admin_interaction(*, guild_id: int = _GUILD_ID, acked: bool = False) -> MagicMock:
    interaction = MagicMock()
    interaction.guild_id = guild_id
    interaction.user = MagicMock(spec=discord.Member)
    interaction.user.id = 1
    interaction.user.guild_permissions.administrator = True
    interaction.user.guild_permissions.manage_guild = False
    interaction.guild.owner_id = 999
    interaction.response.is_done.return_value = acked
    interaction.response.send_message = AsyncMock()
    interaction.followup.send = AsyncMock()
    return interaction


def _member_interaction(*, guild_id: int = _GUILD_ID, acked: bool = False) -> MagicMock:
    interaction = MagicMock()
    interaction.guild_id = guild_id
    interaction.user = MagicMock(spec=discord.Member)
    interaction.user.id = 2
    interaction.user.guild_permissions.administrator = False
    interaction.user.guild_permissions.manage_guild = False
    interaction.guild.owner_id = 999
    interaction.response.is_done.return_value = acked
    interaction.response.send_message = AsyncMock()
    interaction.followup.send = AsyncMock()
    return interaction


def _runtime(
    *,
    sessionmaker: object,
    anthropic: object,
    deployment_default: DeploymentDefault | None = None,
) -> DiscordRuntime:
    return DiscordRuntime(
        settings=MagicMock(),
        anthropic=anthropic,  # type: ignore[arg-type]  # a real fake AsyncAnthropic, never a bare mock
        sessionmaker=sessionmaker,  # type: ignore[arg-type]  # a real async_sessionmaker or a spy MagicMock, never invoked as a client
        notebook_rate_limiter=RateLimiter(max_requests=999),
        billing_config=None,
        deployment_default=deployment_default or DeploymentDefault(),
        resolver_cache=new_resolver_cache(),
        turn_deps=MagicMock(),  # pyright: ignore[reportArgumentType]  # never runs a turn
    )


def _make_agent(
    *, ma_agent_id: str, tenant_id: uuid.UUID, name: str, managed: bool
) -> BetaManagedAgentsAgent:
    metadata = {"daimon_tenant": str(tenant_id)}
    if managed:
        metadata["daimon_managed"] = "true"
    return BetaManagedAgentsAgent(
        id=ma_agent_id,
        type="agent",
        name=name,
        model={"id": "claude-sonnet-4-6"},
        metadata=metadata,
        description=None,
        created_at="2026-04-21T00:00:00Z",
        updated_at="2026-04-21T00:00:00Z",
        version=1,
        mcp_servers=[],
        skills=[],
        tools=[],
        system=None,
    )


def _list_agents_handler(
    agents: list[BetaManagedAgentsAgent],
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return list_response([agent.model_dump(mode="json") for agent in agents])

    return handler


def _counting_handler(
    agents: list[BetaManagedAgentsAgent], *, calls: list[httpx.Request]
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return list_response([agent.model_dump(mode="json") for agent in agents])

    return handler


def _sent_message(interaction: MagicMock) -> Any:  # noqa: ANN401 -- MagicMock call_args positional arg is untyped by construction
    """Return the ephemeral message text an interaction was sent, on whichever half fired."""
    if interaction.response.send_message.called:
        return interaction.response.send_message.call_args.args[0]
    return interaction.followup.send.call_args.args[0]


async def test_defaults_managed_target_member_refuses_with_shared_agent_message(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    ma_agent_id = "agent_seeded_1"
    async with db_session_factory() as session, session.begin():
        tenant = await make_tenant(session, platform="discord", workspace_id=str(_GUILD_ID))
    agent_id = derive_agent_uuid(tenant_id=tenant.id, ma_agent_id=ma_agent_id)
    agent = _make_agent(ma_agent_id=ma_agent_id, tenant_id=tenant.id, name="daimon", managed=True)
    client = build_fake_anthropic(_list_agents_handler([agent]))
    runtime = _runtime(sessionmaker=db_session_factory, anthropic=client)
    interaction = _member_interaction()

    refused = await refuse_if_shared_and_not_admin_for_request(
        interaction, runtime=runtime, tenant_id=tenant.id, agent_id=agent_id
    )

    assert refused is True, "a member must not bind a repo to a defaults-managed agent"
    assert _sent_message(interaction) == _SHARED_AGENT_MESSAGE


async def test_reachable_non_managed_target_flips_with_the_scope_row(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Branches (d) and (e) share one refusal message, so text alone cannot tell
    them apart. Running the same fixture with and without the scope row that
    makes the agent reachable, and asserting the result flips, is the only way
    to pin branch (e) specifically — deleting it would leave this test green
    with the scope row removed unless the flip is checked both ways."""
    ma_agent_id = "agent_reachable_1"
    agent_name = "bot"
    async with db_session_factory() as session, session.begin():
        tenant = await make_tenant(session, platform="discord", workspace_id=str(_GUILD_ID))
    agent_id = derive_agent_uuid(tenant_id=tenant.id, ma_agent_id=ma_agent_id)
    agent = _make_agent(
        ma_agent_id=ma_agent_id, tenant_id=tenant.id, name=agent_name, managed=False
    )
    client = build_fake_anthropic(_list_agents_handler([agent]))
    runtime = _runtime(sessionmaker=db_session_factory, anthropic=client)

    async with db_session_factory() as session, session.begin():
        await scoped_config_write.set_fields(
            session,
            scope=TenantScopeRef(tenant_id=tenant.id),
            tenant_id=tenant.id,
            agent_name=agent_name,
        )
    scoped_interaction = _member_interaction()
    refused_when_reachable = await refuse_if_shared_and_not_admin_for_request(
        scoped_interaction, runtime=runtime, tenant_id=tenant.id, agent_id=agent_id
    )
    assert refused_when_reachable is True, "the tenant-scoped default agent must refuse a member"
    assert _sent_message(scoped_interaction) == _SHARED_AGENT_MESSAGE

    async with db_session_factory() as session, session.begin():
        await scoped_config_write.unset_fields(
            session, scope=TenantScopeRef(tenant_id=tenant.id), fields=["agent_name"]
        )
    unscoped_interaction = _member_interaction()
    refused_when_unreachable = await refuse_if_shared_and_not_admin_for_request(
        unscoped_interaction, runtime=runtime, tenant_id=tenant.id, agent_id=agent_id
    )
    assert refused_when_unreachable is False, (
        "removing the only scope row naming this agent must free it for its own member"
    )
    assert refused_when_reachable != refused_when_unreachable, (
        "only branch (e), reachability, can produce this flip — a deleted branch (e) "
        "would leave both calls returning True"
    )


async def test_non_managed_non_reachable_target_member_allowed_no_ephemeral(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    ma_agent_id = "agent_private_1"
    async with db_session_factory() as session, session.begin():
        tenant = await make_tenant(session, platform="discord", workspace_id=str(_GUILD_ID))
    agent_id = derive_agent_uuid(tenant_id=tenant.id, ma_agent_id=ma_agent_id)
    agent = _make_agent(ma_agent_id=ma_agent_id, tenant_id=tenant.id, name="bot", managed=False)
    client = build_fake_anthropic(_list_agents_handler([agent]))
    runtime = _runtime(sessionmaker=db_session_factory, anthropic=client)
    interaction = _member_interaction()

    refused = await refuse_if_shared_and_not_admin_for_request(
        interaction, runtime=runtime, tenant_id=tenant.id, agent_id=agent_id
    )

    assert refused is False, "a private, unreachable agent is the member's own to bind a repo to"
    interaction.response.send_message.assert_not_called()
    interaction.followup.send.assert_not_called()


async def test_admin_passes_on_defaults_managed_target_before_any_ma_request(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Ordering pin: reordering the admin branch after the defaults-managed
    branch must turn this red. Verified empirically by temporarily swapping
    them (see summary)."""
    ma_agent_id = "agent_seeded_2"
    async with db_session_factory() as session, session.begin():
        tenant = await make_tenant(session, platform="discord", workspace_id=str(_GUILD_ID))
    agent_id = derive_agent_uuid(tenant_id=tenant.id, ma_agent_id=ma_agent_id)
    agent = _make_agent(ma_agent_id=ma_agent_id, tenant_id=tenant.id, name="daimon", managed=True)
    calls: list[httpx.Request] = []
    client = build_fake_anthropic(_counting_handler([agent], calls=calls))
    runtime = _runtime(sessionmaker=db_session_factory, anthropic=client)
    interaction = _admin_interaction()

    refused = await refuse_if_shared_and_not_admin_for_request(
        interaction, runtime=runtime, tenant_id=tenant.id, agent_id=agent_id
    )

    assert refused is False, "a live admin must be able to bind a repo to the seeded agent"
    assert len(calls) == 0, "the admin branch must precede any MA request"
    interaction.response.send_message.assert_not_called()
    interaction.followup.send.assert_not_called()


async def test_unknown_agent_uuid_refuses_with_agent_gone_message(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with db_session_factory() as session, session.begin():
        tenant = await make_tenant(session, platform="discord", workspace_id=str(_GUILD_ID))
    client = build_fake_anthropic(_list_agents_handler([]))
    runtime = _runtime(sessionmaker=db_session_factory, anthropic=client)
    interaction = _member_interaction()

    refused = await refuse_if_shared_and_not_admin_for_request(
        interaction, runtime=runtime, tenant_id=tenant.id, agent_id=uuid.uuid4()
    )

    assert refused is True, "a derived agent uuid matching no MA agent must fail closed"
    assert _sent_message(interaction) == _AGENT_GONE_MESSAGE


async def test_wrong_guild_refuses_before_the_admin_check(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Using an admin interaction here is the point: it proves the tenant
    re-derivation precedes even the admin short-circuit."""
    seeded_workspace_id = "555001"
    async with db_session_factory() as session, session.begin():
        tenant = await make_tenant(session, platform="discord", workspace_id=seeded_workspace_id)
    calls: list[httpx.Request] = []
    client = build_fake_anthropic(_counting_handler([], calls=calls))
    runtime = _runtime(sessionmaker=db_session_factory, anthropic=client)
    interaction = _admin_interaction(guild_id=_GUILD_ID)

    refused = await refuse_if_shared_and_not_admin_for_request(
        interaction, runtime=runtime, tenant_id=tenant.id, agent_id=uuid.uuid4()
    )

    assert refused is True, "a request row minted for a different guild's tenant must refuse"
    assert _sent_message(interaction) == _WRONG_GUILD_MESSAGE
    assert len(calls) == 0, "the tenant check must precede every MA request, admin or not"


async def test_unacked_refusal_uses_response_send_message(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    ma_agent_id = "agent_seeded_3"
    async with db_session_factory() as session, session.begin():
        tenant = await make_tenant(session, platform="discord", workspace_id=str(_GUILD_ID))
    agent_id = derive_agent_uuid(tenant_id=tenant.id, ma_agent_id=ma_agent_id)
    agent = _make_agent(ma_agent_id=ma_agent_id, tenant_id=tenant.id, name="daimon", managed=True)
    client = build_fake_anthropic(_list_agents_handler([agent]))
    runtime = _runtime(sessionmaker=db_session_factory, anthropic=client)
    interaction = _member_interaction(acked=False)

    refused = await refuse_if_shared_and_not_admin_for_request(
        interaction, runtime=runtime, tenant_id=tenant.id, agent_id=agent_id
    )

    assert refused is True
    interaction.response.send_message.assert_called_once()
    interaction.followup.send.assert_not_called()


async def test_acked_refusal_uses_followup_send(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    ma_agent_id = "agent_seeded_4"
    async with db_session_factory() as session, session.begin():
        tenant = await make_tenant(session, platform="discord", workspace_id=str(_GUILD_ID))
    agent_id = derive_agent_uuid(tenant_id=tenant.id, ma_agent_id=ma_agent_id)
    agent = _make_agent(ma_agent_id=ma_agent_id, tenant_id=tenant.id, name="daimon", managed=True)
    client = build_fake_anthropic(_list_agents_handler([agent]))
    runtime = _runtime(sessionmaker=db_session_factory, anthropic=client)
    interaction = _member_interaction(acked=True)

    refused = await refuse_if_shared_and_not_admin_for_request(
        interaction, runtime=runtime, tenant_id=tenant.id, agent_id=agent_id
    )

    assert refused is True
    interaction.response.send_message.assert_not_called()
    interaction.followup.send.assert_called_once()


async def test_shared_agent_message_matches_the_panel_gate_character_for_character() -> None:
    assert _SHARED_AGENT_MESSAGE == panel_authz._SHARED_AGENT_MESSAGE, (
        "the chat gate's shared-agent refusal copy must never drift from the panel's"
    )


@pytest.mark.parametrize(
    ("is_admin", "is_system", "reachable"),
    [
        (is_admin, is_system, reachable)
        for is_admin in (True, False)
        for is_system in (True, False)
        for reachable in (True, False)
    ],
)
async def test_decision_table_parity_with_the_panel_gate(
    db_session_factory: async_sessionmaker[AsyncSession],
    is_admin: bool,
    is_system: bool,
    reachable: bool,
) -> None:
    """Parity is between the two DECISIONS, not the two call sites: the chat
    path additionally runs this gate as a pre-filter before a modal opens
    (added by the next plan), and the panel's own repo-bind path gates at
    open and at submit too — so even that shape matches — but this test's
    scope is only the shared boolean.

    Each side's own fail-closed guards (a missing roster entry on the panel
    side; a wrong guild or an unresolvable agent on the chat side) are inputs
    the other side does not have, and are held out of the comparison by
    construction: the panel side always gets a real entry, and the chat side
    always gets a matching tenant and a resolvable agent.
    """
    ma_agent_id = "agent_parity"
    agent_name = "bot"
    async with db_session_factory() as session, session.begin():
        tenant = await make_tenant(session, platform="discord", workspace_id=str(_GUILD_ID))
    agent_id = derive_agent_uuid(tenant_id=tenant.id, ma_agent_id=ma_agent_id)
    agent = _make_agent(
        ma_agent_id=ma_agent_id, tenant_id=tenant.id, name=agent_name, managed=is_system
    )
    client = build_fake_anthropic(_list_agents_handler([agent]))
    if reachable:
        async with db_session_factory() as session, session.begin():
            await scoped_config_write.set_fields(
                session,
                scope=TenantScopeRef(tenant_id=tenant.id),
                tenant_id=tenant.id,
                agent_name=agent_name,
            )
    runtime = _runtime(sessionmaker=db_session_factory, anthropic=client)

    chat_interaction = _admin_interaction() if is_admin else _member_interaction()
    chat_refused = await refuse_if_shared_and_not_admin_for_request(
        chat_interaction, runtime=runtime, tenant_id=tenant.id, agent_id=agent_id
    )

    panel_interaction = _admin_interaction() if is_admin else _member_interaction()
    entry = RosterEntry(
        name=agent_name,
        model="claude-sonnet-4-6",
        spec=AgentSpec(name=agent_name, model="claude-sonnet-4-6"),
        is_system=is_system,
    )
    panel_refused = await panel_authz.refuse_if_shared_and_not_admin(
        panel_interaction, runtime=runtime, entry=entry
    )

    assert chat_refused == panel_refused, (
        f"decision mismatch for admin={is_admin} system={is_system} reachable={reachable}: "
        f"chat gate returned {chat_refused}, panel gate returned {panel_refused}"
    )
