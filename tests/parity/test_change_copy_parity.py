"""Scenario: the Discord adapter's configuration-change acks are the core
renderer's own output, byte-for-byte -- never adapter-side hand-written copy.

`daimon.core.continuity.messages.render_change_confirmation` is the single
place person-facing configuration-change copy is written; every adapter ack
site is required to call it rather than compose its own string. This file
drives three representative Discord entry points -- one per `ConfigurationChange`
axis (a plain key add, a bulk paste, and a removal) spanning both modules that
own acks (`credential_modals.py`, `agent_setup/credentials.py`) -- and asserts
the posted text equals a fresh, independent call to the renderer with the same
fields. The remaining ack sites (model/instructions, mcp/mcp_removed,
skill_removed, repo) are covered the same way as unit tests colocated with
each adapter module (`packages/adapters/discord/tests/...`); this file is the
cross-adapter parity anchor, not the exhaustive sweep.

Each test function name carries `discord` so the Slack half of this suite
(added separately, to the same file) never collides with these.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import discord
from daimon.adapters.discord.agent_setup.credentials import CredentialsSubView, PasteSecretModal
from daimon.adapters.discord.agent_setup.state import PanelState, RosterEntry
from daimon.adapters.discord.credential_modals import EnvCredentialModal
from daimon.adapters.discord.runtime import DiscordRuntime
from daimon.core.continuity.messages import ConfigurationChange, render_change_confirmation
from daimon.core.credential_requests import mint_request_token
from daimon.core.ma_identity import derive_agent_uuid
from daimon.core.ma_resolver import new_resolver_cache
from daimon.core.notebooks._rate_limit import RateLimiter
from daimon.core.scope import DeploymentDefault
from daimon.core.specs import AgentSpec
from daimon.core.stores.agent_files import list_agent_files
from daimon.core.stores.credential_requests import create_credential_request
from daimon.testing.factories import make_tenant
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class _AsyncIter:
    def __init__(self, items: list[object]) -> None:
        self._items = iter(items)

    def __aiter__(self) -> _AsyncIter:
        return self

    async def __anext__(self) -> object:
        try:
            return next(self._items)
        except StopIteration as err:
            raise StopAsyncIteration from err


def _make_runtime(
    sessionmaker: async_sessionmaker[AsyncSession], *, agents: list[object] | None = None
) -> DiscordRuntime:
    settings = MagicMock()
    settings.mcp.public_url = None
    agent_list = list(agents or [])
    anthropic = AsyncMock()

    # A fresh iterator per call: this stub client's agent list is walked more
    # than once per submit (the target-availability gate, then the ack's own
    # name lookup), and a single shared iterator would starve the second walk.
    def _list_agents(**_kwargs: object) -> _AsyncIter:
        return _AsyncIter(list(agent_list))

    anthropic.beta.agents.list = MagicMock(side_effect=_list_agents)
    return DiscordRuntime(
        settings=settings,
        anthropic=anthropic,
        sessionmaker=sessionmaker,
        notebook_rate_limiter=RateLimiter(max_requests=999),
        billing_config=None,
        deployment_default=DeploymentDefault(),
        resolver_cache=new_resolver_cache(),
        turn_deps=MagicMock(),  # pyright: ignore[reportArgumentType]  # never runs a turn
    )


def _make_ma_agent(ma_agent_id: str, *, name: str, tenant_id: uuid.UUID) -> object:
    from anthropic.types.beta import BetaManagedAgentsAgent
    from anthropic.types.beta.beta_managed_agents_model_config import (
        BetaManagedAgentsModelConfig,
    )
    from daimon.core.defaults.metadata import MA_METADATA_KEY_NAME, MA_METADATA_KEY_TENANT

    return BetaManagedAgentsAgent(
        id=ma_agent_id,
        type="agent",
        name=name,
        model=BetaManagedAgentsModelConfig(id="claude-sonnet-4-6"),
        metadata={MA_METADATA_KEY_TENANT: str(tenant_id), MA_METADATA_KEY_NAME: name},
        description=None,
        created_at="2026-06-14T00:00:00Z",  # pyright: ignore[reportArgumentType]
        updated_at="2026-06-14T00:00:00Z",  # pyright: ignore[reportArgumentType]
        version=1,
        mcp_servers=[],
        skills=[],
        tools=[],
        system=None,
    )


def _interaction(*, user_id: int = 100000000000000001, guild_id: int | None = None) -> MagicMock:
    """A live guild-admin interaction by default -- every write below routes
    through `refuse_if_shared_and_not_admin`, which admits a guild admin
    without any tenant read."""
    interaction = MagicMock()
    interaction.guild_id = guild_id
    interaction.user = MagicMock(spec=discord.Member)
    interaction.user.id = user_id
    interaction.user.guild_permissions.administrator = True
    interaction.user.guild_permissions.manage_guild = False
    interaction.response.defer = AsyncMock()
    interaction.followup.send = AsyncMock()
    interaction.edit_original_response = AsyncMock()
    return interaction


async def test_discord_env_key_add_ack_matches_core_renderer(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    workspace_id = "800000001"
    requester_user_id = 100000000000000042
    ma_agent_id = "ag_parity_copy"
    token = mint_request_token()
    async with db_session_factory() as session, session.begin():
        tenant = await make_tenant(session, platform="discord", workspace_id=workspace_id)
        row = await create_credential_request(
            session,
            token=token,
            kind="env",
            tenant_id=tenant.id,
            agent_id=derive_agent_uuid(tenant_id=tenant.id, ma_agent_id=ma_agent_id),
            account_id=uuid.uuid4(),
            target="STRIPE_KEY",
            mcp_server_url=None,
            requester_platform_user_id=str(requester_user_id),
            channel_id="chan-1",
            expires_at=datetime.now(UTC) + timedelta(minutes=30),
        )

    agent = _make_ma_agent(ma_agent_id, name="stripe-bot", tenant_id=tenant.id)
    runtime = _make_runtime(db_session_factory, agents=[agent])
    modal = EnvCredentialModal(runtime=runtime, request_row=row)
    modal.value_input._value = "sk_live_do_not_leak"  # pyright: ignore[reportPrivateUsage]

    interaction = _interaction(user_id=requester_user_id, guild_id=int(workspace_id))
    await modal.on_submit(interaction)

    posted = interaction.followup.send.call_args.args[0]
    expected = render_change_confirmation(
        ConfigurationChange(
            target_name="stripe-bot", kind="key", availability="next_message", detail="STRIPE_KEY"
        )
    )
    assert posted == expected, f"expected the renderer's own copy, got {posted!r}"

    async with db_session_factory() as session:
        stored = await list_agent_files(session, tenant_id=row.tenant_id, agent_id=row.agent_id)
    assert [f.key for f in stored] == ["STRIPE_KEY"], "sanity: the key write actually happened"


async def test_discord_paste_keys_bulk_add_ack_matches_core_renderer(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with db_session_factory() as session, session.begin():
        tenant = await make_tenant(session, platform="discord", workspace_id="800000002")
    tenant_id = tenant.id
    agent_id = uuid.uuid4()
    entry = RosterEntry(
        name="research-bot",
        model="claude-sonnet-4-6",
        spec=AgentSpec(name="research-bot", model="claude-sonnet-4-6"),
    )
    runtime = _make_runtime(db_session_factory)
    on_added = AsyncMock()
    modal = PasteSecretModal(
        runtime=runtime, tenant_id=tenant_id, agent_id=agent_id, entry=entry, on_added=on_added
    )
    modal.content_input._value = (  # pyright: ignore[reportPrivateUsage]
        "XERO_API_KEY=abc123\nTOGGL_TOKEN=xyz789\n"
    )

    interaction = _interaction()
    await modal.on_submit(interaction)

    posted = interaction.followup.send.call_args.args[0]
    expected = render_change_confirmation(
        ConfigurationChange(
            target_name="research-bot", kind="keys_bulk", availability="next_message", count=2
        )
    )
    assert posted == expected, f"expected the renderer's own copy, got {posted!r}"


async def test_discord_key_remove_ack_matches_core_renderer(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    tenant_id = uuid.uuid4()
    agent_id = uuid.uuid4()
    entry = RosterEntry(
        name="ops-bot",
        model="claude-sonnet-4-6",
        spec=AgentSpec(name="ops-bot", model="claude-sonnet-4-6"),
    )
    state = PanelState(roster=[entry], selected=entry, account_id=uuid.uuid4())
    runtime = _make_runtime(db_session_factory)
    view = CredentialsSubView(
        runtime=runtime,
        state=state,
        allowed_user_id=100000000000000001,
        tenant_id=tenant_id,
        agent_id=agent_id,
        secret_names=["LINEAR_TOKEN"],
    )

    interaction = _interaction()
    await view._on_remove(interaction, "LINEAR_TOKEN")  # pyright: ignore[reportPrivateUsage]

    posted = interaction.followup.send.call_args.args[0]
    expected = render_change_confirmation(
        ConfigurationChange(
            target_name="ops-bot", kind="key_removed", availability="saved", detail="LINEAR_TOKEN"
        )
    )
    assert posted == expected, f"expected the renderer's own copy, got {posted!r}"
