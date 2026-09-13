"""Tests for the task-continuity wiring in `DaimonBot._orchestrate`.

Covers what happens after `bind_session` returns or raises: the
`SessionPreparationFailed` / `SessionAgentMismatch` copy paths (no turn runs
either way), `is_setup` derived from `thread_binding_kind` rather than
`thread_binding_id`, `session_state` threaded into `render_turn_origin`, the
pre-answer loss/replacement notices, and the post-answer "must finish"
follow-up. `bind_session` and `run_prepared_turn` are patched at the names
`bot.py` imports (mirrors the existing `build_context_xml` patching
precedent in `test_orchestration.py`) so each test controls exactly the
`ContinuityOutcome` it wants to observe, without driving the full
session-preparation/compat pipeline (covered at the core level).
"""

from __future__ import annotations

import types
import uuid
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import anthropic as _anthropic
import discord
import httpx
from anthropic.types.beta import BetaEnvironment, BetaManagedAgentsAgent
from anthropic.types.beta.beta_cloud_config import BetaCloudConfig
from anthropic.types.beta.beta_managed_agents_model_config import (
    BetaManagedAgentsModelConfig as _AgentModelConfig,
)
from anthropic.types.beta.beta_packages import BetaPackages
from anthropic.types.beta.beta_unrestricted_network import BetaUnrestrictedNetwork
from daimon.adapters.discord.bot import DaimonBot
from daimon.adapters.discord.runtime import DiscordRuntime, build_turn_deps
from daimon.core.config import McpSettings, ThreadNamingSettings
from daimon.core.defaults.metadata import MA_METADATA_KEY_NAME, MA_METADATA_KEY_TENANT
from daimon.core.ma_resolver import ResolverCache
from daimon.core.notebooks._rate_limit import RateLimiter
from daimon.core.scope import DeploymentDefault, ResolvedConfig
from daimon.core.stores import tenant_ledger
from daimon.core.turn.deps import TurnDeps
from daimon.core.turn.errors import SessionAgentMismatch, SessionPreparationFailed
from daimon.core.turn.prepare import ContinuityOutcome, PreparedTurn
from daimon.core.turn.run import RunOutcome
from daimon.core.turn.state import TurnState
from daimon.core.turn_origin import turn_origin as real_turn_origin
from daimon.testing.factories import make_tenant
from daimon.testing.ma import build_stub_anthropic
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_TENANT_UUID_NS = uuid.UUID("00000000-0000-0000-0000-0000000000aa")


async def _noop_recorder(*, event: object) -> None:
    return None


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


def _make_fake_agent(
    *, agent_id: str = "ag_test", name: str = "test-agent", tenant_id: uuid.UUID | None = None
) -> BetaManagedAgentsAgent:
    metadata = {MA_METADATA_KEY_NAME: name}
    if tenant_id is not None:
        metadata[MA_METADATA_KEY_TENANT] = str(tenant_id)
    return BetaManagedAgentsAgent(
        id=agent_id,
        version=1,
        name=name,
        type="agent",
        model=_AgentModelConfig(id="claude-sonnet-4-5"),
        created_at=datetime(2026, 4, 28, tzinfo=UTC),
        updated_at=datetime(2026, 4, 28, tzinfo=UTC),
        mcp_servers=[],
        metadata=metadata,
        skills=[],
        tools=[],
    )


def _make_fake_environment(name: str = "test-env") -> BetaEnvironment:
    return BetaEnvironment(
        id="env_test",
        name=name,
        type="environment",
        config=BetaCloudConfig(
            type="cloud",
            networking=BetaUnrestrictedNetwork(type="unrestricted"),
            packages=BetaPackages(apt=[], cargo=[], gem=[], go=[], npm=[], pip=[]),
        ),
        created_at="2026-04-28T00:00:00Z",
        updated_at="2026-04-28T00:00:00Z",
        description="",
        metadata={},
    )


def _stub_resolved_config(
    *,
    thread_binding_id: uuid.UUID | None = None,
    thread_binding_kind: str | None = None,
) -> ResolvedConfig:
    return ResolvedConfig(
        agent_name="test-agent",
        agent_name_tier="tenant",
        environment_name="test-env",
        environment_name_tier="tenant",
        thread_binding_id=thread_binding_id,
        thread_binding_kind=thread_binding_kind,  # pyright: ignore[reportArgumentType]
    )


def _make_turn_deps(
    settings: MagicMock,
    anthropic: MagicMock,
    sessionmaker: async_sessionmaker[AsyncSession],
    *,
    resolver_cache: ResolverCache,
    deployment_default: DeploymentDefault,
) -> TurnDeps:
    return build_turn_deps(
        settings,
        anthropic,
        sessionmaker,
        deployment_default=deployment_default,
        resolver_cache=resolver_cache,
        billing_config=None,
    )


def _make_runtime(
    sessionmaker: async_sessionmaker[AsyncSession],
    *,
    anthropic: _anthropic.AsyncAnthropic | None = None,
) -> DiscordRuntime:
    settings = MagicMock()
    settings.mcp = McpSettings()
    settings.billing.markup = Decimal("1.0")
    settings.billing.signup_credit = Decimal("0")
    discord_settings = MagicMock()
    discord_settings.max_concurrent_turns_per_tenant = 100
    discord_settings.per_caller_thread_sessions = True
    settings.discord = discord_settings
    settings.thread_naming = ThreadNamingSettings(enabled=False)
    if anthropic is None:
        anthropic = AsyncMock()
        anthropic.beta.agents.retrieve = AsyncMock(return_value=_make_fake_agent())
        anthropic.beta.environments.retrieve = AsyncMock(return_value=_make_fake_environment())
        anthropic.beta.agents.list = MagicMock(return_value=_AsyncIter([]))
    from daimon.core.ma_resolver import new_resolver_cache

    resolver_cache = new_resolver_cache()
    deployment_default = DeploymentDefault()
    return DiscordRuntime(
        settings=settings,
        anthropic=anthropic,
        sessionmaker=sessionmaker,
        notebook_rate_limiter=RateLimiter(max_requests=999),
        billing_config=None,
        deployment_default=deployment_default,
        resolver_cache=resolver_cache,
        turn_deps=_make_turn_deps(
            settings,
            anthropic,
            sessionmaker,
            resolver_cache=resolver_cache,
            deployment_default=deployment_default,
        ),
    )


def _make_bot(runtime: DiscordRuntime) -> DaimonBot:
    intents = discord.Intents.default()
    intents.message_content = True
    bot = DaimonBot(runtime=runtime, intents=intents)
    bot._connection.user = MagicMock(spec=discord.ClientUser)  # pyright: ignore[reportPrivateUsage]
    bot._connection.user.id = 999  # pyright: ignore[reportPrivateUsage]
    bot._connection.user.mentioned_in = MagicMock(return_value=True)  # pyright: ignore[reportPrivateUsage]
    return bot


def _make_thread_message(
    *,
    content: str = "<@999> hello",
    guild_id: int = 123456,
    thread_id: int = 5555,
    parent_id: int = 789,
    author_id: int = 111,
) -> MagicMock:
    message = MagicMock(spec=discord.Message)
    message.content = content
    message.author = MagicMock()
    message.author.bot = False
    message.author.id = author_id
    message.guild = MagicMock(spec=discord.Guild)
    message.guild.id = guild_id
    message.guild.owner_id = 1
    thread = MagicMock(spec=discord.Thread)
    thread.id = thread_id
    thread.parent_id = parent_id
    message_ref = MagicMock()
    message_ref.id = 42
    message_ref.edit = AsyncMock()
    thread.send = AsyncMock(return_value=message_ref)
    message.channel = thread
    message.add_reaction = AsyncMock()
    message.attachments = []
    message.mentions = [types.SimpleNamespace(id=999)]
    return message


def _make_prepared_turn(
    *, continuity: ContinuityOutcome, account_id: uuid.UUID, mapping_id: uuid.UUID
) -> PreparedTurn:
    from daimon.core.turn.admission import Admission

    admission = Admission(
        account_id=account_id,
        agent=_make_fake_agent(),
        environment=_make_fake_environment(),
        config=_stub_resolved_config(),
    )
    return PreparedTurn(
        admission=admission,
        ma_session_id="sess_test",
        mapping_id=mapping_id,
        watermark=None,
        reused=True,
        session_account_id=account_id,
        _record=_noop_recorder,
        continuity=continuity,
    )


async def _seed_tenant(db_session: AsyncSession, *, guild_id: str) -> uuid.UUID:
    tenant = await make_tenant(db_session, platform="discord", workspace_id=guild_id)
    await tenant_ledger.insert_entry(
        db_session,
        tenant_id=tenant.id,
        delta_usd=Decimal("100.00"),
        reason="trial",
        idempotency_key=f"trial:{tenant.id}",
    )
    await db_session.commit()
    return tenant.id


@patch("daimon.core.turn.admission.resolve_agent", new_callable=AsyncMock)
@patch("daimon.core.turn.admission.resolve_environment", new_callable=AsyncMock)
@patch("daimon.core.turn.admission.resolve_config", new_callable=AsyncMock)
async def test_session_preparation_failed_posts_copy_and_runs_no_turn(
    mock_resolve_config: AsyncMock,
    mock_resolve_env: AsyncMock,
    mock_resolve_agent: AsyncMock,
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    guild_id = "700000001"
    tenant_id = await _seed_tenant(db_session, guild_id=guild_id)
    mock_resolve_config.return_value = _stub_resolved_config()
    mock_resolve_agent.return_value = "ag_test"
    mock_resolve_env.return_value = "env_test"

    runtime = _make_runtime(db_session_factory)
    bot = _make_bot(runtime)
    message = _make_thread_message(guild_id=int(guild_id))

    with (
        patch(
            "daimon.adapters.discord.bot.bind_session",
            new_callable=AsyncMock,
            side_effect=SessionPreparationFailed(
                reasons=("agent_identity",),
                stage="checkpointed",
                retry_after=datetime.now(UTC),
            ),
        ),
        patch(
            "daimon.adapters.discord.bot.run_prepared_turn", new_callable=AsyncMock
        ) as mock_run_prepared_turn,
    ):
        await bot.on_message(message)

    mock_run_prepared_turn.assert_not_called()
    posted = [
        c.kwargs.get("content") for c in message.channel.send.return_value.edit.call_args_list
    ]
    assert any(p is not None and "could not get test-agent ready" in p.lower() for p in posted), (
        f"expected the preparation-failed copy, got {posted}"
    )
    assert any(p is not None and "mention me again to retry" in p.lower() for p in posted)
    _ = tenant_id


@patch("daimon.core.turn.admission.resolve_agent", new_callable=AsyncMock)
@patch("daimon.core.turn.admission.resolve_environment", new_callable=AsyncMock)
@patch("daimon.core.turn.admission.resolve_config", new_callable=AsyncMock)
async def test_responder_changed_without_handoff_posts_offer_and_runs_no_turn(
    mock_resolve_config: AsyncMock,
    mock_resolve_env: AsyncMock,
    mock_resolve_agent: AsyncMock,
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    guild_id = "700000002"
    tenant_id = await _seed_tenant(db_session, guild_id=guild_id)
    mock_resolve_config.return_value = _stub_resolved_config()
    mock_resolve_agent.return_value = "ag_test"
    mock_resolve_env.return_value = "env_test"

    runtime = _make_runtime(db_session_factory)
    # The owner-name lookup walks the tenant's live agents; give it one match
    # for the mismatch's `source_agent_id` so the copy names the real owner
    # rather than falling back to "the previous agent".
    runtime.anthropic.beta.agents.list = MagicMock(  # pyright: ignore[reportAttributeAccessIssue]
        return_value=_AsyncIter(
            [_make_fake_agent(agent_id="ag_owner", name="owner-bot", tenant_id=tenant_id)]
        )
    )
    bot = _make_bot(runtime)
    message = _make_thread_message(guild_id=int(guild_id))

    with (
        patch(
            "daimon.adapters.discord.bot.bind_session",
            new_callable=AsyncMock,
            side_effect=SessionAgentMismatch(
                mapping_id=uuid.uuid4(),
                session_id="sess_dead",
                source_agent_id="ag_owner",
                destination_agent_id="ag_test",
            ),
        ),
        patch(
            "daimon.adapters.discord.bot.run_prepared_turn", new_callable=AsyncMock
        ) as mock_run_prepared_turn,
    ):
        await bot.on_message(message)

    mock_run_prepared_turn.assert_not_called()
    posted = [
        c.kwargs.get("content") for c in message.channel.send.return_value.edit.call_args_list
    ]
    assert any(
        p is not None and "test-agent now answers" in p and "belongs to owner-bot" in p
        for p in posted
    ), f"expected the responder-changed-without-handoff offer, got {posted}"
    assert any(p is not None and "have test-agent take over this task" in p for p in posted)


@patch("daimon.core.turn.admission.resolve_agent", new_callable=AsyncMock)
@patch("daimon.core.turn.admission.resolve_environment", new_callable=AsyncMock)
@patch("daimon.core.turn.admission.resolve_config", new_callable=AsyncMock)
@patch("daimon.adapters.discord.bot.build_context_xml", new_callable=AsyncMock)
async def test_is_setup_false_for_a_handoff_binding(
    mock_build_context_xml: AsyncMock,
    mock_resolve_config: AsyncMock,
    mock_resolve_env: AsyncMock,
    mock_resolve_agent: AsyncMock,
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    mock_build_context_xml.return_value = ("<user_query>hello</user_query>", [])
    guild_id = "700000003"
    tenant_id = await _seed_tenant(db_session, guild_id=guild_id)
    mock_resolve_config.return_value = _stub_resolved_config(
        thread_binding_id=uuid.uuid4(), thread_binding_kind="handoff"
    )
    mock_resolve_agent.return_value = "ag_test"
    mock_resolve_env.return_value = "env_test"

    agent = _make_fake_agent(tenant_id=tenant_id)

    def _agents_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=agent.model_dump(mode="json"))

    runtime = _make_runtime(db_session_factory, anthropic=build_stub_anthropic(_agents_handler))
    bot = _make_bot(runtime)
    message = _make_thread_message(guild_id=int(guild_id))
    account_id = uuid.uuid4()
    mapping_id = uuid.uuid4()
    prepared = _make_prepared_turn(
        continuity=ContinuityOutcome(), account_id=account_id, mapping_id=mapping_id
    )
    run_outcome = RunOutcome(
        state=TurnState(),
        ma_session_id="sess_test",
        mapping_id=mapping_id,
        recovered=False,
    )

    with (
        patch(
            "daimon.adapters.discord.bot.bind_session",
            new_callable=AsyncMock,
            return_value=prepared,
        ),
        patch(
            "daimon.adapters.discord.bot.run_prepared_turn",
            new_callable=AsyncMock,
            return_value=run_outcome,
        ),
        patch("daimon.adapters.discord.bot.turn_origin", wraps=real_turn_origin) as spy,
    ):
        await bot.on_message(message)

    assert spy.call_args is not None
    assert spy.call_args.kwargs["is_setup"] is False, (
        "a handoff binding must not read as a setup conversation"
    )


@patch("daimon.core.turn.admission.resolve_agent", new_callable=AsyncMock)
@patch("daimon.core.turn.admission.resolve_environment", new_callable=AsyncMock)
@patch("daimon.core.turn.admission.resolve_config", new_callable=AsyncMock)
@patch("daimon.adapters.discord.bot.build_context_xml", new_callable=AsyncMock)
async def test_replaced_after_loss_posts_the_unexpected_loss_copy_before_the_answer(
    mock_build_context_xml: AsyncMock,
    mock_resolve_config: AsyncMock,
    mock_resolve_env: AsyncMock,
    mock_resolve_agent: AsyncMock,
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    mock_build_context_xml.return_value = ("<user_query>hello</user_query>", [])
    guild_id = "700000004"
    await _seed_tenant(db_session, guild_id=guild_id)
    mock_resolve_config.return_value = _stub_resolved_config()
    mock_resolve_agent.return_value = "ag_test"
    mock_resolve_env.return_value = "env_test"

    runtime = _make_runtime(db_session_factory)
    bot = _make_bot(runtime)
    message = _make_thread_message(guild_id=int(guild_id))
    account_id = uuid.uuid4()
    mapping_id = uuid.uuid4()
    prepared = _make_prepared_turn(
        continuity=ContinuityOutcome(state="replaced_after_loss", transfer_kind="transcript"),
        account_id=account_id,
        mapping_id=mapping_id,
    )
    run_outcome = RunOutcome(
        state=TurnState(), ma_session_id="sess_test", mapping_id=mapping_id, recovered=False
    )

    with (
        patch(
            "daimon.adapters.discord.bot.bind_session",
            new_callable=AsyncMock,
            return_value=prepared,
        ),
        patch(
            "daimon.adapters.discord.bot.run_prepared_turn",
            new_callable=AsyncMock,
            return_value=run_outcome,
        ),
    ):
        await bot.on_message(message)

    sent_texts = [c.args[0] for c in message.channel.send.call_args_list if c.args]
    assert any("lost the workspace this task was running in" in t for t in sent_texts), (
        f"expected the unexpected-loss copy posted directly to the thread, got {sent_texts}"
    )


@patch("daimon.core.turn.admission.resolve_agent", new_callable=AsyncMock)
@patch("daimon.core.turn.admission.resolve_environment", new_callable=AsyncMock)
@patch("daimon.core.turn.admission.resolve_config", new_callable=AsyncMock)
@patch("daimon.adapters.discord.bot.build_context_xml", new_callable=AsyncMock)
async def test_pending_change_posts_must_finish_copy_after_the_answer(
    mock_build_context_xml: AsyncMock,
    mock_resolve_config: AsyncMock,
    mock_resolve_env: AsyncMock,
    mock_resolve_agent: AsyncMock,
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    mock_build_context_xml.return_value = ("<user_query>hello</user_query>", [])
    guild_id = "700000005"
    await _seed_tenant(db_session, guild_id=guild_id)
    mock_resolve_config.return_value = _stub_resolved_config()
    mock_resolve_agent.return_value = "ag_test"
    mock_resolve_env.return_value = "env_test"

    runtime = _make_runtime(db_session_factory)
    bot = _make_bot(runtime)
    message = _make_thread_message(guild_id=int(guild_id))
    account_id = uuid.uuid4()
    mapping_id = uuid.uuid4()
    prepared = _make_prepared_turn(
        continuity=ContinuityOutcome(pending=("model",)),
        account_id=account_id,
        mapping_id=mapping_id,
    )
    run_outcome = RunOutcome(
        state=TurnState(), ma_session_id="sess_test", mapping_id=mapping_id, recovered=False
    )

    with (
        patch(
            "daimon.adapters.discord.bot.bind_session",
            new_callable=AsyncMock,
            return_value=prepared,
        ),
        patch(
            "daimon.adapters.discord.bot.run_prepared_turn",
            new_callable=AsyncMock,
            return_value=run_outcome,
        ),
    ):
        await bot.on_message(message)

    sent_texts = [c.args[0] for c in message.channel.send.call_args_list if c.args]
    assert any(
        "still working on the previous message here" in t and "picks it up on the next message" in t
        for t in sent_texts
    ), f"expected the must-finish copy posted after the answer, got {sent_texts}"
