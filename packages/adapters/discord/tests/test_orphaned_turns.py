"""The boot sweep that lays to rest turns whose process died mid-flight.

A turn's render loop lives in the process that started it, so a container
recreate freezes its embed on 'thinking' forever while MA completes the answer
server-side. These cover the sweep itself — an embed it can edit, one it
cannot, and the reconnect that must not trigger it. The marker's own semantics
live in the store tests.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import discord
from daimon.adapters.discord import theme
from daimon.adapters.discord.bot import DaimonBot
from daimon.adapters.discord.runtime import DiscordRuntime
from daimon.core.config import McpSettings
from daimon.core.ma_resolver import new_resolver_cache
from daimon.core.notebooks._rate_limit import RateLimiter
from daimon.core.scope import DeploymentDefault
from daimon.core.stores.thread_sessions import (
    create_thread_session,
    list_orphaned_turns,
    mark_turn_active,
)
from daimon.testing.factories import make_tenant
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


def _make_bot(sessionmaker: async_sessionmaker[AsyncSession]) -> DaimonBot:
    settings = MagicMock()
    settings.mcp = McpSettings()
    discord_settings = MagicMock()
    discord_settings.max_concurrent_turns_per_tenant = 3
    settings.discord = discord_settings
    runtime = DiscordRuntime(
        settings=settings,
        anthropic=AsyncMock(),
        sessionmaker=sessionmaker,
        notebook_rate_limiter=RateLimiter(max_requests=999),
        billing_config=None,
        deployment_default=DeploymentDefault(),
        resolver_cache=new_resolver_cache(),
        turn_deps=MagicMock(),  # pyright: ignore[reportArgumentType]  # the sweep never runs a turn
    )
    intents = discord.Intents.default()
    intents.message_content = True
    return DaimonBot(runtime=runtime, intents=intents)


async def _make_orphan(
    session: AsyncSession,
    *,
    thread_id: str = "555",
    message_id: str = "777",
) -> uuid.UUID:
    tenant = await make_tenant(session)
    row = await create_thread_session(
        session,
        tenant_id=tenant.id,
        platform="discord",
        thread_id=thread_id,
        account_id=uuid.uuid4(),
        ma_session_id="sesn_test",
    )
    await mark_turn_active(
        session,
        id=row.id,
        active_turn_message_id=message_id,
        now=datetime.now(UTC),
    )
    await session.commit()
    return row.id


async def test_sweep_marks_the_embed_failed_and_clears_the_row(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await _make_orphan(db_session)
    bot = _make_bot(db_session_factory)
    message = MagicMock(spec=discord.Message)
    message.edit = AsyncMock()
    thread = MagicMock(spec=discord.Thread)
    thread.fetch_message = AsyncMock(return_value=message)
    bot.get_channel = MagicMock(return_value=thread)  # pyright: ignore[reportAttributeAccessIssue]

    await bot._retire_orphaned_turns()  # pyright: ignore[reportPrivateUsage]

    thread.fetch_message.assert_awaited_once_with(777)
    edited = message.edit.await_args.kwargs["embed"]  # pyright: ignore[reportAny]
    assert edited.color.value == theme.COLOR_RED, "an interrupted turn must render as an error"
    assert "interrupted" in edited.description, (
        "the user must be told the turn is dead, not left on a frozen spinner"
    )
    assert await list_orphaned_turns(db_session, platform="discord") == [], (
        "a retired turn must not be retired again on the next boot"
    )


async def test_sweep_clears_the_row_even_when_the_embed_is_unreachable(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A deleted message or lost permission must not make the sweep retry forever."""
    await _make_orphan(db_session)
    bot = _make_bot(db_session_factory)
    thread = MagicMock(spec=discord.Thread)
    thread.fetch_message = AsyncMock(
        side_effect=discord.NotFound(MagicMock(status=404), "Unknown Message")
    )
    bot.get_channel = MagicMock(return_value=thread)  # pyright: ignore[reportAttributeAccessIssue]

    await bot._retire_orphaned_turns()  # pyright: ignore[reportPrivateUsage]

    assert await list_orphaned_turns(db_session, platform="discord") == [], (
        "an unreachable embed must still clear its marker"
    )


async def test_sweep_runs_once_per_process_not_once_per_reconnect(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """on_ready re-fires on a full gateway reconnect; a live turn is not an orphan."""
    await _make_orphan(db_session)
    bot = _make_bot(db_session_factory)
    message = MagicMock(spec=discord.Message)
    message.edit = AsyncMock()
    thread = MagicMock(spec=discord.Thread)
    thread.fetch_message = AsyncMock(return_value=message)
    bot.get_channel = MagicMock(return_value=thread)  # pyright: ignore[reportAttributeAccessIssue]

    await bot._retire_orphaned_turns()  # pyright: ignore[reportPrivateUsage]
    # A turn started after the sweep — its marker belongs to THIS process.
    mapping_id = await _make_orphan(db_session, thread_id="556", message_id="778")
    await bot._retire_orphaned_turns()  # pyright: ignore[reportPrivateUsage]

    assert [row.id for row in await list_orphaned_turns(db_session, platform="discord")] == [
        mapping_id
    ], "a reconnect must not reap the turns this process is still rendering"
