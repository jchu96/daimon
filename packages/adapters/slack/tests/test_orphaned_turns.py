"""The boot sweep that lays to rest Slack turns whose process died mid-flight.

Mirrors the intent of the Discord adapter's test_orphaned_turns.py, but the
row-state coverage differs on purpose. Discord's third case is a reconnect
that must not re-trigger the sweep -- that does not apply here (the sweep is
spawned once from a straight-line statement in ``main()`` and cannot run
twice in one process, see boot_sweep.py's docstring). It is replaced by two
cases unique to Slack's per-tenant token model: an uninstalled workspace
(no bot-token row) and a legacy row with no channel (every orphan already in
production before the channel column shipped). A further case covers the
sweep's own compare-and-clear: a marker rewritten mid-sweep, by a turn
admitted while the sweep was still running, must survive it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import httpx
import yarl
from cryptography.fernet import Fernet
from daimon.adapters.slack.boot_sweep import retire_orphaned_turns
from daimon.adapters.slack.runtime import SlackRuntime
from daimon.core.defaults.provisioning import provision_tenant
from daimon.core.github_credentials import build_multifernet, encrypt_token
from daimon.core.ma_identity import derive_tenant_uuid
from daimon.core.stores.slack_bot_tokens import upsert_slack_bot_token
from daimon.core.stores.thread_sessions import (
    create_thread_session,
    list_orphaned_turns,
    mark_turn_active,
)
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .conftest import CHAT_OK_PAYLOAD

_NOW = datetime(2026, 8, 29, 12, 0, 0, tzinfo=UTC)
_UPDATE_URL = yarl.URL("https://slack.com/api/chat.update")
_POST_URL = yarl.URL("https://slack.com/api/chat.postMessage")


def _build_runtime(fernet_key: str, db_factory: async_sessionmaker[AsyncSession]) -> SlackRuntime:
    settings = MagicMock()
    settings.crypto.keys = (SecretStr(fernet_key),)
    return SlackRuntime(
        settings=settings,
        anthropic=MagicMock(),
        sessionmaker=db_factory,
        billing_config=None,
        http_client=MagicMock(spec=httpx.AsyncClient),
        resolver_cache=MagicMock(),  # pyright: ignore[reportArgumentType]  # stub, turn path not exercised
        turn_deps=MagicMock(),  # pyright: ignore[reportArgumentType]  # stub, turn path not exercised
    )


async def _seed_orphan(
    db_factory: async_sessionmaker[AsyncSession],
    fernet_key: str,
    *,
    team_id: str,
    thread_id: str = "1000.1",
    channel_id: str | None = "C_TEST",
    message_id: str = "1000000000.000001",
    with_token: bool = True,
) -> uuid.UUID:
    """Seed one tenant with a mid-flight thread_sessions row.

    ``with_token=False`` skips the ``slack_bot_tokens`` insert, reproducing an
    uninstalled workspace. ``channel_id=None`` reproduces a legacy row from
    before this phase's channel column existed.
    """
    tenant_id = derive_tenant_uuid(platform="slack", workspace_id=team_id)
    await provision_tenant(db_factory, platform="slack", workspace_id=team_id)
    if with_token:
        fernet = build_multifernet((fernet_key,))
        async with db_factory() as s:
            await upsert_slack_bot_token(
                s, team_id=team_id, encrypted_token=encrypt_token(fernet, "xoxb-test")
            )
            await s.commit()
    async with db_factory() as s:
        row = await create_thread_session(
            s,
            tenant_id=tenant_id,
            platform="slack",
            thread_id=thread_id,
            account_id=uuid.uuid4(),
            ma_session_id="sesn_test",
        )
        await mark_turn_active(
            s,
            id=row.id,
            active_turn_message_id=message_id,
            active_turn_channel_id=channel_id,
            now=_NOW,
        )
        await s.commit()
    return row.id


async def test_sweep_edits_the_frozen_card_and_clears_the_marker(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    fernet_key = Fernet.generate_key().decode()
    await _seed_orphan(
        db_session_factory,
        fernet_key,
        team_id="T_REACHABLE",
        channel_id="C_REACHABLE",
        message_id="1111.1",
    )
    runtime = _build_runtime(fernet_key, db_session_factory)

    await retire_orphaned_turns(runtime, now=_NOW)

    update_calls = fake_slack_web_client.mock.requests.get(("POST", _UPDATE_URL), [])
    assert len(update_calls) == 1, "exactly one chat.update for the one seeded orphan"
    body = update_calls[0].kwargs["json"]
    assert body["channel"] == "C_REACHABLE", "must edit the row's own channel"
    assert body["ts"] == "1111.1", "must edit the row's own message ts"
    assert "interrupted" in body["blocks"][0]["text"]["text"], (
        "the card must say the turn was interrupted"
    )

    post_calls = fake_slack_web_client.mock.requests.get(("POST", _POST_URL), [])
    assert not post_calls, "D-09: the sweep edits in place, it never posts a new message"

    assert await list_orphaned_turns(db_session, platform="slack") == [], (
        "a retired turn must not be retired again on the next boot"
    )


async def test_sweep_clears_the_marker_when_the_card_is_unreachable(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    fernet_key = Fernet.generate_key().decode()
    await _seed_orphan(db_session_factory, fernet_key, team_id="T_UNREACHABLE")
    fake_slack_web_client.mock.clear()
    fake_slack_web_client.mock.post(  # pyright: ignore[reportUnknownMemberType]
        str(_UPDATE_URL),
        payload={"ok": False, "error": "message_not_found"},
        repeat=True,
    )
    runtime = _build_runtime(fernet_key, db_session_factory)

    await retire_orphaned_turns(runtime, now=_NOW)  # must not raise

    assert await list_orphaned_turns(db_session, platform="slack") == [], (
        "a permanently unreachable card must not be retried on every boot"
    )


async def test_sweep_skips_the_api_call_and_still_clears_when_the_workspace_uninstalled(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    fernet_key = Fernet.generate_key().decode()
    await _seed_orphan(db_session_factory, fernet_key, team_id="T_UNINSTALLED", with_token=False)
    runtime = _build_runtime(fernet_key, db_session_factory)

    await retire_orphaned_turns(runtime, now=_NOW)

    update_calls = fake_slack_web_client.mock.requests.get(("POST", _UPDATE_URL), [])
    assert not update_calls, "an uninstalled workspace has no token to build a client from"
    assert await list_orphaned_turns(db_session, platform="slack") == [], (
        "the row must still clear so it is not retried forever"
    )


async def test_sweep_clears_a_legacy_row_with_no_channel_without_calling_slack(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    fernet_key = Fernet.generate_key().decode()
    await _seed_orphan(db_session_factory, fernet_key, team_id="T_LEGACY", channel_id=None)
    runtime = _build_runtime(fernet_key, db_session_factory)

    await retire_orphaned_turns(runtime, now=_NOW)

    update_calls = fake_slack_web_client.mock.requests.get(("POST", _UPDATE_URL), [])
    assert not update_calls, "a NULL channel means there is nothing addressable to edit"
    assert await list_orphaned_turns(db_session, platform="slack") == [], (
        "a legacy pre-channel row must still clear without an API call"
    )


async def test_sweep_one_tenants_missing_token_does_not_stop_the_next_tenants_edit(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    fernet_key = Fernet.generate_key().decode()
    await _seed_orphan(db_session_factory, fernet_key, team_id="T_NO_TOKEN", with_token=False)
    await _seed_orphan(
        db_session_factory,
        fernet_key,
        team_id="T_HAS_TOKEN",
        channel_id="C_HAS_TOKEN",
        message_id="2222.2",
    )
    runtime = _build_runtime(fernet_key, db_session_factory)

    await retire_orphaned_turns(runtime, now=_NOW)

    update_calls = fake_slack_web_client.mock.requests.get(("POST", _UPDATE_URL), [])
    assert len(update_calls) == 1, "the reachable tenant's card must still be edited"
    assert update_calls[0].kwargs["json"]["channel"] == "C_HAS_TOKEN", (
        "one workspace's missing token must not divert the edit to another workspace"
    )
    assert await list_orphaned_turns(db_session, platform="slack") == [], (
        "both tenants' rows must clear regardless of which one was reachable"
    )


async def test_sweep_leaves_a_marker_that_moved_while_the_sweep_was_running(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    """A marker rewritten between the sweep's read and its clear must survive.

    The sweep's orphan list is a snapshot taken once, up front. A marker
    rewritten after that snapshot belongs to a turn a live process now owns
    -- clearing it here would leave that turn unprotected against a second
    crash, with no future sweep able to find it. This drives the interleave
    through a callback on the fake Slack transport rather than patching the
    sweep or the store: the fake is transport-level by construction, and a
    patch-based interleave would prove nothing about the real code path.
    """
    fernet_key = Fernet.generate_key().decode()
    row_id = await _seed_orphan(
        db_session_factory,
        fernet_key,
        team_id="T_MOVED",
        channel_id="C_MOVED",
        message_id="3333.3",
    )

    async def _admit_a_fresh_turn_mid_sweep(url: yarl.URL, **kwargs: object) -> None:
        # Fires from inside the sweep's own chat_update -- the exact window
        # between the orphan read and the clear. Models a mention admitted
        # while the sweep was still working: a live process writes a new
        # marker on the same row before the sweep gets to clear it.
        async with db_session_factory() as session:
            await mark_turn_active(
                session,
                id=row_id,
                active_turn_message_id="4444.4",
                active_turn_channel_id="C_MOVED",
                now=_NOW,
            )
            await session.commit()

    fake_slack_web_client.mock.clear()
    fake_slack_web_client.mock.post(  # pyright: ignore[reportUnknownMemberType]
        str(_UPDATE_URL),
        payload=CHAT_OK_PAYLOAD,
        callback=_admit_a_fresh_turn_mid_sweep,
        repeat=True,
    )
    runtime = _build_runtime(fernet_key, db_session_factory)

    await retire_orphaned_turns(runtime, now=_NOW)

    update_calls = fake_slack_web_client.mock.requests.get(("POST", _UPDATE_URL), [])
    assert len(update_calls) == 1, "exactly one chat.update for the one seeded orphan"
    assert update_calls[0].kwargs["json"]["ts"] == "3333.3", (
        "the sweep must edit the ts it read at sweep start -- the frozen card, never the live one"
    )

    orphans = await list_orphaned_turns(db_session, platform="slack")
    assert [o.id for o in orphans] == [row_id], (
        "a marker written after the sweep's read belongs to a live turn and must survive the sweep"
    )
    assert orphans[0].active_turn_message_id == "4444.4", (
        "the surviving marker must be the one the mid-sweep turn wrote, not the sweep's stale read"
    )
    assert orphans[0].active_turn_channel_id == "C_MOVED", (
        "the conditional clear must null nothing at all here, not partially clear the row"
    )
