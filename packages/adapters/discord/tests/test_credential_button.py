"""Tests for CredentialRequestButton -- from_custom_id / interaction_check / callback."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

import daimon.adapters.discord.credential_button as credential_button_mod
import structlog
from daimon.adapters.discord.credential_button import CredentialRequestButton
from daimon.adapters.discord.credential_modals import EnvCredentialModal, McpCredentialModal
from daimon.core.credential_requests import build_button_label, build_custom_id, mint_request_token
from daimon.core.stores.credential_requests import create_credential_request
from daimon.core.stores.domain import CredentialRequestRow
from daimon.testing.factories import make_tenant
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_REQUESTER_ID = "100000000000000001"
_OTHER_USER_ID = "200000000000000002"


def _fake_bot(sessionmaker: Any) -> Any:
    """A minimal stand-in for DaimonBot -- only `.runtime.sessionmaker` is read."""
    return SimpleNamespace(runtime=SimpleNamespace(sessionmaker=sessionmaker))


def _interaction(*, user_id: str, client: Any) -> MagicMock:
    interaction = MagicMock()
    interaction.user.id = int(user_id)
    interaction.client = client
    interaction.response.send_message = AsyncMock()
    interaction.response.send_modal = AsyncMock()
    return interaction


def _match(token: str) -> Any:
    custom_id = build_custom_id(token)
    matched = CredentialRequestButton.__discord_ui_compiled_template__.fullmatch(custom_id)
    assert matched is not None, "test token must satisfy the button's own custom_id template"
    return matched


def _row(
    *,
    token: str,
    kind: Literal["env", "mcp"] = "env",
    target: str = "OPENAI_API_KEY",
    mcp_server_url: str | None = None,
    requester_platform_user_id: str = _REQUESTER_ID,
    expires_at: datetime | None = None,
    used_at: datetime | None = None,
) -> CredentialRequestRow:
    """Build a CredentialRequestRow in memory -- no DB needed for interaction_check/callback tests."""
    now = datetime.now(UTC)
    return CredentialRequestRow(
        token=token,
        kind=kind,
        tenant_id=uuid.uuid4(),
        agent_id=uuid.uuid4(),
        account_id=uuid.uuid4(),
        target=target,
        mcp_server_url=mcp_server_url,
        requester_platform_user_id=requester_platform_user_id,
        channel_id="chan-1",
        created_at=now,
        expires_at=expires_at or (now + timedelta(minutes=30)),
        used_at=used_at,
    )


async def _seed_request(
    db_session_factory: async_sessionmaker[AsyncSession],
    *,
    kind: Literal["env", "mcp"] = "env",
    target: str = "OPENAI_API_KEY",
    mcp_server_url: str | None = None,
    requester_platform_user_id: str = _REQUESTER_ID,
) -> str:
    """Seed a real credential_requests row and return its token."""
    token = mint_request_token()
    async with db_session_factory() as session, session.begin():
        tenant = await make_tenant(session, platform="discord", workspace_id=f"guild-{token[:8]}")
        await create_credential_request(
            session,
            token=token,
            kind=kind,
            tenant_id=tenant.id,
            agent_id=uuid.uuid4(),
            account_id=uuid.uuid4(),
            target=target,
            mcp_server_url=mcp_server_url,
            requester_platform_user_id=requester_platform_user_id,
            channel_id="chan-1",
            expires_at=datetime.now(UTC) + timedelta(minutes=30),
        )
    return token


# --- from_custom_id (real DB) -----------------------------------------------


async def test_from_custom_id_known_token_builds_button_with_target_label(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    token = await _seed_request(db_session_factory, kind="env", target="OPENAI_API_KEY")
    bot = _fake_bot(db_session_factory)
    interaction = _interaction(user_id=_REQUESTER_ID, client=bot)

    item = await CredentialRequestButton.from_custom_id(interaction, MagicMock(), _match(token))

    assert item.request_row is not None, "a known token must resolve its row"
    assert item.item.label == build_button_label("env", "OPENAI_API_KEY"), (
        "the reconstructed button must name the exact target"
    )


async def test_from_custom_id_unknown_token_returns_item_with_no_row_and_fallback_label(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    bot = _fake_bot(db_session_factory)
    interaction = _interaction(user_id=_REQUESTER_ID, client=bot)
    token = "never-minted-token-abcdef123456"

    item = await CredentialRequestButton.from_custom_id(interaction, MagicMock(), _match(token))

    assert item.request_row is None, "an unknown token must yield no row rather than raising"
    assert item.item.label == "Add credential", "fallback label is used when no row is found"


async def test_from_custom_id_db_failure_is_logged_and_interaction_check_rejects_gracefully(
    monkeypatch: Any,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A DB failure during the lookup must be logged and must not raise -- the
    exception does not escape into discord.py's swallowing dispatcher."""

    async def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise SQLAlchemyError("connection reset")

    monkeypatch.setattr(credential_button_mod, "peek_credential_request", _boom)

    bot = _fake_bot(db_session_factory)
    interaction = _interaction(user_id=_REQUESTER_ID, client=bot)
    token = "db-failure-simulated-token-000001"

    cap = structlog.testing.LogCapture()
    structlog.configure(processors=[cap])
    try:
        item = await CredentialRequestButton.from_custom_id(interaction, MagicMock(), _match(token))
    finally:
        structlog.reset_defaults()

    assert item.request_row is None, "a DB failure must not raise -- it degrades to no row"
    assert any(entry.get("event") == "credential_button.lookup_failed" for entry in cap.entries), (
        "the failure must be logged for observability, since discord.py's own logger "
        "would otherwise be the only trace of it"
    )

    check_interaction = _interaction(user_id=_REQUESTER_ID, client=bot)
    allowed = await item.interaction_check(check_interaction)

    assert allowed is False, "an item built from a failed lookup must reject the click"
    check_interaction.response.send_message.assert_awaited_once()


# --- interaction_check (in-memory row, no DB) -------------------------------


async def test_interaction_check_unknown_row_sends_ephemeral_and_rejects() -> None:
    item = CredentialRequestButton(
        token="unknown12345678901234", label="Add credential", request_row=None
    )
    interaction = _interaction(user_id=_REQUESTER_ID, client=None)

    allowed = await item.interaction_check(interaction)

    assert allowed is False, "no row means the click must be rejected"
    interaction.response.send_message.assert_awaited_once()
    message = interaction.response.send_message.call_args.args[0]
    assert "no longer valid" in message, "unknown-row rejection must tell the user to ask again"


async def test_interaction_check_wrong_requester_sends_ephemeral_and_rejects() -> None:
    row = _row(token="wrongrequester12345678", requester_platform_user_id=_REQUESTER_ID)
    item = CredentialRequestButton(token=row.token, label="Add credential", request_row=row)
    interaction = _interaction(user_id=_OTHER_USER_ID, client=None)

    allowed = await item.interaction_check(interaction)

    assert allowed is False, "a non-requester click must be rejected"
    message = interaction.response.send_message.call_args.args[0]
    assert "someone else" in message, "rejection must indicate the request targeted another user"


async def test_interaction_check_expired_sends_ephemeral_and_rejects() -> None:
    row = _row(
        token="expiredrow123456789012",
        expires_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    item = CredentialRequestButton(token=row.token, label="Add credential", request_row=row)
    interaction = _interaction(user_id=_REQUESTER_ID, client=None)

    allowed = await item.interaction_check(interaction)

    assert allowed is False, "an expired row must be rejected"
    message = interaction.response.send_message.call_args.args[0]
    assert "expired" in message, "rejection must say the request expired"


async def test_interaction_check_already_used_sends_ephemeral_and_rejects() -> None:
    row = _row(token="usedrow1234567890123456", used_at=datetime.now(UTC))
    item = CredentialRequestButton(token=row.token, label="Add credential", request_row=row)
    interaction = _interaction(user_id=_REQUESTER_ID, client=None)

    allowed = await item.interaction_check(interaction)

    assert allowed is False, "an already-used row must be rejected"
    message = interaction.response.send_message.call_args.args[0]
    assert "already used" in message, "rejection must say the request was already used"


async def test_interaction_check_allows_requester_with_valid_row_and_sends_nothing() -> None:
    row = _row(token="validrow123456789012345", requester_platform_user_id=_REQUESTER_ID)
    item = CredentialRequestButton(token=row.token, label="Add credential", request_row=row)
    interaction = _interaction(user_id=_REQUESTER_ID, client=None)

    allowed = await item.interaction_check(interaction)

    assert allowed is True, "the requester with a fresh, unused row must be allowed through"
    interaction.response.send_message.assert_not_awaited()


# --- callback ----------------------------------------------------------------


async def test_callback_dispatches_env_modal_for_env_kind() -> None:
    row = _row(token="envcallback1234567890123", kind="env")
    item = CredentialRequestButton(token=row.token, label="Add credential", request_row=row)
    bot = _fake_bot(MagicMock())
    interaction = _interaction(user_id=_REQUESTER_ID, client=bot)

    await item.callback(interaction)

    interaction.response.send_modal.assert_awaited_once()
    sent_modal = interaction.response.send_modal.call_args.args[0]
    assert isinstance(sent_modal, EnvCredentialModal), "env-kind rows must open EnvCredentialModal"


async def test_callback_dispatches_mcp_modal_for_mcp_kind() -> None:
    row = _row(
        token="mcpcallback1234567890123",
        kind="mcp",
        mcp_server_url="https://ext.example.com/mcp",
    )
    item = CredentialRequestButton(token=row.token, label="Add credential", request_row=row)
    bot = _fake_bot(MagicMock())
    interaction = _interaction(user_id=_REQUESTER_ID, client=bot)

    await item.callback(interaction)

    interaction.response.send_modal.assert_awaited_once()
    sent_modal = interaction.response.send_modal.call_args.args[0]
    assert isinstance(sent_modal, McpCredentialModal), "mcp-kind rows must open McpCredentialModal"


# --- dispatch matching (no DB) -----------------------------------------------


def test_custom_id_that_does_not_fullmatch_template_never_dispatches() -> None:
    pattern = CredentialRequestButton.__discord_ui_compiled_template__
    assert pattern.fullmatch("not-a-credential-button-id") is None, (
        "an unrelated custom_id must never match this button's dispatch template"
    )
    assert pattern.fullmatch(build_custom_id("a" * 20)) is not None, (
        "a well-formed minted custom_id must match the dispatch template"
    )
