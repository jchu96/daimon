"""Tests for EnvCredentialModal / McpCredentialModal -- atomic consume then
the existing write paths, with secret-hygiene assertions."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import structlog
from daimon.adapters.discord.credential_modals import EnvCredentialModal, McpCredentialModal
from daimon.adapters.discord.runtime import DiscordRuntime
from daimon.core.credential_requests import mint_request_token
from daimon.core.ma_resolver import new_resolver_cache
from daimon.core.notebooks._rate_limit import RateLimiter
from daimon.core.scope import DeploymentDefault
from daimon.core.stores.agent_files import list_agent_files
from daimon.core.stores.credential_requests import create_credential_request
from daimon.core.stores.domain import CredentialRequestRow
from daimon.testing.factories import make_tenant
from daimon.testing.ma import build_stub_anthropic
from pydantic import HttpUrl
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_SECRET_VALUE = "super-secret-env-value-do-not-leak"
_MCP_TOKEN = "super-secret-mcp-token-do-not-leak"


def _runtime(
    *,
    sessionmaker: async_sessionmaker[AsyncSession],
    anthropic: Any = None,
    public_url: HttpUrl | None = None,
    jwt_secret: str | None = None,
) -> DiscordRuntime:
    settings = MagicMock()
    settings.mcp.public_url = public_url
    if jwt_secret is not None:
        secret_mock = MagicMock()
        secret_mock.get_secret_value.return_value = jwt_secret
        settings.mcp.jwt_secret = secret_mock
    else:
        settings.mcp.jwt_secret = None
    return DiscordRuntime(
        settings=settings,
        anthropic=anthropic if anthropic is not None else build_stub_anthropic(),
        sessionmaker=sessionmaker,
        notebook_rate_limiter=RateLimiter(max_requests=999),
        billing_config=None,
        deployment_default=DeploymentDefault(),
        resolver_cache=new_resolver_cache(),
        turn_deps=MagicMock(),  # pyright: ignore[reportArgumentType]  # credential-modal tests never run a turn
    )


async def _seed_env_request(
    db_session_factory: async_sessionmaker[AsyncSession],
    *,
    target: str = "OPENAI_API_KEY",
    expires_at: datetime | None = None,
) -> CredentialRequestRow:
    token = mint_request_token()
    async with db_session_factory() as session, session.begin():
        tenant = await make_tenant(session, platform="discord", workspace_id=f"g-{token[:8]}")
        row = await create_credential_request(
            session,
            token=token,
            kind="env",
            tenant_id=tenant.id,
            agent_id=uuid.uuid4(),
            account_id=uuid.uuid4(),
            target=target,
            mcp_server_url=None,
            requester_platform_user_id="100000000000000001",
            channel_id="chan-1",
            expires_at=expires_at or (datetime.now(UTC) + timedelta(minutes=30)),
        )
    return row


async def _seed_mcp_request(
    db_session_factory: async_sessionmaker[AsyncSession],
    *,
    mcp_server_url: str = "https://ext.example.com/mcp",
) -> CredentialRequestRow:
    token = mint_request_token()
    async with db_session_factory() as session, session.begin():
        tenant = await make_tenant(session, platform="discord", workspace_id=f"g-{token[:8]}")
        row = await create_credential_request(
            session,
            token=token,
            kind="mcp",
            tenant_id=tenant.id,
            agent_id=uuid.uuid4(),
            account_id=uuid.uuid4(),
            target="linear",
            mcp_server_url=mcp_server_url,
            requester_platform_user_id="100000000000000001",
            channel_id="chan-1",
            expires_at=datetime.now(UTC) + timedelta(minutes=30),
        )
    return row


def _interaction() -> MagicMock:
    interaction = MagicMock()
    interaction.response.defer = AsyncMock()
    interaction.followup.send = AsyncMock()
    return interaction


# --- EnvCredentialModal ------------------------------------------------------


async def test_env_modal_submit_consumes_token_and_writes_agent_file(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    row = await _seed_env_request(db_session_factory, target="OPENAI_API_KEY")
    runtime = _runtime(sessionmaker=db_session_factory)
    modal = EnvCredentialModal(runtime=runtime, request_row=row)
    modal.value_input._value = _SECRET_VALUE  # pyright: ignore[reportPrivateUsage]

    interaction = _interaction()
    await modal.on_submit(interaction)

    rows = await list_agent_files(db_session, tenant_id=row.tenant_id, agent_id=row.agent_id)
    assert len(rows) == 1, "exactly one agent_files row must be written"
    assert rows[0].key == "OPENAI_API_KEY", "the key must come from the consumed row's target"
    assert rows[0].content == _SECRET_VALUE, "the value comes from the modal's TextInput"

    toast = interaction.followup.send.call_args.args[0]
    assert "OPENAI_API_KEY" in toast, "confirmation names the key"
    assert _SECRET_VALUE not in toast, "confirmation never echoes the secret value"


async def test_env_modal_double_submit_writes_exactly_one_row(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    row = await _seed_env_request(db_session_factory, target="TOGGL_TOKEN")
    runtime = _runtime(sessionmaker=db_session_factory)

    first_modal = EnvCredentialModal(runtime=runtime, request_row=row)
    first_modal.value_input._value = "first-value"  # pyright: ignore[reportPrivateUsage]
    await first_modal.on_submit(_interaction())

    second_modal = EnvCredentialModal(runtime=runtime, request_row=row)
    second_modal.value_input._value = "second-value"  # pyright: ignore[reportPrivateUsage]
    second_interaction = _interaction()
    await second_modal.on_submit(second_interaction)

    rows = await list_agent_files(db_session, tenant_id=row.tenant_id, agent_id=row.agent_id)
    assert len(rows) == 1, "a resubmission on an already-consumed token must write nothing new"
    assert rows[0].content == "first-value", "only the first submission's value is stored"

    message = second_interaction.followup.send.call_args.args[0]
    assert "no longer valid" in message, "the second submission must report the request is dead"


async def test_env_modal_expired_row_consumes_nothing_and_writes_nothing(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    row = await _seed_env_request(
        db_session_factory,
        target="EXPIRED_KEY",
        expires_at=datetime.now(UTC) - timedelta(minutes=1),
    )
    runtime = _runtime(sessionmaker=db_session_factory)
    modal = EnvCredentialModal(runtime=runtime, request_row=row)
    modal.value_input._value = _SECRET_VALUE  # pyright: ignore[reportPrivateUsage]

    interaction = _interaction()
    await modal.on_submit(interaction)

    rows = await list_agent_files(db_session, tenant_id=row.tenant_id, agent_id=row.agent_id)
    assert rows == [], "an expired row must never produce a write"


async def test_env_modal_empty_value_writes_nothing_and_does_not_consume(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    row = await _seed_env_request(db_session_factory, target="EMPTY_KEY")
    runtime = _runtime(sessionmaker=db_session_factory)
    modal = EnvCredentialModal(runtime=runtime, request_row=row)
    modal.value_input._value = "   "  # pyright: ignore[reportPrivateUsage]

    interaction = _interaction()
    await modal.on_submit(interaction)

    rows = await list_agent_files(db_session, tenant_id=row.tenant_id, agent_id=row.agent_id)
    assert rows == [], "a blank value must never produce a write"
    message = interaction.followup.send.call_args.args[0]
    assert "empty" in message.lower(), "the toast must ask the user to try again"

    # The token must still be usable -- a fail-fast rejection did not consume it.
    retry_modal = EnvCredentialModal(runtime=runtime, request_row=row)
    retry_modal.value_input._value = _SECRET_VALUE  # pyright: ignore[reportPrivateUsage]
    await retry_modal.on_submit(_interaction())
    retry_rows = await list_agent_files(db_session, tenant_id=row.tenant_id, agent_id=row.agent_id)
    assert len(retry_rows) == 1, "the token must still be consumable after a validation rejection"


async def test_env_modal_value_over_byte_cap_writes_nothing_and_does_not_consume(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    row = await _seed_env_request(db_session_factory, target="TOO_BIG_KEY")
    runtime = _runtime(sessionmaker=db_session_factory)
    modal = EnvCredentialModal(runtime=runtime, request_row=row)
    modal.value_input._value = "x" * 5000  # pyright: ignore[reportPrivateUsage]  # over the 4096-byte cap

    interaction = _interaction()
    await modal.on_submit(interaction)

    rows = await list_agent_files(db_session, tenant_id=row.tenant_id, agent_id=row.agent_id)
    assert rows == [], "an over-cap value must never produce a write"
    message = interaction.followup.send.call_args.args[0]
    assert "too large" in message.lower(), "the toast must report the size cap"

    retry_modal = EnvCredentialModal(runtime=runtime, request_row=row)
    retry_modal.value_input._value = _SECRET_VALUE  # pyright: ignore[reportPrivateUsage]
    await retry_modal.on_submit(_interaction())
    retry_rows = await list_agent_files(db_session, tenant_id=row.tenant_id, agent_id=row.agent_id)
    assert len(retry_rows) == 1, "the token must still be consumable after a cap rejection"


async def test_env_modal_confirmation_states_shared_agent_exposure(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    row = await _seed_env_request(db_session_factory, target="SHARED_KEY")
    runtime = _runtime(sessionmaker=db_session_factory)
    modal = EnvCredentialModal(runtime=runtime, request_row=row)
    modal.value_input._value = _SECRET_VALUE  # pyright: ignore[reportPrivateUsage]

    interaction = _interaction()
    await modal.on_submit(interaction)

    toast = interaction.followup.send.call_args.args[0]
    assert "anyone who talks to this agent" in toast, (
        "confirmation must disclose the credential is usable by every caller of the agent"
    )


async def test_env_modal_never_logs_the_secret_value(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    row = await _seed_env_request(db_session_factory, target="LOGGED_KEY")
    runtime = _runtime(sessionmaker=db_session_factory)
    modal = EnvCredentialModal(runtime=runtime, request_row=row)
    modal.value_input._value = _SECRET_VALUE  # pyright: ignore[reportPrivateUsage]

    cap = structlog.testing.LogCapture()
    structlog.configure(processors=[cap])
    try:
        await modal.on_submit(_interaction())
    finally:
        structlog.reset_defaults()

    for entry in cap.entries:
        assert _SECRET_VALUE not in repr(entry), "no log line may contain the secret value"


# --- McpCredentialModal ------------------------------------------------------


def _vault_handler(
    vault_id: str, per_agent_display: str, creds_created: list[dict[str, Any]]
) -> Any:
    def _handler(req: httpx.Request) -> httpx.Response:
        if req.method == "GET" and req.url.path == "/v1/vaults":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "id": vault_id,
                            "type": "vault",
                            "display_name": per_agent_display,
                            "metadata": None,
                            "archived_at": None,
                            "created_at": "2026-04-01T00:00:00Z",
                        }
                    ],
                    "has_more": False,
                },
            )
        if req.method == "GET" and req.url.path == f"/v1/vaults/{vault_id}/credentials":
            return httpx.Response(200, json={"data": [], "has_more": False})
        if req.method == "POST" and req.url.path == f"/v1/vaults/{vault_id}/credentials":
            body = json.loads(req.content)
            creds_created.append(body)
            return httpx.Response(
                200,
                json={
                    "id": "vcrd_1",
                    "type": "credential",
                    "vault_id": vault_id,
                    "auth": {
                        "type": "static_bearer",
                        "mcp_server_url": body["auth"]["mcp_server_url"],
                    },
                },
            )
        raise AssertionError(f"unexpected: {req.method} {req.url.path}")

    return _handler


async def test_mcp_modal_submit_consumes_token_and_writes_vault_credential(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    row = await _seed_mcp_request(db_session_factory, mcp_server_url="https://ext.example.com/mcp")
    vault_id = "vlt_credmodal"
    per_agent_display = f"daimon-mcp:{row.account_id}:{row.agent_id}"
    creds_created: list[dict[str, Any]] = []

    runtime = _runtime(
        sessionmaker=db_session_factory,
        anthropic=build_stub_anthropic(_vault_handler(vault_id, per_agent_display, creds_created)),
        public_url=HttpUrl("https://mcp.example.com/mcp"),
        jwt_secret="x" * 32,
    )
    modal = McpCredentialModal(runtime=runtime, request_row=row)
    modal.token_input._value = _MCP_TOKEN  # pyright: ignore[reportPrivateUsage]

    interaction = _interaction()
    await modal.on_submit(interaction)

    assert len(creds_created) == 1, "exactly one credential must be POSTed to the per-agent vault"
    assert creds_created[0]["auth"]["mcp_server_url"] == "https://ext.example.com/mcp", (
        "the mcp_server_url must come from the consumed row, never user input"
    )
    assert creds_created[0]["auth"]["token"] == _MCP_TOKEN

    toast = interaction.followup.send.call_args.args[0]
    assert "anyone who talks to this agent" in toast.lower(), (
        "confirmation must disclose the credential is usable by every caller of the agent"
    )


async def test_mcp_modal_unconfigured_mcp_reports_misconfiguration_no_consume_no_vault_write(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    def _no_http(req: httpx.Request) -> httpx.Response:
        raise AssertionError(f"no vault HTTP calls expected: {req.method} {req.url}")

    row = await _seed_mcp_request(db_session_factory)
    runtime = _runtime(
        sessionmaker=db_session_factory,
        anthropic=build_stub_anthropic(_no_http),
        public_url=None,
        jwt_secret=None,
    )
    modal = McpCredentialModal(runtime=runtime, request_row=row)
    modal.token_input._value = _MCP_TOKEN  # pyright: ignore[reportPrivateUsage]

    interaction = _interaction()
    await modal.on_submit(interaction)

    message = interaction.followup.send.call_args.args[0]
    assert "not configured" in message, "must report the daimon-mcp misconfiguration"

    # The token must still be unconsumed -- retry with configured settings must succeed.
    retry_runtime = _runtime(
        sessionmaker=db_session_factory,
        anthropic=build_stub_anthropic(
            _vault_handler("vlt_retry", f"daimon-mcp:{row.account_id}:{row.agent_id}", [])
        ),
        public_url=HttpUrl("https://mcp.example.com/mcp"),
        jwt_secret="x" * 32,
    )
    retry_modal = McpCredentialModal(runtime=retry_runtime, request_row=row)
    retry_modal.token_input._value = _MCP_TOKEN  # pyright: ignore[reportPrivateUsage]
    retry_interaction = _interaction()
    await retry_modal.on_submit(retry_interaction)
    retry_message = retry_interaction.followup.send.call_args.args[0]
    assert "not configured" not in retry_message, (
        "the request must still be consumable once daimon-mcp is configured"
    )


async def test_mcp_modal_vault_write_failure_surfaces_only_exception_class_name(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    def _failing_vault(req: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("upstream reset by peer -- request envelope: secret=abc123")

    row = await _seed_mcp_request(db_session_factory)
    runtime = _runtime(
        sessionmaker=db_session_factory,
        anthropic=build_stub_anthropic(_failing_vault),
        public_url=HttpUrl("https://mcp.example.com/mcp"),
        jwt_secret="x" * 32,
    )
    modal = McpCredentialModal(runtime=runtime, request_row=row)
    modal.token_input._value = _MCP_TOKEN  # pyright: ignore[reportPrivateUsage]

    interaction = _interaction()
    await modal.on_submit(interaction)

    message = interaction.followup.send.call_args.args[0]
    assert "APIConnectionError" in message, "only the exception class name is surfaced"
    assert "secret=abc123" not in message, (
        "the stringified exception (which can carry the request envelope) must never reach the user"
    )


async def test_mcp_modal_never_logs_the_raw_token(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    row = await _seed_mcp_request(db_session_factory)
    vault_id = "vlt_hygiene"
    per_agent_display = f"daimon-mcp:{row.account_id}:{row.agent_id}"
    runtime = _runtime(
        sessionmaker=db_session_factory,
        anthropic=build_stub_anthropic(_vault_handler(vault_id, per_agent_display, [])),
        public_url=HttpUrl("https://mcp.example.com/mcp"),
        jwt_secret="x" * 32,
    )
    modal = McpCredentialModal(runtime=runtime, request_row=row)
    modal.token_input._value = _MCP_TOKEN  # pyright: ignore[reportPrivateUsage]

    cap = structlog.testing.LogCapture()
    structlog.configure(processors=[cap])
    try:
        await modal.on_submit(_interaction())
    finally:
        structlog.reset_defaults()

    for entry in cap.entries:
        assert _MCP_TOKEN not in repr(entry), "no log line may contain the raw token"
