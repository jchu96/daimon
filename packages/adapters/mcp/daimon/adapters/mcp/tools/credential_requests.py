"""Credential-request tools: request_env_credential, request_mcp_credential.

These two tools replace the "go open /agent-setup" redirect for a task that
needs an env secret or an auth-required MCP server: the agent mints a
single-use, TTL-bounded ``credential_requests`` row and posts a
target-naming button in the thread (``tools/discord/_credential_button.py``).
Clicking the button opens a native Discord modal in a different process (the
Discord bot, worker VM) — the secret itself never travels through either
process's tool arguments, so neither tool below has a token/secret/value
parameter, and none should ever be added.

Deliberately NOT admin-gated — the chat-mutation admin check other tools call
at the top of their impl is intentionally absent here. Authorization for the
actual credential write happens at click time instead, when the Discord
button's ``interaction_check`` compares the clicking user against
``requester_platform_user_id`` on the minted row. This widens today's
admin-only credential writes (``/agent-setup``'s mutation buttons are
``is_admin``-only) in exchange for one-click UX — a deliberate, accepted
trade documented here so a future reviewer does not "fix" the omission by
adding that admin gate back.
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from urllib.parse import urlparse

from daimon.adapters.mcp.auth.resolver import AuthIdentity
from daimon.adapters.mcp.runtime import McpRuntime
from daimon.adapters.mcp.tools._ctx import _auth  # pyright: ignore[reportPrivateUsage]
from daimon.adapters.mcp.tools.discord import (
    _post_credential_button_impl,  # pyright: ignore[reportPrivateUsage]
)
from daimon.core.credential_requests import (
    DEFAULT_TTL,
    CredentialRequestKind,
    mint_request_token,
)
from daimon.core.defaults.ma_index import find_agent_by_daimon_tag
from daimon.core.ma_identity import derive_agent_uuid
from daimon.core.stores.credential_requests import create_credential_request
from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, ConfigDict

# Mirrors packages/adapters/discord/daimon/adapters/discord/agent_setup/credentials.py's
# _POSIX_KEY_RE. Duplicated rather than imported: the Discord adapter and the
# MCP adapter cannot import each other (import-linter's independence
# contract), and this rule is small enough not to warrant a core lift.
_POSIX_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class RequestCredentialResult(BaseModel):
    """Result of minting and posting a credential-request button."""

    model_config = ConfigDict(frozen=True)

    kind: CredentialRequestKind
    target: str
    expires_at: datetime
    message_id: str


async def _resolve_agent_uuid(
    runtime: McpRuntime,
    auth: AuthIdentity,
    agent_name: str,
) -> uuid.UUID:
    ma_agent = await find_agent_by_daimon_tag(
        runtime.client, tenant_id=auth.tenant_id, name=agent_name
    )
    if ma_agent is None:
        raise ToolError(f"agent '{agent_name}' not found in this tenant")
    return derive_agent_uuid(tenant_id=auth.tenant_id, ma_agent_id=str(ma_agent.id))


async def _mint_and_post(
    runtime: McpRuntime,
    auth: AuthIdentity,
    *,
    kind: CredentialRequestKind,
    target: str,
    mcp_server_url: str | None,
    agent_id: uuid.UUID,
    requester_platform_user_id: str,
    agent_name: str,
    purpose: str,
    channel_id: str,
) -> RequestCredentialResult:
    token = mint_request_token()
    expires_at = datetime.now(UTC) + DEFAULT_TTL
    async with runtime.session_factory.begin() as session:
        row = await create_credential_request(
            session,
            token=token,
            kind=kind,
            tenant_id=auth.tenant_id,
            agent_id=agent_id,
            account_id=auth.account_id,
            target=target,
            mcp_server_url=mcp_server_url,
            requester_platform_user_id=requester_platform_user_id,
            channel_id=channel_id,
            expires_at=expires_at,
        )
    try:
        message_id = await _post_credential_button_impl(
            runtime,
            auth,
            channel_id=channel_id,
            kind=kind,
            target=target,
            token=token,
            agent_name=agent_name,
            purpose=purpose,
        )
    except ToolError as exc:
        # The row already exists (single-use + TTL bound it regardless), but
        # with no live button it is silently unusable — say so rather than
        # letting the caller believe the request succeeded.
        raise ToolError(
            f"credential request was created but posting the button failed: {exc}"
        ) from exc
    return RequestCredentialResult(
        kind=kind, target=target, expires_at=row.expires_at, message_id=message_id
    )


async def _request_env_credential_impl(
    runtime: McpRuntime,
    auth: AuthIdentity,
    *,
    agent_name: str,
    key: str,
    purpose: str,
    channel_id: str,
) -> RequestCredentialResult:
    if auth.platform == "slack":
        raise ToolError("request_env_credential is not supported on Slack yet")
    if auth.platform_user_id is None:
        raise ToolError("credential requests require a platform-bound identity")
    if not _POSIX_KEY_RE.match(key):
        raise ToolError(
            "env key must match [A-Za-z_][A-Za-z0-9_]* "
            "(letters, digits, underscores; must not start with a digit)"
        )
    agent_id = await _resolve_agent_uuid(runtime, auth, agent_name)
    return await _mint_and_post(
        runtime,
        auth,
        kind="env",
        target=key,
        mcp_server_url=None,
        agent_id=agent_id,
        requester_platform_user_id=auth.platform_user_id,
        agent_name=agent_name,
        purpose=purpose,
        channel_id=channel_id,
    )


async def _request_mcp_credential_impl(
    runtime: McpRuntime,
    auth: AuthIdentity,
    *,
    agent_name: str,
    server_name: str,
    url: str,
    channel_id: str,
) -> RequestCredentialResult:
    if auth.platform == "slack":
        raise ToolError("request_mcp_credential is not supported on Slack yet")
    if auth.platform_user_id is None:
        raise ToolError("credential requests require a platform-bound identity")
    if urlparse(url).scheme not in ("http", "https"):
        raise ToolError("mcp server url must be http or https")
    agent_id = await _resolve_agent_uuid(runtime, auth, agent_name)
    return await _mint_and_post(
        runtime,
        auth,
        kind="mcp",
        target=server_name,
        mcp_server_url=url,
        agent_id=agent_id,
        requester_platform_user_id=auth.platform_user_id,
        agent_name=agent_name,
        purpose=f"connecting the MCP server '{server_name}'",
        channel_id=channel_id,
    )


def register_credential_request_tools(mcp: FastMCP, runtime: McpRuntime) -> None:
    @mcp.tool
    async def request_env_credential(  # pyright: ignore[reportUnusedFunction]
        ctx: Context,
        agent_name: str,
        key: str,
        purpose: str,
        channel_id: str,
    ) -> RequestCredentialResult:
        """Request an environment secret from the user via a private modal.

        Call this INSTEAD of ever accepting a secret value in chat. Posts a
        button in the thread naming the exact env key; clicking it opens a
        native Discord modal where the user enters the value privately — it
        never appears in this channel or in your context. Once added, the
        credential becomes usable by everyone who talks to ``agent_name``.
        """
        return await _request_env_credential_impl(
            runtime,
            await _auth(ctx),
            agent_name=agent_name,
            key=key,
            purpose=purpose,
            channel_id=channel_id,
        )

    @mcp.tool
    async def request_mcp_credential(  # pyright: ignore[reportUnusedFunction]
        ctx: Context,
        agent_name: str,
        server_name: str,
        url: str,
        channel_id: str,
    ) -> RequestCredentialResult:
        """Request an auth-required MCP server's credential via a private modal.

        Call this INSTEAD of ever accepting a token value in chat, and instead
        of ``attach_mcp_server`` when the server requires authentication.
        Posts a button in the thread naming the exact server; clicking it
        opens a native Discord modal where the user enters the token
        privately — it never appears in this channel or in your context.
        Once added, the credential becomes usable by everyone who talks to
        ``agent_name``.
        """
        return await _request_mcp_credential_impl(
            runtime,
            await _auth(ctx),
            agent_name=agent_name,
            server_name=server_name,
            url=url,
            channel_id=channel_id,
        )
