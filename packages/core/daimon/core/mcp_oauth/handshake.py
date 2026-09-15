"""The two halves of an MCP OAuth handshake daimon drives itself.

`begin_mcp_oauth_flow` runs in the chat adapter on the requester's click: it
spends nothing but mints the flow row (state + PKCE verifier) whose start
link only that person sees. `prepare_authorization` runs on the mcp process
when the link is opened: discovery, dynamic client registration, and the
authorize URL to redirect to. Both take their clock, randomness and clients
as arguments.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Final
from urllib.parse import quote

import httpx
from cryptography.fernet import MultiFernet
from daimon.core.github_credentials import encrypt_token
from daimon.core.mcp_oauth.discovery import discover_authorization_server, probe_mcp_server
from daimon.core.mcp_oauth.flow import build_authorization_url, generate_pkce, register_client
from daimon.core.stores import mcp_oauth_flows as flows_store
from daimon.core.stores.domain import CredentialRequestRow, McpOAuthFlowRow
from sqlalchemy.ext.asyncio import AsyncSession

#: A person has this long between clicking the card and finishing sign-in.
FLOW_TTL: Final[timedelta] = timedelta(minutes=10)
CLIENT_NAME: Final[str] = "daimon"


def invite_copy(*, server_name: str, agent_name: str) -> str:
    """The ephemeral text next to the sign-in button, identical on every platform."""
    return (
        f"Sign in to connect {server_name} to {agent_name} with your account. "
        "This link is yours alone and works for ten minutes; nobody else can use "
        "the connection it creates."
    )


INVITE_BUTTON_LABEL: Final[str] = "Open sign-in"


def start_url(app_root_url: str, *, state: str) -> str:
    return f"{app_root_url.rstrip('/')}/oauth/mcp/start?state={quote(state, safe='')}"


def callback_url(app_root_url: str) -> str:
    return f"{app_root_url.rstrip('/')}/oauth/mcp/callback"


async def begin_mcp_oauth_flow(
    session: AsyncSession,
    *,
    request: CredentialRequestRow,
    app_root_url: str,
    now: datetime,
    state: str | None = None,
    code_verifier: str | None = None,
) -> McpOAuthFlowRow:
    """Mint the flow row for a consumed `mcp_oauth` request; return it.

    The caller has already spent the request row for its requester, so the
    identity on the flow (tenant, account, agent) is the requester's and the
    credential the callback writes lands in that person's vault only.
    """
    if request.kind != "mcp_oauth":
        raise ValueError(f"kind={request.kind!r} does not start an OAuth flow")
    if request.mcp_server_url is None:
        raise ValueError("an mcp_oauth request must name its server URL")
    pkce = generate_pkce(verifier=code_verifier)
    return await flows_store.create_flow(
        session,
        state=state if state is not None else secrets.token_urlsafe(32),
        request_token=request.token,
        tenant_id=request.tenant_id,
        account_id=request.account_id,
        agent_id=request.agent_id,
        server_name=request.target,
        mcp_server_url=request.mcp_server_url,
        redirect_uri=callback_url(app_root_url),
        code_verifier=pkce.code_verifier,
        expires_at=now + FLOW_TTL,
    )


@dataclass(frozen=True, slots=True)
class PreparedAuthorization:
    flow: McpOAuthFlowRow
    authorize_url: str


async def prepare_authorization(
    session: AsyncSession,
    http: httpx.AsyncClient,
    *,
    flow: McpOAuthFlowRow,
    fernet: MultiFernet,
) -> PreparedAuthorization | None:
    """Discover the server's authorization server, register a client, build the URL.

    Returns None when the flow was spent between the read and this write (a
    second open of the same link); the caller shows the expired page.
    """
    probe = await probe_mcp_server(http, mcp_server_url=flow.mcp_server_url)
    discovery = await discover_authorization_server(
        http,
        mcp_server_url=flow.mcp_server_url,
        resource_metadata_url=probe.resource_metadata_url,
    )
    metadata = discovery.authorization_server
    scopes = discovery.resource.scopes_supported if discovery.resource else None
    scope = " ".join(scopes) if scopes else None
    resource = discovery.resource.resource if discovery.resource else None
    client = await register_client(
        http, metadata, redirect_uri=flow.redirect_uri, client_name=CLIENT_NAME, scope=scope
    )
    saved = await flows_store.save_flow_client(
        session,
        state=flow.state,
        client_id=client.client_id,
        client_secret_encrypted=(
            encrypt_token(fernet, client.client_secret).decode()
            if client.client_secret is not None
            else None
        ),
        token_endpoint_auth_method=client.token_endpoint_auth_method,
        token_endpoint=metadata.token_endpoint,
        resource=resource,
        scope=scope,
    )
    if saved is None:
        return None
    authorize_url = build_authorization_url(
        metadata,
        client_id=client.client_id,
        redirect_uri=flow.redirect_uri,
        state=flow.state,
        code_challenge=generate_pkce(verifier=flow.code_verifier).code_challenge,
        scope=scope,
        resource=resource,
    )
    return PreparedAuthorization(flow=saved, authorize_url=authorize_url)
