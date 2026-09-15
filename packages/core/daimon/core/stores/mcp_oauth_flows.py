"""Async store for mcp_oauth_flows — one row per in-flight MCP OAuth authorization.

The click mints the row (`create_flow`), `/oauth/mcp/start` fills in the
registered client (`save_flow_client`), and the callback spends it
(`consume_flow`), which is the single-use gate: one UPDATE whose WHERE
clause only matches an unused, unexpired row. No try/except — DB exceptions
propagate.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from daimon.core._models import McpOAuthFlow
from daimon.core.stores.domain import McpOAuthFlowRow
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession


async def create_flow(
    session: AsyncSession,
    *,
    state: str,
    request_token: str,
    tenant_id: uuid.UUID,
    account_id: uuid.UUID,
    agent_id: uuid.UUID,
    server_name: str,
    mcp_server_url: str,
    redirect_uri: str,
    code_verifier: str,
    expires_at: datetime,
) -> McpOAuthFlowRow:
    orm = McpOAuthFlow(
        state=state,
        request_token=request_token,
        tenant_id=tenant_id,
        account_id=account_id,
        agent_id=agent_id,
        server_name=server_name,
        mcp_server_url=mcp_server_url,
        redirect_uri=redirect_uri,
        code_verifier=code_verifier,
        expires_at=expires_at,
    )
    session.add(orm)
    await session.flush()
    return McpOAuthFlowRow.model_validate(orm)


async def get_flow(session: AsyncSession, *, state: str) -> McpOAuthFlowRow | None:
    """The row for `state` regardless of lifecycle, or None when unknown."""
    orm = await session.scalar(select(McpOAuthFlow).where(McpOAuthFlow.state == state))
    return None if orm is None else McpOAuthFlowRow.model_validate(orm)


async def save_flow_client(
    session: AsyncSession,
    *,
    state: str,
    client_id: str,
    client_secret_encrypted: str | None,
    token_endpoint_auth_method: str,
    token_endpoint: str,
    resource: str | None,
    scope: str | None,
) -> McpOAuthFlowRow | None:
    """Record the discovered endpoints and registered client on an unspent row."""
    stmt = (
        update(McpOAuthFlow)
        .where(McpOAuthFlow.state == state, McpOAuthFlow.used_at.is_(None))
        .values(
            client_id=client_id,
            client_secret_encrypted=client_secret_encrypted,
            token_endpoint_auth_method=token_endpoint_auth_method,
            token_endpoint=token_endpoint,
            resource=resource,
            scope=scope,
        )
        .returning(McpOAuthFlow)
    )
    orm = (await session.execute(stmt)).scalar_one_or_none()
    await session.flush()
    return None if orm is None else McpOAuthFlowRow.model_validate(orm)


async def consume_flow(
    session: AsyncSession, *, state: str, now: datetime
) -> McpOAuthFlowRow | None:
    """Atomically spend `state` iff unused and unexpired; None otherwise."""
    stmt = (
        update(McpOAuthFlow)
        .where(
            McpOAuthFlow.state == state,
            McpOAuthFlow.used_at.is_(None),
            McpOAuthFlow.expires_at > now,
        )
        .values(used_at=now)
        .returning(McpOAuthFlow)
    )
    orm = (await session.execute(stmt)).scalar_one_or_none()
    await session.flush()
    return None if orm is None else McpOAuthFlowRow.model_validate(orm)
