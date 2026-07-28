"""Async store for credential_requests — the credential-button handshake.

No try/except anywhere in this module — exceptions propagate to the adapter
boundary. `consume_credential_request`'s single atomic UPDATE is the entire
single-use gate: its own row lock serializes concurrent clicks, so no
advisory lock (unlike the vault get-or-create critical section in
`mcp_vault.py`, which needs one because it spans a read-then-create) is
needed here.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, cast

from daimon.core._models import CredentialRequest
from daimon.core.credential_requests import CredentialRequestKind
from daimon.core.stores.domain import CredentialRequestRow
from sqlalchemy import CursorResult, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession


async def create_credential_request(
    session: AsyncSession,
    *,
    token: str,
    kind: CredentialRequestKind,
    tenant_id: uuid.UUID,
    agent_id: uuid.UUID,
    account_id: uuid.UUID,
    target: str,
    mcp_server_url: str | None,
    requester_platform_user_id: str,
    channel_id: str,
    expires_at: datetime,
) -> CredentialRequestRow:
    """Insert a fresh, unused credential-request row and return it."""
    orm = CredentialRequest(
        token=token,
        kind=kind,
        tenant_id=tenant_id,
        agent_id=agent_id,
        account_id=account_id,
        target=target,
        mcp_server_url=mcp_server_url,
        requester_platform_user_id=requester_platform_user_id,
        channel_id=channel_id,
        expires_at=expires_at,
    )
    session.add(orm)
    await session.flush()
    return CredentialRequestRow.model_validate(orm)


async def peek_credential_request(
    session: AsyncSession,
    *,
    token: str,
) -> CredentialRequestRow | None:
    """Return the row for `token`, or None if it was never minted.

    Deliberately applies no lifecycle filter (unused/unexpired) — mirrors
    `github_oauth_states.get_by_state`'s rationale: the caller needs to tell
    "expired", "already used", and "unknown token" apart, which requires the
    full row, not a filtered read that collapses all three into None.
    """
    orm = await session.get(CredentialRequest, token)
    if orm is None:
        return None
    return CredentialRequestRow.model_validate(orm)


async def consume_credential_request(
    session: AsyncSession,
    *,
    token: str,
    now: datetime,
) -> CredentialRequestRow | None:
    """Atomically mark `token` used, iff it is unused and unexpired.

    One statement is the authoritative single-use gate: the UPDATE's own row
    lock serializes concurrent consumers, so the loser's WHERE clause simply
    matches zero rows — no advisory lock needed. Returns None for "not
    consumable" (unknown token, already used, or expired); the caller cannot
    and must not distinguish those cases from this return value alone (use
    `peek_credential_request` for that).
    """
    stmt = (
        update(CredentialRequest)
        .where(
            CredentialRequest.token == token,
            CredentialRequest.used_at.is_(None),
            CredentialRequest.expires_at > now,
        )
        .values(used_at=now)
        .returning(CredentialRequest)
    )
    result = await session.execute(stmt)
    orm = result.scalar_one_or_none()
    await session.flush()
    if orm is None:
        return None
    return CredentialRequestRow.model_validate(orm)


async def delete_credential_requests_for_platform_user(
    session: AsyncSession,
    *,
    platform_user_id: str,
    tenant_id: uuid.UUID | None = None,
) -> int:
    """Delete ALL credential-request rows for `platform_user_id`. Idempotent.

    Deliberately applies no lifecycle filter — used and expired rows still
    carry `requester_platform_user_id` PII and must be erased, mirroring
    `github_oauth_states.delete_states_for_platform_user`.

    Callers should always pass `tenant_id`: a platform user id is not
    globally unique (e.g. Discord snowflakes are scoped to the platform, not
    globally deduped across a hypothetical re-key), so a tenant-agnostic
    delete risks erasing another tenant's in-flight handshake rows.
    """
    predicates = [CredentialRequest.requester_platform_user_id == platform_user_id]
    if tenant_id is not None:
        predicates.append(CredentialRequest.tenant_id == tenant_id)
    result = await session.execute(delete(CredentialRequest).where(*predicates))
    rowcount = cast(CursorResult[Any], result).rowcount
    await session.flush()
    return rowcount


async def count_credential_requests_for_platform_user(
    session: AsyncSession,
    *,
    platform_user_id: str,
    tenant_id: uuid.UUID | None = None,
) -> int:
    """Count credential-request rows that `delete_credential_requests_for_platform_user`
    would delete. Read-only. Pass the SAME `tenant_id` the delete caller uses,
    or the preview diverges from the purge (parity contract in daimon.core.privacy).
    """
    predicates = [CredentialRequest.requester_platform_user_id == platform_user_id]
    if tenant_id is not None:
        predicates.append(CredentialRequest.tenant_id == tenant_id)
    stmt = select(func.count()).select_from(CredentialRequest).where(*predicates)
    return int((await session.execute(stmt)).scalar_one())
