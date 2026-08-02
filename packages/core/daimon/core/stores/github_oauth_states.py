"""OAuth state GDPR helpers — the OAuth flow

itself (create/peek/get_by_state/consume), but legacy `github_oauth_states`
rows may still exist from before the removal and must stay purgeable. These
two read/delete helpers are retained solely for `daimon.core.privacy` /
`daimon.core.purge` erasure of that legacy PII. No try/except —
exceptions propagate.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

from daimon.core._models import GitHubOauthState
from sqlalchemy import CursorResult, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession


async def delete_states_for_platform_user(
    session: AsyncSession,
    *,
    platform: str,
    platform_user_id: str,
    tenant_id: uuid.UUID,
) -> int:
    """Delete ALL oauth-state rows for a (platform, platform_user_id). Idempotent.

    Returns rowcount; never raises on 0. Used by the GDPR purge orchestrator.

    Deliberately does NOT filter consumed_at or _cutoff() — consumed and expired
    handshake rows still carry platform_user_id PII and must be erased. This
    mirrors the design of get_by_state, which also omits the TTL/consumed filters
    to access the full row regardless of its lifecycle state.

    Callers pass platform="cli" with platform_user_id=<os_user> for CLI principals
    (the CLI auth flow writes such rows), as well as real platform/external_id pairs
    for platform principals.

    `tenant_id` is required: neither `os_user` (CLI) nor a platform `external_id`
    is globally unique. Two machines can both be `ubuntu`, and Slack user ids are
    workspace-scoped (`U123` in two workspaces are two different humans), so a
    tenant-agnostic delete would erase another tenant's in-flight handshake rows.
    A deliberate cross-tenant sweep of ghost rows under stale tenant_ids (re-key
    drift) needs a separate, explicitly-named helper — it must not be this one.
    """
    result = await session.execute(
        delete(GitHubOauthState).where(
            GitHubOauthState.platform == platform,
            GitHubOauthState.platform_user_id == platform_user_id,
            GitHubOauthState.tenant_id == tenant_id,
        )
    )
    rowcount = cast(CursorResult[Any], result).rowcount
    await session.flush()
    return rowcount


async def count_states_for_platform_user(
    session: AsyncSession,
    *,
    platform: str,
    platform_user_id: str,
    tenant_id: uuid.UUID,
) -> int:
    """Count oauth-state rows that `delete_states_for_platform_user` would delete. Read-only.

    `tenant_id` is required and always applied — the same predicate the delete
    uses, unconditionally, so the preview cannot diverge from the purge (parity
    contract in daimon.core.privacy). There is no optional-scope branch to
    misuse: a caller that genuinely needs a cross-tenant count needs a separate,
    explicitly-named helper, not a `None` passed here.
    """
    stmt = (
        select(func.count())
        .select_from(GitHubOauthState)
        .where(
            GitHubOauthState.platform == platform,
            GitHubOauthState.platform_user_id == platform_user_id,
            GitHubOauthState.tenant_id == tenant_id,
        )
    )
    return int((await session.execute(stmt)).scalar_one())
