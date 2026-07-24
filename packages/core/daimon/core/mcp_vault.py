"""Idempotent per-agent MA vault bootstrap for the daimon-mcp credential.

One vault per agent, named `daimon-mcp:<account_uuid>:<agent_uuid>`. Cold path
creates vault + single static_bearer credential pointing at `public_url`.
Warm path returns the oldest existing vault with the matching display name
(MA enforces no uniqueness — verified against the live MA API).

The get-or-create critical section (list-then-create) is serialized per
(account_id, agent_id) via a blocking `pg_advisory_xact_lock` — MA has no
server-side uniqueness constraint on `display_name`, so two concurrent
callers racing the empty-list check would otherwise each create a vault.

Shell function; injects `AsyncAnthropic` and `now`. No global state, no
adapter imports. Called from `daimon.core.sessions.create_session`
conditionally when `settings.mcp.public_url` is set.
"""

from __future__ import annotations

import datetime as dt
import uuid

from anthropic import AsyncAnthropic
from anthropic.types.beta import BetaManagedAgentsVault
from daimon.core.mcp_auth import mint_jwt
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

GITHUB_COPILOT_MCP_URL = "https://api.githubcopilot.com/mcp"

# Namespace prefix for the pg_advisory_xact_lock key — keeps this subsystem's
# lock keys distinct from any other advisory-lock user in the same database.
LOCK_NAMESPACE = "mcp_vault"


async def _find_vault_by_name(
    client: AsyncAnthropic, *, display_name: str
) -> BetaManagedAgentsVault | None:
    """Return the oldest vault whose ``display_name`` matches, or ``None``.

    MA enforces no uniqueness on ``display_name`` — verified against the live
    MA API — so more than one vault can share a name; the oldest is always
    the canonical one (shared list-then-filter logic for both callers below).
    """
    matching = [v async for v in client.beta.vaults.list() if v.display_name == display_name]
    if not matching:
        return None
    return min(matching, key=lambda v: v.created_at)


async def _lock_vault_namespace(
    session: AsyncSession,
    *,
    account_id: uuid.UUID,
    agent_id: uuid.UUID,
) -> None:
    """Acquire the blocking, transaction-scoped advisory lock for this
    (account_id, agent_id)'s vault get-or-create critical section.

    Blocking, not try-lock: the second concurrent caller must WAIT for the
    first to finish creating the vault, then proceed to see it on its own
    (post-lock) list call — a non-blocking try-lock would let the loser
    proceed with no vault_id, which is worse than the race it replaces.
    Hashed inside Postgres (``hashtextextended``), never with Python's
    ``hash()`` (PYTHONHASHSEED-salted, unstable across processes).
    """
    lock_key = f"{LOCK_NAMESPACE}:{account_id}:{agent_id}"
    await session.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
        {"key": lock_key},
    )


async def _ensure_agent_mcp_vault_locked(
    client: AsyncAnthropic,
    *,
    account_id: uuid.UUID,
    agent_id: uuid.UUID,
    jwt_secret: bytes,
    public_url: str,
    now: dt.datetime,
) -> str:
    """Core get-or-create body, assuming the caller already holds the
    per-(account_id, agent_id) advisory lock for this transaction.

    Factored out so ``add_external_mcp_credential``'s bootstrap branch can
    invoke this directly (same already-locked session) instead of calling
    the public, lock-acquiring ``ensure_agent_mcp_vault`` — nesting a second
    lock-acquisition attempt on the same key from a *different* connection
    would deadlock against the outer transaction's own held lock.
    """
    display_name = f"daimon-mcp:{account_id}:{agent_id}"
    oldest = await _find_vault_by_name(client, display_name=display_name)
    if oldest is not None:
        # Inspect existing static_bearer credentials. Only act on URL mismatch:
        # no credential matches `public_url` — stale URL from an earlier deploy
        # (e.g. cloudflare-tunnel → fly URL migration), leaves MA unable to find
        # a credential for the agent's current mcp_server URL.
        # MA blocks PATCH (405) and POST-duplicate (409), so the only way to
        # update a credential is delete + recreate (verified against the live MA API).
        has_matching_url = False
        async for cred in client.beta.vaults.credentials.list(vault_id=oldest.id):
            if cred.auth.type != "static_bearer":
                continue
            if cred.auth.mcp_server_url == public_url:
                has_matching_url = True
                break
        url_mismatch = not has_matching_url
        if url_mismatch:
            # No credential for the current URL yet — create fresh.
            await client.beta.vaults.credentials.create(
                vault_id=oldest.id,
                auth={
                    "type": "static_bearer",
                    "mcp_server_url": public_url,
                    "token": mint_jwt(
                        account_id=account_id,
                        secret=jwt_secret,
                        now=now,
                    ),
                },
            )
        return oldest.id

    vault = await client.beta.vaults.create(display_name=display_name)
    token = mint_jwt(
        account_id=account_id,
        secret=jwt_secret,
        now=now,
    )
    await client.beta.vaults.credentials.create(
        vault_id=vault.id,
        auth={
            "type": "static_bearer",
            "mcp_server_url": public_url,
            "token": token,
        },
    )
    return vault.id


async def ensure_agent_mcp_vault(
    client: AsyncAnthropic,
    *,
    account_id: uuid.UUID,
    agent_id: uuid.UUID,
    jwt_secret: bytes,
    public_url: str,
    now: dt.datetime,
    session_factory: async_sessionmaker[AsyncSession],
) -> str:
    """Return the ``ma_vault_id`` for this agent's daimon-mcp vault.

    Creates the vault + credential on cold path. On warm path, only acts when
    there is no credential matching the current ``public_url`` (URL-drift case,
    e.g. cloudflare-tunnel → fly URL migration) — a fresh credential is created
    at the new URL; the prior credential is left as an inert orphan.

    The long-lived credential is always non-admin and never carries the ``internal``
    discriminator claim — admin is resolved live from the DB ``role`` by the verifier
    on each request (ADMIN-01). A Discord vault token's baked ``is_admin`` claim alone
    never elevates a non-admin caller at the MCP gate (#162 escalation closed).

    Per-turn delete+recreate (the old re-stamp limb) is intentionally removed (Phase
    88-03 T-88-03-02): the credential is identity-stable (``sub`` = account, no
    ``is_admin``, no ``internal``), so nothing per-turn needs to mutate it. Removing
    the re-stamp limb eliminates the cross-thread race where an in-flight session
    re-reads the shared per-(account,agent) vault credential mid-turn (A3).

    We do NOT delete credentials on URL drift. The vault is shared with user-added
    external MCP credentials (``add_external_mcp_credential``) whose URLs we cannot
    authenticate as "ours". Deleting any cred that doesn't match the current
    ``public_url`` would silently nuke user data on the first deploy-URL change.
    Cost: O(deploys-with-URL-change) orphan creds per agent, bounded and harmless.

    The daimon-mcp JWT claims are account-scoped only — no agent claim is added (SC-4).
    Only the vault's storage location is per-agent.

    The entire list-then-create body runs inside a blocking Postgres
    advisory-transaction lock keyed on ``(account_id, agent_id)`` (SYNC-01):
    two concurrent callers for the same agent can never both pass the
    empty-match check and each create a vault — MA has no server-side
    uniqueness constraint on ``display_name`` to fall back on.
    """
    async with session_factory() as session, session.begin():
        await _lock_vault_namespace(session, account_id=account_id, agent_id=agent_id)
        return await _ensure_agent_mcp_vault_locked(
            client,
            account_id=account_id,
            agent_id=agent_id,
            jwt_secret=jwt_secret,
            public_url=public_url,
            now=now,
        )


async def add_github_copilot_credential(
    client: AsyncAnthropic,
    *,
    vault_id: str,
    token: str,
) -> None:
    """Create or replace the GitHub Copilot MCP `static_bearer` credential.

    The default GitHub Copilot MCP server (per MA SDK) consumes credentials
    via vault-injection at `mcp_server_url=https://api.githubcopilot.com/mcp`.
    This is the second credential in the per-agent vault — the first is the
    daimon-mcp JWT placed by `ensure_agent_mcp_vault`.

    The caller (OAuth callback) runs this best-effort.
    If it raises, the local Fernet blob is already the source of truth;
    the operator can retry by re-OAuthing.

    Idempotent on retry: list existing credentials, delete any pointed at the
    GitHub Copilot URL, then create the new one.
    """
    async for cred in client.beta.vaults.credentials.list(vault_id=vault_id):
        # SDK 0.117 widened the auth union with an `environment_variable`
        # member that has no `mcp_server_url`; narrow to static_bearer first
        # (these creds are created as static_bearer below), matching the guard
        # in ensure_agent_mcp_vault.
        if cred.auth.type != "static_bearer":
            continue
        if cred.auth.mcp_server_url == GITHUB_COPILOT_MCP_URL:
            await client.beta.vaults.credentials.delete(
                cred.id,
                vault_id=vault_id,
            )

    await client.beta.vaults.credentials.create(
        vault_id=vault_id,
        auth={
            "type": "static_bearer",
            "mcp_server_url": GITHUB_COPILOT_MCP_URL,
            "token": token,
        },
    )


async def add_external_mcp_credential(
    client: AsyncAnthropic,
    *,
    account_id: uuid.UUID,
    agent_id: uuid.UUID,
    jwt_secret: bytes,
    public_url: str,
    now: dt.datetime,
    mcp_server_url: str,
    token: str,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Create or replace a `static_bearer` credential in the caller's
    per-agent vault for an external (user-supplied) MCP server.

    Looks up the vault by ``display_name == f"daimon-mcp:{account_id}:{agent_id}"``.
    Bootstraps the per-agent vault (creates it + mints the daimon-mcp JWT) when
    it does not yet exist — no longer raises on missing vault.

    Idempotent on retry: deletes any existing ``static_bearer`` credential
    whose ``auth.mcp_server_url`` matches ``mcp_server_url``, then creates the
    new one. Mirrors ``add_github_copilot_credential`` but accepts the URL as
    a parameter (per-user MCP servers each have distinct URLs).

    The entire list-then-create body (including the bootstrap branch) runs
    inside the same blocking Postgres advisory-transaction lock as
    ``ensure_agent_mcp_vault``, keyed on ``(account_id, agent_id)`` (SYNC-01) —
    it shares the vault get-or-create race with that function and must
    serialize against it, not just against itself.
    """
    display_name = f"daimon-mcp:{account_id}:{agent_id}"
    async with session_factory() as session, session.begin():
        await _lock_vault_namespace(session, account_id=account_id, agent_id=agent_id)
        oldest = await _find_vault_by_name(client, display_name=display_name)
        if oldest is not None:
            vault_id = oldest.id
        else:
            # Bootstrap: create the per-agent vault + daimon-mcp JWT when it
            # doesn't exist yet. Already holding the lock acquired above —
            # call the locked-core helper directly, NOT the public
            # ensure_agent_mcp_vault (which would try to re-acquire the same
            # lock key from a fresh connection and deadlock against this one).
            vault_id = await _ensure_agent_mcp_vault_locked(
                client,
                account_id=account_id,
                agent_id=agent_id,
                jwt_secret=jwt_secret,
                public_url=public_url,
                now=now,
            )

        async for cred in client.beta.vaults.credentials.list(vault_id=vault_id):
            if cred.auth.type != "static_bearer":
                continue
            if cred.auth.mcp_server_url == mcp_server_url:
                await client.beta.vaults.credentials.delete(cred.id, vault_id=vault_id)

        await client.beta.vaults.credentials.create(
            vault_id=vault_id,
            auth={
                "type": "static_bearer",
                "mcp_server_url": mcp_server_url,
                "token": token,
            },
        )
