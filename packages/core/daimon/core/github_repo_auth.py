"""Clone-credential resolution for App-or-PAT repo auth.

Owns the mode decision (pure) and the token-resolution orchestration + panel
coverage probe (shell, injected httpx). No DB access, no module-level
singletons — callers inject the `httpx.AsyncClient` and already-resolved
per-agent PAT.

Pure function:
  select_clone_auth — deterministic pat/app/public/none decision table.

Shell functions (injected httpx):
  resolve_clone_token — PAT short-circuit (zero GitHub I/O) -> App
    installation-token mint (only for a binding with recorded proof) ->
    operator fallback PAT (only for a binding with a recorded
    verified-public proof) -> raise. Never returns an empty string.
  is_app_installed_for_repo — bind-time App-coverage probe for setup panels;
    returns False (never raises) when App creds are unset.
"""

from __future__ import annotations

from typing import Literal

import httpx
import structlog
from daimon.core.errors import DaimonError
from daimon.core.github_app_auth import (
    build_app_jwt,
    get_installation_id_for_repo,
    mint_installation_token,
)
from daimon.core.stores.domain import AgentRepoBindingRow, RepoProofKind
from pydantic import SecretStr

log = structlog.get_logger()

__all__ = ["select_clone_auth", "resolve_clone_token", "is_app_installed_for_repo"]


def select_clone_auth(
    *,
    has_per_agent_pat: bool,
    app_installed: bool,
    proof_kind: RepoProofKind | None,
    has_fallback_pat: bool,
) -> Literal["pat", "app", "public", "none"]:
    """Decide the clone-auth mode per the precedence table.

    Order:
      1. A per-agent PAT always wins and needs no proof — it IS the
         credential, and the caller supplied it directly. GitHub itself
         enforces what it can read.
      2. An App installation token requires that the binding recorded proof
         the binder could read the repo (``proof_kind is not None``). App
         installation coverage on its own says nothing about who bound the
         repo: the deployment's App is installed by repo owners for their
         own use, not by the tenant binding the repo, so coverage alone
         cannot stand in for a demonstrated access check.
      3. The operator fallback token requires specifically a verified-public
         proof kind. That token is public-read-only; serving it for anything
         else (including a PAT-kind proof, which only demonstrates the
         binder could read the repo with a token, not that the repo is
         public) produces a clone that cannot work.
      4. Otherwise none — caller must raise; never emit an empty
         ``authorization_token``.

    ``proof_kind`` is the RECORDED proof read off the binding row, never a
    fresh probe — proof is established once at bind time and is not
    re-verified here or by any caller of this function.

    Pure — no I/O, no clock. Callers resolve ``app_installed`` (an
    installation lookup) and ``has_fallback_pat`` before calling this.
    """
    if has_per_agent_pat:
        return "pat"
    if app_installed and proof_kind is not None:
        return "app"
    if proof_kind == "public" and has_fallback_pat:
        return "public"
    return "none"


async def resolve_clone_token(
    http_client: httpx.AsyncClient,
    *,
    binding: AgentRepoBindingRow,
    per_agent_pat: str | None,
    fallback_pat: str | None,
    app_id: str | None,
    app_private_key: SecretStr | None,
    now: int,
) -> str:
    """Resolve the clone token for a bound repo.

    Short-circuits on ``per_agent_pat`` before any GitHub HTTP call (per-agent PAT
    wins; Pitfall 2 — never mint a JWT / do an installation lookup when a PAT
    is already available). Otherwise attempts the App installation-token
    path (on-demand ``GET /repos/{owner}/{repo}/installation`` ->
    ``mint_installation_token``) when the binding recorded proof of access,
    falls back to the operator fallback PAT on a binding that recorded a
    verified-public proof, and raises ``DaimonError`` when none of those
    apply. Never returns an empty string (MA rejects an empty
    ``authorization_token`` with a 400).

    Args:
        http_client: Injected async HTTP client. Caller owns lifecycle.
        binding: The agent's repo binding (repo_url is canonical owner/repo).
        per_agent_pat: Already-resolved per-agent PAT overlay, or None.
        fallback_pat: Operator-wide fallback PAT, or None/empty (treated the
            same — an empty string is "no token").
        app_id: GitHub App id, or None if the App is not configured.
        app_private_key: GitHub App private key, or None if not configured.
        now: Current Unix timestamp (int) — caller provides this so the App
            JWT mint stays pure (no clock read inside this module).

    Returns:
        The resolved clone token (PAT or minted installation token).

    Raises:
        DaimonError: When no PAT, App coverage, or fallback PAT resolves —
            the fail-loud branch.
    """
    # Empty string is "no token" (same as fallback_pat's bool() handling below);
    # returning it verbatim would emit an empty authorization_token (MA 400s).
    if per_agent_pat:
        return per_agent_pat

    owner, repo = binding.repo_url.split("/", 1)
    # The legacy no-token string tag means "no PAT was supplied at bind
    # time" — it is not evidence the repo is public. The binding's recorded
    # proof is; select_clone_auth reads it directly, below.
    has_fallback_pat = bool(fallback_pat)

    # Best-effort App path: a transient GitHub failure on the installation
    # lookup / token mint (e.g. a 403 secondary-rate-limit, common once the App
    # has many installs) must not take down a clone that a public+fallback-PAT
    # binding could still serve. On any HTTP error, degrade to "App unavailable"
    # and fall through to the fallback/none decision. A malformed App private
    # key still raises from build_app_jwt (operator misconfig — fail loud so it
    # is fixed, not silently masked). A private binding with no App and no PAT
    # still fails loudly at the raise below.
    app_token: str | None = None
    if app_id is not None and app_private_key is not None:
        app_jwt = build_app_jwt(app_private_key.get_secret_value(), app_id, now=now)
        try:
            installation_id = await get_installation_id_for_repo(
                http_client, jwt=app_jwt, owner=owner, repo=repo
            )
            if installation_id is not None:
                app_token = await mint_installation_token(
                    http_client, jwt=app_jwt, installation_id=installation_id
                )
        except httpx.HTTPError as err:
            log.warning(
                "github_repo_auth.app_path_failed",
                repo_url=binding.repo_url,
                error=str(err),
            )

    mode = select_clone_auth(
        has_per_agent_pat=False,
        app_installed=app_token is not None,
        proof_kind=binding.proof_kind,
        has_fallback_pat=has_fallback_pat,
    )

    if mode == "app":
        assert app_token is not None  # narrows: app_installed implies a minted token
        return app_token
    if mode == "public":
        assert fallback_pat  # narrows: has_fallback_pat implies this is truthy
        return fallback_pat
    raise DaimonError(
        f"No credential is authorized to clone {binding.repo_url}. Re-bind this repo "
        "with a GitHub token that can read it, from the agent setup panel's GitHub option."
    )


async def is_app_installed_for_repo(
    http_client: httpx.AsyncClient,
    *,
    app_id: str | None,
    app_private_key: SecretStr | None,
    owner: str,
    repo: str,
    now: int,
) -> bool:
    """Bind-time App-coverage probe for setup panels.

    Returns False (never raises) when App creds are unset, so panels on
    non-App deployments just show the PAT path.

    Args:
        http_client: Injected async HTTP client. Caller owns lifecycle.
        app_id: GitHub App id, or None if the App is not configured.
        app_private_key: GitHub App private key, or None if not configured.
        owner: Repository owner (org or user login).
        repo: Repository name (no owner prefix).
        now: Current Unix timestamp (int).

    Returns:
        True if the App is installed on the repo, False otherwise (including
        when App creds are unset).
    """
    if app_id is None or app_private_key is None:
        return False
    jwt = build_app_jwt(app_private_key.get_secret_value(), app_id, now=now)
    installation_id = await get_installation_id_for_repo(
        http_client, jwt=jwt, owner=owner, repo=repo
    )
    return installation_id is not None
