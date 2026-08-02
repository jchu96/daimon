"""Clone-credential resolution for App-or-PAT repo auth.

Owns the mode decisions (pure) and the token-resolution orchestrations
(shell, injected httpx) for BOTH of daimon's independent GitHub-repo-auth
callers: the per-agent clone path (`agent_repo_binding`, a bound repo with a
recorded proof of access) and the skill-sync path (a bare repo URL that may
have no binding row at all). No DB access, no module-level singletons —
callers inject the `httpx.AsyncClient` and already-resolved per-agent PAT.

Pure functions:
  select_clone_auth — deterministic pat/app/public/none decision table for a
    bound repo, gated on a recorded proof of access.
  select_skill_sync_auth — the skill-sync sibling: same tier ORDERING
    (per-agent token -> App installation -> operator fallback -> anonymous),
    but no proof kind, because a skill-sync URL may have no binding row.
  normalize_owner_repo — public `owner/repo` normalizer shared by both shell
    resolvers below (existing private copies elsewhere are untouched by this
    module; see each copy's own docstring).

Shell functions (injected httpx):
  resolve_clone_token — PAT short-circuit (zero GitHub I/O) -> App
    installation-token mint (only for a binding with recorded proof) ->
    operator fallback PAT (only for a binding with a recorded
    verified-public proof) -> raise. Never returns an empty string.
  resolve_skill_sync_token — PAT short-circuit (zero GitHub I/O) -> App
    installation-token mint (only when an injected installation lookup is
    supplied and resolves) -> operator fallback PAT -> None (anonymous
    fetch). Never returns an empty string.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
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

__all__ = [
    "select_clone_auth",
    "resolve_clone_token",
    "select_skill_sync_auth",
]

type InstallationLookup = Callable[[str, str], Awaitable[int | None]]
"""Injected `(owner, repo) -> installation_id | None` lookup.

Lets the caller choose freshness: an interactive caller injects a live
GitHub lookup (`get_installation_id_for_repo`); an unattended batch caller
injects a cheap read of the cached `github_app_installations` table. That
choice belongs to the caller, not to `resolve_skill_sync_token`.
"""


def normalize_owner_repo(url: str) -> str:
    """Extract `owner/repo` from a URL or short-form path.

    Accepts: 'https://github.com/owner/repo', 'github.com/owner/repo',
    'owner/repo', any with a trailing '/' or '.git'.

    Public so the skill-sync credential seam can take a raw repo URL from
    either caller (the interactive MCP tool or the webhook resync batch)
    without either owning its own copy. Two private copies of this same body
    exist elsewhere in the codebase today
    (`daimon.core.skill_sync.fetcher._normalize_owner_repo` and
    `daimon.core.stores.agent_repo_binding._normalize_owner_repo`) — this
    plan does not repoint or delete either; that is out of scope here.
    """
    return (
        url.removeprefix("https://github.com/")
        .removeprefix("http://github.com/")
        .removeprefix("github.com/")
        .removesuffix(".git")
        .rstrip("/")
    )


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

    See also ``select_skill_sync_auth``, the skill-sync sibling of this
    function: it shares this exact tier ORDERING on purpose (a test asserts
    the two agree), but takes no ``proof_kind`` at all, because a skill-sync
    URL may have no binding row to record proof against. This function is
    NOT widened to accept an optional binding for that case — doing so would
    thread an implicit "no proof required" branch through a decision whose
    whole point is that proof is required, so the two stay separate
    functions differing only in their inputs.
    """
    if has_per_agent_pat:
        return "pat"
    if app_installed and proof_kind is not None:
        return "app"
    if proof_kind == "public" and has_fallback_pat:
        return "public"
    return "none"


def select_skill_sync_auth(
    *,
    has_per_agent_pat: bool,
    app_installed: bool,
    has_fallback_pat: bool,
) -> Literal["pat", "app", "public", "none"]:
    """Decide the skill-sync credential mode for a bare repo URL.

    Scope: credential selection for a skill-repo fetch — the interactive
    ``sync_skills`` MCP tool and the per-agent webhook skill-resync path
    both select a credential for a repo URL that may have no
    ``agent_repo_binding`` row at all (no recorded proof of access to key
    a decision on).

    Order, identical to ``select_clone_auth``: a per-agent token wins; else
    an App installation covering the repo; else the operator fallback
    token; else ``"none"``. This function shares ``select_clone_auth``'s
    tier ORDERING deliberately — a test
    (``test_skill_sync_and_clone_selectors_agree_on_tier_ordering``) asserts
    the two agree on every combination of the three shared booleans, so
    neither can silently drift from the other again.

    It is a separate function rather than a parameter on ``select_clone_auth``
    because the two decisions are keyed on different inputs, not because
    they disagree on ordering: the clone decision is keyed on a proof of
    access recorded at bind time, and a skill-sync URL may have no binding
    row at all, so accepting an optional binding here would introduce an
    implicit "no proof required" branch into a decision whose whole point is
    that proof is required. The difference between the two functions is
    their INPUTS, not their ordering.

    ``"none"`` here means the legitimate anonymous public fetch, not a
    refusal — unlike the clone path, which raises when its decision reaches
    ``"none"``. The shell counterpart, ``resolve_skill_sync_token``,
    converts ``"none"`` to ``None``, which the skill fetchers already accept
    as "no credential" for a public repo. This is the one behavioral
    difference from the clone path: public skill repos legitimately sync
    with no credential on a deployment that never configured an operator
    fallback token, so anonymous is not an error here the way it is for a
    bound clone.

    Pure — no I/O, no clock. Callers resolve ``app_installed`` (an
    installation lookup) and ``has_fallback_pat`` before calling this.
    """
    if has_per_agent_pat:
        return "pat"
    if app_installed:
        return "app"
    if has_fallback_pat:
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
