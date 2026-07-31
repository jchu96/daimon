"""daimon repo-bindings — operator visibility into the proof-of-access model.

Every `agent_repo_binding` row written before the proof migration has no
recorded proof; the fail-closed gate this phase installs stops treating the
legacy `anon:` tag as evidence a repo is public. Two subcommands give an
operator what they need to survive that transition without a database
console:

- `list-proofless`: every binding with no recorded proof, across every
  install, optionally classified by an unauthenticated visibility probe.
- `backfill-public-proof`: a one-shot idempotent stamp of verified-public
  proof onto no-token bindings whose repo actually probes public.

Both commands use only `is_public_repo` — an unauthenticated GET. Neither
sends a GitHub credential of any kind; that absence is the entire safety
argument for a command that touches every tenant's data in one run.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

import httpx
import typer
from daimon.adapters.cli.errors import run_cli
from daimon.adapters.cli.flags import YES_OPTION
from daimon.adapters.cli.output import emit_rows
from daimon.adapters.cli.prompt import confirm_or_abort
from daimon.adapters.cli.runtime import CliRuntime, build_runtime
from daimon.core.config import load_settings
from daimon.core.github_visibility import is_public_repo
from daimon.core.stores.agent_repo_binding import list_bindings_without_proof, record_proof
from daimon.core.stores.domain import RepoAccessProof
from pydantic import BaseModel
from rich.console import Console

repo_bindings_app = typer.Typer(help="Operator visibility into repo-binding proof state.")

Visibility = Literal["public", "not-public", "probe-error"]
Classification = Literal["STAMP", "SKIP"]


class _ProoflessRow(BaseModel):
    """Report row for `list-proofless`."""

    tenant_id: str
    agent_id: str
    repo_url: str
    ma_secret_ref: str
    created_at: str
    visibility: str  # empty string when --check-visibility was not passed


class _BackfillPlanRow(BaseModel):
    """Report row for `backfill-public-proof`'s plan phase."""

    tenant_id: str
    agent_id: str
    repo_url: str
    classification: Classification


# ---------------------------------------------------------------------------
# list-proofless
# ---------------------------------------------------------------------------


@repo_bindings_app.command("list-proofless")
def list_proofless_command(
    check_visibility: Annotated[
        bool,
        typer.Option("--check-visibility", help="Probe each distinct repo's public visibility."),
    ] = False,
) -> None:
    """List every repo binding with no recorded proof, across every install.

    Read-only — writes nothing. With --check-visibility, additionally probes
    each DISTINCT repo_url exactly once (never once per binding, so an
    install with many agents on one repo does not multiply requests against
    the unauthenticated rate limit) and reports 'public', 'not-public', or
    'probe-error' per row.
    """
    settings = load_settings()
    console = Console(highlight=False)

    async def _run() -> None:
        async with build_runtime(settings) as rt:
            await list_proofless(rt=rt, console=console, check_visibility=check_visibility)

    run_cli(_run(), console=console)


async def list_proofless(
    *,
    rt: CliRuntime,
    console: Console,
    check_visibility: bool,
    http_client: httpx.AsyncClient | None = None,
) -> None:
    """List every proof-NULL binding; optionally classify each distinct repo's visibility."""
    async with rt.sessionmaker() as session:
        bindings = await list_bindings_without_proof(session)

    visibility_by_repo: dict[str, Visibility] = {}
    if check_visibility:

        async def _probe_all(client: httpx.AsyncClient) -> None:
            for repo_url in sorted({b.repo_url for b in bindings}):
                visibility_by_repo[repo_url] = await _probe_visibility(client, repo_url)

        if http_client is not None:
            await _probe_all(http_client)
        else:
            async with httpx.AsyncClient(timeout=30.0) as owned:
                await _probe_all(owned)

    rows = [
        _ProoflessRow(
            tenant_id=str(binding.tenant_id),
            agent_id=str(binding.agent_id),
            repo_url=binding.repo_url,
            ma_secret_ref=binding.ma_secret_ref,
            created_at=binding.created_at.isoformat(),
            visibility=visibility_by_repo.get(binding.repo_url, "") if check_visibility else "",
        )
        for binding in bindings
    ]

    columns = ("tenant_id", "agent_id", "repo_url", "ma_secret_ref", "created_at")
    if check_visibility:
        columns = (*columns, "visibility")
    emit_rows(console, rows, columns=columns, as_json=False)

    if check_visibility:
        counts: dict[str, int] = {}
        for row in rows:
            counts[row.visibility] = counts.get(row.visibility, 0) + 1
        summary = ", ".join(f"{bucket}={count}" for bucket, count in sorted(counts.items()))
        console.print(f"[dim]{summary or 'no bindings'}[/dim]")


async def _probe_visibility(http_client: httpx.AsyncClient, repo_url: str) -> Visibility:
    """Classify one repo for the listing. A raised probe surfaces as its own bucket."""
    try:
        public = await is_public_repo(http_client, owner_repo=repo_url)
    except httpx.HTTPError:
        return "probe-error"
    return "public" if public else "not-public"


# ---------------------------------------------------------------------------
# backfill-public-proof
# ---------------------------------------------------------------------------


@repo_bindings_app.command("backfill-public-proof")
def backfill_public_proof_command(
    yes: Annotated[bool, YES_OPTION] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be stamped without writing."),
    ] = False,
) -> None:
    """Stamp verified-public proof onto no-token bindings whose repo probes public.

    Three narrowings, each load-bearing:

    - Only `ma_secret_ref == "anon:"` bindings are candidates. A binding
      backed by a token resolves that token at clone time regardless of
      proof, so stamping it would record a claim nobody checked.
    - Only a repo that probes public is stamped. A 404 counts as not-public,
      exactly as the probe itself treats it — an unauthenticated caller
      cannot distinguish a private repo from a missing one and must not
      guess in the permissive direction.
    - The stamped proof's account is left unset, because no person asserted
      this claim; an unauthenticated probe did.

    Idempotent: list_bindings_without_proof only ever returns proof-NULL
    rows, so a stamped row drops out of the candidate set on the next run —
    a second invocation reports zero rows to stamp and sends zero requests.
    """
    settings = load_settings()
    console = Console(highlight=False)

    async def _run() -> None:
        async with build_runtime(settings) as rt:
            await backfill_public_proof(rt=rt, console=console, yes=yes, dry_run=dry_run)

    run_cli(_run(), console=console)


async def backfill_public_proof(
    *,
    rt: CliRuntime,
    console: Console,
    yes: bool,
    dry_run: bool,
    http_client: httpx.AsyncClient | None = None,
) -> None:
    """Plan phase (always) + apply phase (unless --dry-run)."""
    async with rt.sessionmaker() as session:
        bindings = await list_bindings_without_proof(session)
    candidates = [b for b in bindings if b.ma_secret_ref == "anon:"]

    if not candidates:
        console.print("[green]✓ No proofless anon: bindings to backfill.[/green]")
        return

    async def _classify_all(client: httpx.AsyncClient) -> dict[str, Classification]:
        classification_by_repo: dict[str, Classification] = {}
        for repo_url in sorted({b.repo_url for b in candidates}):
            classification_by_repo[repo_url] = await _classify_repo(client, repo_url)
        return classification_by_repo

    if http_client is not None:
        classification_by_repo = await _classify_all(http_client)
    else:
        async with httpx.AsyncClient(timeout=30.0) as owned:
            classification_by_repo = await _classify_all(owned)

    plan_rows = [
        _BackfillPlanRow(
            tenant_id=str(binding.tenant_id),
            agent_id=str(binding.agent_id),
            repo_url=binding.repo_url,
            classification=classification_by_repo[binding.repo_url],
        )
        for binding in candidates
    ]
    stamp_candidates = [b for b in candidates if classification_by_repo[b.repo_url] == "STAMP"]
    n_skip = len(plan_rows) - len(stamp_candidates)

    emit_rows(
        console,
        plan_rows,
        columns=("tenant_id", "agent_id", "repo_url", "classification"),
        as_json=False,
    )
    console.print(f"[dim]{len(stamp_candidates)} to STAMP, {n_skip} to SKIP[/dim]")

    if dry_run:
        console.print("[yellow]dry-run: nothing written[/yellow]")
        return

    if not stamp_candidates:
        console.print("[green]✓ Nothing to stamp.[/green]")
        return

    confirm_or_abort(
        console, f"Stamp verified-public proof on {len(stamp_candidates)} binding(s)?", yes=yes
    )

    now = datetime.now(UTC)
    async with rt.sessionmaker() as session, session.begin():
        for binding in stamp_candidates:
            await record_proof(
                session,
                tenant_id=binding.tenant_id,
                agent_id=binding.agent_id,
                proof=RepoAccessProof(kind="public", at=now, account_id=None),
            )

    console.print(f"[green]✓ Stamped {len(stamp_candidates)} binding(s).[/green]")


async def _classify_repo(http_client: httpx.AsyncClient, repo_url: str) -> Classification:
    """Classify one candidate repo for the backfill plan. A raised probe is SKIP, not a crash."""
    try:
        public = await is_public_repo(http_client, owner_repo=repo_url)
    except httpx.HTTPError:
        return "SKIP"
    return "STAMP" if public else "SKIP"
