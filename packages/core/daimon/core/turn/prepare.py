"""Stage two of the two-stage turn chokepoint (D-01): `bind_session()`.

A single core call performs thread-session find-or-create, the full
`create_session` kwarg assembly, the `thread_sessions` mapping write, and
usage-recorder binding — returning a frozen `PreparedTurn` whose recorder is
a non-public field. Adapters never see or construct billing wiring.

`fernet=deps.fernet` is unconditional here: this is the fix for SPEC Req
7(a), the historical Slack gap where `create_session` was called without a
`fernet` argument.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import functools
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import structlog
from anthropic import APIStatusError
from anthropic.types.beta import BetaManagedAgentsSession
from anthropic.types.beta.sessions.beta_managed_agents_github_repository_resource import (
    BetaManagedAgentsGitHubRepositoryResource,
)
from daimon.core.agent_mcp_credentials import sync_agent_mcp_credentials
from daimon.core.credential_env import assemble_env_bytes
from daimon.core.ma_identity import derive_agent_uuid
from daimon.core.pricing import MODEL_PRICING
from daimon.core.session_snapshot import (
    SessionSnapshot,
    fingerprint_identity,
    fingerprint_mutable,
    hash_env_bytes,
    snapshot_from_created_session,
    snapshot_from_retrieved_session,
)
from daimon.core.sessions import create_session
from daimon.core.stores.agent_files import list_agent_files
from daimon.core.stores.domain import ThreadSessionRow
from daimon.core.stores.thread_sessions import (
    create_thread_session,
    get_live_thread_session,
    record_snapshot,
)
from daimon.core.turn.admission import Admission
from daimon.core.turn.ceiling import ceiling_error, remaining_s, turn_deadline
from daimon.core.turn.deps import TurnDeps
from daimon.core.turn.posture import UsageRecorder
from daimon.core.turn.session_identity import check_session_agent
from daimon.core.usage_recording import record_turn_usage

log = structlog.get_logger(__name__)

__all__ = ["FreshSession", "PreparedTurn", "bind_session"]


@dataclass(frozen=True)
class PreparedTurn:
    """Everything `run_prepared_turn` (06-05) needs to drive a turn.

    `_record` is intentionally underscore-prefixed and excluded from the
    public contract adapters consume — the recorder is reachable only
    through `run_prepared_turn`.
    """

    admission: Admission
    ma_session_id: str
    mapping_id: uuid.UUID | None
    watermark: str | None
    reused: bool
    session_account_id: uuid.UUID
    _record: UsageRecorder


@dataclass(frozen=True)
class FreshSession:
    """A just-created MA session, its mapping row, and the config it froze.

    `snapshot` is what the session will run for the rest of its life: MA
    freezes the agent at creation time, so this — not `admission.agent` — is
    what a later turn must bill and compare against.
    """

    ma_session_id: str
    mapping_id: uuid.UUID
    snapshot: SessionSnapshot


async def _env_bytes_sha256(
    deps: TurnDeps, *, tenant_id: uuid.UUID, agent_uuid: uuid.UUID
) -> str | None:
    """Hash of the `.env` bytes `create_session` is about to mount, or None.

    Mirrors `credential_env.upload_env_and_mount`: the same tenant-scoped rows
    through the same `assemble_env_bytes`, and None for an agent with no
    secrets (that agent gets no `.env` resource at all). Read BEFORE the
    session is created, deliberately — a key written while `create_session`
    runs then records as stale rather than as current, and a stale hash costs
    one redundant refresh, where a falsely-current one would leave the session
    silently running the old secrets.
    """
    async with deps.sessionmaker() as session:
        rows = await list_agent_files(session, tenant_id=tenant_id, agent_id=agent_uuid)
    if not rows:
        return None
    return hash_env_bytes(assemble_env_bytes(rows))


async def create_fresh_session(
    deps: TurnDeps,
    admission: Admission,
    *,
    tenant_id: uuid.UUID,
    platform: str,
    thread_id: str,
    session_account_id: uuid.UUID,
) -> FreshSession:
    """Create a brand-new MA session and its `thread_sessions` mapping row.

    The single shared `create_session` call site for a fresh session. Both
    `bind_session`'s no-live-row path and 06-05's dead-session recovery cycle
    call this helper rather than each carrying their own `create_session`
    call -- a divergent second call site is exactly the bug shape this phase
    exists to kill.

    The configuration the new session froze is snapshotted from the object
    `sessions.create` returned — the authority on what it will execute — and
    persisted on the mapping row with both fingerprints.
    """
    agent_uuid = derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=str(admission.agent.id))
    env_sha256 = await _env_bytes_sha256(deps, tenant_id=tenant_id, agent_uuid=agent_uuid)
    ma_session = await create_session(
        deps.anthropic,
        agent=admission.agent,
        environment=admission.environment,
        mcp_settings=deps.mcp,
        account_id=admission.account_id,
        tenant_id=tenant_id,
        agent_uuid=agent_uuid,
        session_factory=deps.sessionmaker,
        fernet=deps.fernet,
        github_fallback_pat=deps.github_fallback_pat,
        github_app_id=deps.github_app_id,
        github_app_private_key=deps.github_app_private_key,
    )
    ma_session_id = ma_session.id

    has_repo = any(
        isinstance(resource, BetaManagedAgentsGitHubRepositoryResource)
        for resource in ma_session.resources
    )
    snapshot = snapshot_from_created_session(
        ma_session,
        env_sha256=env_sha256,
        # Left to the builder: it reads the id off the session's own `.env`
        # file resource, which is the same object `upload_env_and_mount`
        # uploaded a moment ago.
        env_file_id=None,
        # `resolve_clone_token` minted the repo credential inside the
        # `create_session` call above, so "now" is when it was issued.
        repo_token_issued_at=int(time.time()) if has_repo else None,
        vault_id=next(iter(ma_session.vault_ids), None),
    )

    async with deps.sessionmaker() as session:
        row = await create_thread_session(
            session,
            tenant_id=tenant_id,
            platform=platform,
            thread_id=thread_id,
            account_id=session_account_id,
            ma_session_id=ma_session_id,
            ma_agent_id=admission.agent.id,
            effective_config=snapshot,
            identity_fingerprint=fingerprint_identity(snapshot),
            mutable_fingerprint=fingerprint_mutable(snapshot),
        )
        await session.commit()

    return FreshSession(ma_session_id=ma_session_id, mapping_id=row.id, snapshot=snapshot)


def bind_recorder(
    deps: TurnDeps,
    *,
    tenant_id: uuid.UUID,
    external_user_id: str,
    ma_session_id: str,
    model_id: str,
) -> UsageRecorder:
    """Build the usage recorder bound to a specific session id and its model.

    `model_id` must be the model THIS SESSION executes — the one frozen in its
    creation-time agent snapshot (`session.agent.model.id`), which is what the
    snapshot on the mapping row records. It is deliberately not the responder
    agent's current model: an `agents.update` never reaches a session that
    already exists, so a session created before a model change keeps running
    the old model, and billing the agent's new one prices work that was never
    done. `headless_runner` already binds `session.agent.model.id` for the same
    reason.

    Factored as a module-level helper (not inlined in `bind_session`) so
    06-05's dead-session recovery cycle can re-invoke it against the NEW
    session id after a recreate, rather than reusing a stale binding.
    """
    pricing = MODEL_PRICING.get(model_id)
    if pricing is None:
        # The turn still runs and still records usage; only the debit is zero.
        # Loud, because the operator is giving compute away until it is fixed.
        log.warning("billing.unpriced_model", model_id=model_id, ma_session_id=ma_session_id)
    return functools.partial(
        record_turn_usage,
        sessionmaker=deps.sessionmaker,
        platform_user_id=external_user_id,
        managed_session_id=ma_session_id,
        model_id=model_id,
        tenant_id=tenant_id,
        markup=deps.markup,
        pricing=pricing,
    )


async def _executing_model_id(
    deps: TurnDeps,
    *,
    existing: ThreadSessionRow,
    observed: BetaManagedAgentsSession | None,
    fallback_model_id: str,
) -> str:
    """The model a REUSED session is actually running, backfilling if needed.

    The recorded snapshot answers this without any MA call. A row written
    before snapshots existed has none, so the session is read once and the
    snapshot backfilled onto the row — `check_session_agent` hands over the
    response when its own legacy `ma_agent_id` branch already fetched it, so a
    legacy row costs at most one `sessions.retrieve` per bind.

    A 404 on that read means the session is gone; the turn's existing
    dead-session recovery will create a replacement and rebind the recorder to
    it, so this returns the caller's fallback rather than failing the bind.
    """
    if existing.effective_config is not None:
        return existing.effective_config.model_id

    if observed is None:
        try:
            observed = await deps.anthropic.beta.sessions.retrieve(existing.ma_session_id)
        except APIStatusError as error:
            if error.status_code == 404:
                return fallback_model_id
            raise

    snapshot = snapshot_from_retrieved_session(observed)
    async with deps.sessionmaker() as session, session.begin():
        await record_snapshot(
            session,
            id=existing.id,
            snapshot=snapshot,
            identity_fingerprint=fingerprint_identity(snapshot),
            mutable_fingerprint=fingerprint_mutable(snapshot),
        )
    return snapshot.model_id


async def bind_session(
    deps: TurnDeps,
    admission: Admission,
    *,
    tenant_id: uuid.UUID,
    platform: str,
    external_user_id: str,
    thread_id: str,
    session_account_id: uuid.UUID,
    reuse_existing: bool,
    deadline: dt.datetime | None = None,
    now: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.UTC),
) -> PreparedTurn:
    """Find-or-create the MA session for this turn and bind its recorder.

    When `reuse_existing` is True, a live `thread_sessions` row for
    (tenant_id, platform, thread_id, session_account_id) is reused verbatim
    (no `create_session` call). Otherwise (no live row, or
    `reuse_existing=False` for Discord's channel-mention path) a fresh
    session is created via the single shared `create_session` call site,
    always passing `fernet=deps.fernet`, and a new `thread_sessions` mapping
    row is written.

    Billing binds to the model the bound session actually runs: the fresh
    session's own snapshot, or the reused row's recorded one (backfilled from
    MA for a row written before snapshots existed). The responder agent's
    current model is only a fallback for a session we cannot read.

    `deadline`/`now` bound this whole body against the per-turn ceiling
    (D-03): the MA `sessions.create` call and the mapping write, and on the
    reuse path `sync_agent_mcp_credentials`. This does NOT cover `admit()`,
    which runs before `bind_session` and is deliberately unbounded (D-04).
    `deadline=None` is fail-safe, not off -- it computes
    `turn_deadline(now=now())` so every caller (including one that never
    passes a deadline) is still ceiling-covered.

    Raises `TypeError` if `admission` is not a real `Admission` -- pyright's
    strict mode already rejects a mistyped caller at type-check time; this
    guard makes the same contract hold at runtime (the type-level chokepoint
    claim tested by 06-05's `test_bind_session_requires_an_admission_value`).
    This check runs BEFORE the ceiling wrap, so a mistyped caller still fails
    immediately with the same error rather than waiting out a timeout.
    """
    if not isinstance(admission, Admission):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise TypeError(
            f"bind_session requires an Admission instance, got {type(admission).__name__}"
        )

    effective_deadline = deadline if deadline is not None else turn_deadline(now=now())

    async def _bind() -> PreparedTurn:
        ma_session_id: str | None = None
        mapping_id: uuid.UUID | None = None
        watermark: str | None = None
        reused = False
        # Only ever the billed model for a session we cannot read: a reused
        # session resolves it from its snapshot, a fresh one from what MA
        # actually froze.
        model_id = admission.agent.model.id

        if reuse_existing:
            async with deps.sessionmaker() as session:
                existing = await get_live_thread_session(
                    session,
                    tenant_id=tenant_id,
                    platform=platform,
                    thread_id=thread_id,
                    account_id=session_account_id,
                )
            if existing is not None:
                identity = await check_session_agent(
                    deps.anthropic,
                    deps.sessionmaker,
                    mapping=existing,
                    responder_ma_agent_id=admission.agent.id,
                )
                session_exists = identity.session_exists
                ma_session_id = existing.ma_session_id
                mapping_id = existing.id
                watermark = existing.watermark_message_id
                reused = True
                if session_exists:
                    model_id = await _executing_model_id(
                        deps,
                        existing=existing,
                        observed=identity.observed,
                        fallback_model_id=model_id,
                    )
                # A reused session skips create_session, so it would never pick
                # up an external MCP credential added to the agent after it was
                # created — the caller would keep failing at MCP init until
                # their session happened to be recreated. The vault this
                # session already mounts is read at each turn's MCP init, so
                # writing into it here reaches this session on this turn.
                if (
                    session_exists
                    and deps.fernet is not None
                    and deps.mcp.public_url is not None
                    and deps.mcp.jwt_secret is not None
                ):
                    await sync_agent_mcp_credentials(
                        deps.anthropic,
                        sessionmaker=deps.sessionmaker,
                        fernet=deps.fernet,
                        tenant_id=tenant_id,
                        agent_id=derive_agent_uuid(
                            tenant_id=tenant_id, ma_agent_id=str(admission.agent.id)
                        ),
                        account_id=admission.account_id,
                        jwt_secret=deps.mcp.jwt_secret.get_secret_value().encode(),
                        public_url=str(deps.mcp.public_url),
                        now=dt.datetime.now(dt.UTC),
                    )

        if not reused:
            fresh = await create_fresh_session(
                deps,
                admission,
                tenant_id=tenant_id,
                platform=platform,
                thread_id=thread_id,
                session_account_id=session_account_id,
            )
            ma_session_id = fresh.ma_session_id
            mapping_id = fresh.mapping_id
            model_id = fresh.snapshot.model_id
            watermark = None

        assert ma_session_id is not None, "ma_session_id must be resolved on every code path"

        record = bind_recorder(
            deps,
            tenant_id=tenant_id,
            external_user_id=external_user_id,
            ma_session_id=ma_session_id,
            model_id=model_id,
        )

        return PreparedTurn(
            admission=admission,
            ma_session_id=ma_session_id,
            mapping_id=mapping_id,
            watermark=watermark,
            reused=reused,
            session_account_id=session_account_id,
            _record=record,
        )

    try:
        return await asyncio.wait_for(_bind(), timeout=remaining_s(effective_deadline, now=now()))
    except TimeoutError as err:
        log.error(
            "turn.ceiling_exceeded",
            phase="bind_session",
            tenant_id=str(tenant_id),
            platform=platform,
            thread_id=thread_id,
            deadline=effective_deadline.isoformat(),
        )
        raise ceiling_error() from err
