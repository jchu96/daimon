"""Tests for `daimon.core.workspace_transfer` — the replacement ladder.

Every MA call runs transport-level through the stateful sessions fake and a
real `AsyncAnthropic`, so the SDK's own parameter validation and response
parsing run in full: the checkpoint turn is really driven by
`turn.driver.run_turn` over a real SSE stream, and the bundle really moves
through `files.download` / `files.delete` / `files.upload`.

The pending-delete enqueue is the one DB write on this path, so these tests
take the real-Postgres session factory.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

import httpx
from anthropic import AsyncAnthropic
from anthropic.types.beta.sessions.beta_managed_agents_agent_message_event import (
    BetaManagedAgentsAgentMessageEvent,
)
from anthropic.types.beta.sessions.beta_managed_agents_session_end_turn import (
    BetaManagedAgentsSessionEndTurn,
)
from anthropic.types.beta.sessions.beta_managed_agents_session_status_idle_event import (
    BetaManagedAgentsSessionStatusIdleEvent,
)
from anthropic.types.beta.sessions.beta_managed_agents_span_model_request_end_event import (
    BetaManagedAgentsSpanModelRequestEndEvent,
)
from anthropic.types.beta.sessions.beta_managed_agents_span_model_usage import (
    BetaManagedAgentsSpanModelUsage,
)
from anthropic.types.beta.sessions.beta_managed_agents_text_block import (
    BetaManagedAgentsTextBlock,
)
from anthropic.types.beta.sessions.beta_managed_agents_user_message_event import (
    BetaManagedAgentsUserMessageEvent,
)
from daimon.core.checkpoint_prompt import CHECKPOINT_EXCLUDED_PATHS, handoff_filename
from daimon.core.session_snapshot import SessionSnapshot
from daimon.core.stores.pending_file_deletes import list_due_pending_file_deletes
from daimon.core.workspace_transfer import (
    HANDOFF_MOUNT_PATH,
    FullHandoff,
    HistoryOnly,
    TranscriptOnly,
    as_prepared_replacement,
    transfer_workspace,
)
from daimon.testing.factories import make_tenant
from daimon.testing.ma import (
    FakeMAState,
    build_fake_anthropic,
    combine_handlers,
    make_fake_ma_handler,
    make_fake_memory_store_handler,
)
from daimon.testing.ma_sessions import (
    FakeSessionsState,
    make_fake_sessions_handler,
    session_turn_sse,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

NOW = datetime(2026, 9, 13, 9, 0, 0, tzinfo=UTC)
TRANSFER_ID = UUID("0f9d1c4e-7c2b-4f5a-9d3e-1b2c3d4e5f60")
TENANT_ID = UUID("11111111-2222-3333-4444-555555555555")
DEADLINE = NOW + timedelta(seconds=180)
TARBALL = b"\x1f\x8b\x08fake gzip bytes for the handoff bundle"


def _now() -> datetime:
    return NOW


async def _no_sleep(delay: float) -> None:
    return None


def _client(state: FakeSessionsState) -> AsyncAnthropic:
    """Compose the sessions fake with the memory-store and agent fakes.

    The agent fake MUST be last: it never raises `NotHandled` (its final
    branch is its own 404), so anything after it is unreachable.
    """
    return build_fake_anthropic(
        combine_handlers(
            make_fake_sessions_handler(state),
            make_fake_memory_store_handler(),
            make_fake_ma_handler(state.ma),
        )
    )


async def _make_session(
    client: AsyncAnthropic, *, model: str = "claude-sonnet-5", name: str = "analysis-bot"
) -> str:
    agent = await client.beta.agents.create(name=name, model=model)
    session = await client.beta.sessions.create(agent=agent.id, environment_id="env_test")
    return session.id


def _snapshot(*, model_id: str = "claude-sonnet-5", repo_mount_path: str | None = None):
    """The recorded configuration of the session being replaced.

    Only `model_id` (what the checkpoint turn is billed at) and
    `repo_mount_path` (whether the prompt asks for git state) are read by the
    transfer; the rest is the identity axis the compatibility check owns.
    """
    return SessionSnapshot(
        ma_agent_id="agnt_old",
        model_id=model_id,
        system_sha256=None,
        skills_sha256="0" * 64,
        environment_id="env_test",
        repo_url=None,
        repo_branch=None,
        memory_store_id=None,
        vault_id=None,
        tools_sha256="1" * 64,
        mcp_servers_sha256="2" * 64,
        env_sha256=None,
        repo_mount_path=repo_mount_path,
        agent_version=1,
        agent_name="analysis-bot",
    )


def _seed_conversation(state: FakeSessionsState, session_id: str, *, with_reply: bool) -> None:
    """Put a real prior conversation in the session's event log.

    `with_reply` is the `is_worth_checkpointing` gate: a session that never
    produced an `agent.message` has nothing to hand over.
    """
    events: list[dict[str, Any]] = [
        BetaManagedAgentsUserMessageEvent(
            id="sevt_user_1",
            type="user.message",
            content=[BetaManagedAgentsTextBlock(type="text", text="fit the hierarchical model")],
            processed_at=None,
        ).model_dump(mode="json")
    ]
    if with_reply:
        events.append(
            BetaManagedAgentsAgentMessageEvent(
                id="sevt_agent_1",
                type="agent.message",
                content=[
                    BetaManagedAgentsTextBlock(type="text", text="Fitted it; trace is in trace.nc")
                ],
                processed_at=NOW,
            ).model_dump(mode="json")
        )
    state.events[session_id] = events


def _script_checkpoint_reply(state: FakeSessionsState, session_id: str, reply: str) -> None:
    """Queue the checkpoint turn's SSE stream: one reply, then a terminal idle."""
    state.stream_scripts[session_id] = [
        session_turn_sse(
            BetaManagedAgentsAgentMessageEvent(
                id="sevt_checkpoint",
                type="agent.message",
                content=[BetaManagedAgentsTextBlock(type="text", text=reply)],
                processed_at=NOW,
            ),
            BetaManagedAgentsSessionStatusIdleEvent(
                id="sevt_idle",
                type="session.status_idle",
                stop_reason=BetaManagedAgentsSessionEndTurn(type="end_turn"),
                processed_at=NOW,
            ),
        )
    ]


# ---------------------------------------------------------------------------
# Rung 1: the full handoff
# ---------------------------------------------------------------------------


async def test_transfer_rehosts_the_bundle_and_enqueues_its_deletion(
    db_session: AsyncSession, db_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """The happy path end to end: the checkpoint turn runs, its archive is
    found in the outputs listing, and the bytes are re-hosted as a standalone
    upload that the successor can mount. The session-scoped copy is dropped
    from the listing (it dies with the session anyway) and the new object is
    queued for deletion rather than left to accumulate."""
    state = FakeSessionsState(ma=FakeMAState())
    client = _client(state)
    old_session = await _make_session(client)
    _seed_conversation(state, old_session, with_reply=True)
    _script_checkpoint_reply(state, old_session, "-rw-r--r-- 1 root root 42 handoff.tar.gz")
    output = state.write_output(old_session, handoff_filename(TRANSFER_ID), TARBALL)

    outcome = await transfer_workspace(
        client,
        db_session_factory,
        old_session_id=old_session,
        old_snapshot=_snapshot(),
        tenant_id=TENANT_ID,
        external_user_id="U123",
        transfer_id=TRANSFER_ID,
        markup=Decimal("1.0"),
        checkpoint_deadline=DEADLINE,
        from_agent_name="analysis-bot",
        sleep=_no_sleep,
        now=_now,
    )

    assert isinstance(outcome, FullHandoff), f"expected a full handoff, got {outcome!r}"
    assert outcome.mount_path == HANDOFF_MOUNT_PATH
    assert outcome.bytes_transferred == len(TARBALL)
    assert outcome.unpreserved == (), "nothing was lost on the happy path"
    assert outcome.transcript is not None and "hierarchical model" in outcome.transcript

    stored_meta, stored_bytes = state.files[outcome.transfer_file_id]
    assert stored_bytes == TARBALL, "the uploaded bytes must equal the downloaded bytes"
    assert stored_meta.filename == handoff_filename(TRANSFER_ID)
    assert output.id not in state.files, "the session-scoped output must be dropped"

    listing = await client.beta.files.list(scope_id=old_session, limit=1000)
    assert listing.data == [], "the retired session's outputs listing is left clean"

    due = await list_due_pending_file_deletes(db_session, now=NOW + timedelta(days=8))
    assert [row.file_id for row in due] == [outcome.transfer_file_id], (
        "the re-hosted bundle must be queued for deletion, not left forever"
    )


async def test_transfer_records_a_repository_commit_it_could_not_prevent(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The checkpoint prompt forbids committing and echoes HEAD on both sides
    of the archive step. Two different hashes mean the session committed
    anyway — daimon detects that and tells the successor, because a silent
    commit is worse than a named one."""
    state = FakeSessionsState(ma=FakeMAState())
    client = _client(state)
    old_session = await _make_session(client)
    _seed_conversation(state, old_session, with_reply=True)
    _script_checkpoint_reply(
        state,
        old_session,
        "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678\n"
        "-rw-r--r-- 1 root root 42 handoff.tar.gz\n"
        "9876543210fedcba9876543210fedcba98765432\n",
    )
    state.write_output(old_session, handoff_filename(TRANSFER_ID), TARBALL)

    outcome = await transfer_workspace(
        client,
        db_session_factory,
        old_session_id=old_session,
        old_snapshot=_snapshot(repo_mount_path="/root/repo"),
        tenant_id=TENANT_ID,
        external_user_id="U123",
        transfer_id=TRANSFER_ID,
        markup=Decimal("1.0"),
        checkpoint_deadline=DEADLINE,
        from_agent_name="analysis-bot",
        sleep=_no_sleep,
        now=_now,
    )

    assert isinstance(outcome, FullHandoff)
    assert outcome.unpreserved == ("the agent committed to the repository during checkpoint",)


async def test_checkpoint_prompt_names_every_excluded_path(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The exclusions are the caller-isolation boundary: credential mounts, the agent's
    memory store and the platform skills belong to the destination, never to
    the task. Daimon cannot inspect the archive the session builds, so the
    prompt naming every excluded root IS the contract — asserted on the bytes
    actually sent to the old session."""
    state = FakeSessionsState(ma=FakeMAState())
    client = _client(state)
    old_session = await _make_session(client)
    _seed_conversation(state, old_session, with_reply=True)
    _script_checkpoint_reply(state, old_session, "ok")
    state.write_output(old_session, handoff_filename(TRANSFER_ID), TARBALL)

    await transfer_workspace(
        client,
        db_session_factory,
        old_session_id=old_session,
        old_snapshot=_snapshot(repo_mount_path="/root/repo"),
        tenant_id=TENANT_ID,
        external_user_id="U123",
        transfer_id=TRANSFER_ID,
        markup=Decimal("1.0"),
        checkpoint_deadline=DEADLINE,
        from_agent_name="analysis-bot",
        sleep=_no_sleep,
        now=_now,
    )

    sent_session, batch = state.sent_batches[0]
    assert sent_session == old_session
    assert [event["type"] for event in batch] == ["user.message"], (
        "the checkpoint turn is a plain user message — no privileged framing"
    )
    prompt = "".join(block["text"] for block in batch[0]["content"] if block["type"] == "text")
    for excluded in CHECKPOINT_EXCLUDED_PATHS:
        assert excluded.strip("/") in prompt, f"{excluded} must be excluded from the archive"
    assert "NEVER RUN GIT COMMIT" in prompt, "the prompt must forbid changing the repository"


async def test_transfer_bills_the_checkpoint_at_the_old_model_as_a_checkpoint_debit(
    db_session: AsyncSession, db_session_factory: async_sessionmaker[AsyncSession]
) -> None:
    """The checkpoint is real model work and the tenant pays for it — at the
    model the OLD session froze, not whatever the agent points at now, and
    under its own ledger reason so it is separable from turns a person asked
    for."""
    from daimon.core._models import TenantLedger, UsageEvent

    tenant = await make_tenant(db_session, workspace_id="guild-checkpoint")
    await db_session.commit()

    state = FakeSessionsState(ma=FakeMAState())
    client = _client(state)
    old_session = await _make_session(client, model="claude-haiku-4-5")
    _seed_conversation(state, old_session, with_reply=True)
    state.stream_scripts[old_session] = [
        session_turn_sse(
            BetaManagedAgentsSpanModelRequestEndEvent(
                id="sevt_span",
                type="span.model_request_end",
                model_request_start_id="sevt_span_start",
                model_usage=BetaManagedAgentsSpanModelUsage(
                    input_tokens=1000,
                    output_tokens=200,
                    cache_creation_input_tokens=0,
                    cache_read_input_tokens=0,
                ),
                processed_at=NOW,
            ),
            BetaManagedAgentsSessionStatusIdleEvent(
                id="sevt_idle",
                type="session.status_idle",
                stop_reason=BetaManagedAgentsSessionEndTurn(type="end_turn"),
                processed_at=NOW,
            ),
        )
    ]
    state.write_output(old_session, handoff_filename(TRANSFER_ID), TARBALL)

    outcome = await transfer_workspace(
        client,
        db_session_factory,
        old_session_id=old_session,
        old_snapshot=_snapshot(model_id="claude-haiku-4-5"),
        tenant_id=tenant.id,
        external_user_id="U123",
        transfer_id=TRANSFER_ID,
        markup=Decimal("1.0"),
        checkpoint_deadline=DEADLINE,
        from_agent_name="analysis-bot",
        sleep=_no_sleep,
        now=_now,
    )

    assert isinstance(outcome, FullHandoff)
    usage = (await db_session.execute(select(UsageEvent))).scalars().all()
    assert [(row.managed_session_id, row.model) for row in usage] == [
        (old_session, "claude-haiku-4-5")
    ], "usage is metered against the session that ran, at the model it froze"
    ledger = (await db_session.execute(select(TenantLedger))).scalars().all()
    assert [row.reason for row in ledger] == ["checkpoint_debit"]
    assert ledger[0].delta_usd < 0, "a checkpoint is a debit"


# ---------------------------------------------------------------------------
# Rung 2: transcript only
# ---------------------------------------------------------------------------


async def test_transfer_skips_the_billed_turn_when_the_session_never_answered(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A session with no `agent.message` produced nothing worth paying to
    save, and its one user message is already being replayed into the
    successor. No turn is spent at all."""
    state = FakeSessionsState(ma=FakeMAState())
    client = _client(state)
    old_session = await _make_session(client)
    _seed_conversation(state, old_session, with_reply=False)

    outcome = await transfer_workspace(
        client,
        db_session_factory,
        old_session_id=old_session,
        old_snapshot=_snapshot(),
        tenant_id=TENANT_ID,
        external_user_id="U123",
        transfer_id=TRANSFER_ID,
        markup=Decimal("1.0"),
        checkpoint_deadline=DEADLINE,
        from_agent_name="analysis-bot",
        sleep=_no_sleep,
        now=_now,
    )

    assert isinstance(outcome, TranscriptOnly), f"expected the transcript rung, got {outcome!r}"
    assert outcome.gap_reason == "not_worth_checkpointing"
    assert "hierarchical model" in outcome.transcript, "the user's own words still cross"
    assert state.sent_batches == [], "no events were sent, so no turn was billed"


async def test_transfer_degrades_to_transcript_when_the_old_session_is_archived(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """An archived session still serves its event log but refuses new events
    with a 400. The conversation crosses; the files cannot."""
    state = FakeSessionsState(ma=FakeMAState())
    client = _client(state)
    old_session = await _make_session(client)
    _seed_conversation(state, old_session, with_reply=True)
    await client.beta.sessions.archive(old_session)

    outcome = await transfer_workspace(
        client,
        db_session_factory,
        old_session_id=old_session,
        old_snapshot=_snapshot(),
        tenant_id=TENANT_ID,
        external_user_id="U123",
        transfer_id=TRANSFER_ID,
        markup=Decimal("1.0"),
        checkpoint_deadline=DEADLINE,
        from_agent_name="analysis-bot",
        sleep=_no_sleep,
        now=_now,
    )

    assert isinstance(outcome, TranscriptOnly)
    assert outcome.gap_reason == "session_dead"
    assert "hierarchical model" in outcome.transcript


async def test_transfer_degrades_to_transcript_when_the_session_vanishes_mid_transfer(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The 404 limb of a dead session: the log is read, then the session is
    deleted before the checkpoint turn's first event lands. Reading the log
    and sending to it are two calls, and anything can happen in between."""
    state = FakeSessionsState(ma=FakeMAState())
    base = make_fake_sessions_handler(state)
    old_session_holder: list[str] = []

    def delete_after_replay(request: httpx.Request) -> httpx.Response:
        response = base(request)
        if request.method == "GET" and re.fullmatch(r"/v1/sessions/[^/]+/events", request.url.path):
            state.deleted_sessions.update(old_session_holder)
        return response

    client = build_fake_anthropic(
        combine_handlers(
            delete_after_replay,
            make_fake_memory_store_handler(),
            make_fake_ma_handler(state.ma),
        )
    )
    old_session = await _make_session(client)
    old_session_holder.append(old_session)
    _seed_conversation(state, old_session, with_reply=True)

    outcome = await transfer_workspace(
        client,
        db_session_factory,
        old_session_id=old_session,
        old_snapshot=_snapshot(),
        tenant_id=TENANT_ID,
        external_user_id="U123",
        transfer_id=TRANSFER_ID,
        markup=Decimal("1.0"),
        checkpoint_deadline=DEADLINE,
        from_agent_name="analysis-bot",
        sleep=_no_sleep,
        now=_now,
    )

    assert isinstance(outcome, TranscriptOnly)
    assert outcome.gap_reason == "session_dead"


async def test_transfer_degrades_to_transcript_when_the_bundle_is_oversize(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A bundle bigger than the cap is never downloaded: the transfer would
    have to stream it through this process and back out again, and the cap
    exists because that is the same path a posted output takes."""
    state = FakeSessionsState(ma=FakeMAState())
    client = _client(state)
    old_session = await _make_session(client)
    _seed_conversation(state, old_session, with_reply=True)
    _script_checkpoint_reply(state, old_session, "ok")
    output = state.write_output(old_session, handoff_filename(TRANSFER_ID), TARBALL)

    outcome = await transfer_workspace(
        client,
        db_session_factory,
        old_session_id=old_session,
        old_snapshot=_snapshot(),
        tenant_id=TENANT_ID,
        external_user_id="U123",
        transfer_id=TRANSFER_ID,
        markup=Decimal("1.0"),
        checkpoint_deadline=DEADLINE,
        from_agent_name="analysis-bot",
        max_bundle_bytes=len(TARBALL) - 1,
        sleep=_no_sleep,
        now=_now,
    )

    assert isinstance(outcome, TranscriptOnly)
    assert outcome.gap_reason == "bundle_oversize"
    assert output.id in state.files, "an oversize bundle is left alone, not deleted"


async def test_transfer_degrades_to_transcript_when_the_checkpoint_wrote_no_archive(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The turn can end successfully and still produce nothing — the session
    talked instead of tarring. A reply is not evidence; the outputs listing
    is."""
    state = FakeSessionsState(ma=FakeMAState())
    client = _client(state)
    old_session = await _make_session(client)
    _seed_conversation(state, old_session, with_reply=True)
    _script_checkpoint_reply(state, old_session, "I would rather not archive anything.")
    state.write_output(old_session, "unrelated-notes.md", b"not a bundle")

    outcome = await transfer_workspace(
        client,
        db_session_factory,
        old_session_id=old_session,
        old_snapshot=_snapshot(),
        tenant_id=TENANT_ID,
        external_user_id="U123",
        transfer_id=TRANSFER_ID,
        markup=Decimal("1.0"),
        checkpoint_deadline=DEADLINE,
        from_agent_name="analysis-bot",
        sleep=_no_sleep,
        now=_now,
    )

    assert isinstance(outcome, TranscriptOnly)
    assert outcome.gap_reason == "checkpoint_failed"


async def test_transfer_degrades_to_transcript_when_the_re_upload_fails(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """The bundle is only useful once it is standalone. An upload that fails
    leaves the transfer with bytes it cannot mount, which is the transcript
    rung — not a crash, and not a silent full handoff."""
    state = FakeSessionsState(ma=FakeMAState())
    base = make_fake_sessions_handler(state)

    def refuse_upload(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path == "/v1/files":
            return httpx.Response(
                400,
                json={
                    "type": "error",
                    "error": {"type": "invalid_request_error", "message": "storage full"},
                },
            )
        return base(request)

    client = build_fake_anthropic(
        combine_handlers(
            refuse_upload,
            make_fake_memory_store_handler(),
            make_fake_ma_handler(state.ma),
        )
    )
    old_session = await _make_session(client)
    _seed_conversation(state, old_session, with_reply=True)
    _script_checkpoint_reply(state, old_session, "ok")
    state.write_output(old_session, handoff_filename(TRANSFER_ID), TARBALL)

    outcome = await transfer_workspace(
        client,
        db_session_factory,
        old_session_id=old_session,
        old_snapshot=_snapshot(),
        tenant_id=TENANT_ID,
        external_user_id="U123",
        transfer_id=TRANSFER_ID,
        markup=Decimal("1.0"),
        checkpoint_deadline=DEADLINE,
        from_agent_name="analysis-bot",
        sleep=_no_sleep,
        now=_now,
    )

    assert isinstance(outcome, TranscriptOnly)
    assert outcome.gap_reason == "upload_failed"


# ---------------------------------------------------------------------------
# Rung 3: history only
# ---------------------------------------------------------------------------


async def test_transfer_returns_history_only_when_the_session_log_is_gone(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A deleted session 404s on `events.list`, so there is no conversation to
    quote and no session to checkpoint. The platform thread is all the
    successor gets, and the copy must say so."""
    state = FakeSessionsState(ma=FakeMAState())
    client = _client(state)
    old_session = await _make_session(client)
    _seed_conversation(state, old_session, with_reply=True)
    await client.beta.sessions.delete(old_session)

    outcome = await transfer_workspace(
        client,
        db_session_factory,
        old_session_id=old_session,
        old_snapshot=_snapshot(),
        tenant_id=TENANT_ID,
        external_user_id="U123",
        transfer_id=TRANSFER_ID,
        markup=Decimal("1.0"),
        checkpoint_deadline=DEADLINE,
        from_agent_name="analysis-bot",
        sleep=_no_sleep,
        now=_now,
    )

    assert outcome == HistoryOnly(gap_reason="events_unavailable")
    assert state.sent_batches == [], "a gone session is never billed"


# ---------------------------------------------------------------------------
# as_prepared_replacement
# ---------------------------------------------------------------------------


def test_as_prepared_replacement_mounts_the_bundle_and_keeps_the_transcript_out_of_system() -> None:
    """On a model that accepts `system.message`, daimon's own framing takes
    the privileged channel and the quoted previous conversation does not: the
    transcript is other people's words, and a handoff is exactly when they
    might be hostile."""
    outcome = FullHandoff(
        transfer_file_id="file_bundle",
        mount_path=HANDOFF_MOUNT_PATH,
        bytes_transferred=4096,
        transcript='<previous_session from="analysis-bot" trust="untrusted">\n'
        '<turn role="user">fit the hierarchical model</turn>\n</previous_session>',
        unpreserved=(),
    )

    prepared = as_prepared_replacement(
        outcome,
        destination_model_id="claude-sonnet-5",
        from_agent_name="analysis-bot",
        to_agent_name="report-bot",
        requested_work="write up the results",
    )

    assert prepared.transfer_kind == "full"
    assert prepared.transfer_file_id == "file_bundle"
    assert prepared.extra_resources == (
        {"type": "file", "file_id": "file_bundle", "mount_path": HANDOFF_MOUNT_PATH},
    )
    assert len(prepared.system_blocks) == 1
    system_text = prepared.system_blocks[0]["text"]
    assert HANDOFF_MOUNT_PATH in system_text, "the framing must say where the archive is"
    assert "hierarchical model" not in system_text, (
        "quoted conversation must never ride the privileged channel"
    )
    assert "hierarchical model" in prepared.user_prefix
    assert "write up the results" in system_text


def test_as_prepared_replacement_puts_everything_in_the_user_prefix_on_an_ungated_model() -> None:
    """haiku rejects a `system.message` outright — the whole request 400s — so
    the same words go in the user message instead, with the transcript after
    them."""
    outcome = FullHandoff(
        transfer_file_id="file_bundle",
        mount_path=HANDOFF_MOUNT_PATH,
        bytes_transferred=4096,
        transcript='<previous_session from="analysis-bot" trust="untrusted">\n'
        '<turn role="user">fit the hierarchical model</turn>\n</previous_session>',
        unpreserved=(),
    )

    prepared = as_prepared_replacement(
        outcome,
        destination_model_id="claude-haiku-4-5",
        from_agent_name="analysis-bot",
        to_agent_name="report-bot",
        requested_work="write up the results",
    )

    assert prepared.system_blocks == (), "no privileged channel exists on this model"
    assert HANDOFF_MOUNT_PATH in prepared.user_prefix
    assert "hierarchical model" in prepared.user_prefix
    assert prepared.user_prefix.index(HANDOFF_MOUNT_PATH) < prepared.user_prefix.index(
        "<previous_session"
    ), "daimon's framing comes before the quoted block it is framing"


def test_as_prepared_replacement_explains_the_gap_without_naming_a_reason_code() -> None:
    """`transfer_kind` and the not-carried line are what stop the copy
    overstating what survived. A reason code in user-facing text would be a
    leak, so the phrasing is a sentence, not an identifier."""
    prepared = as_prepared_replacement(
        TranscriptOnly(transcript="", gap_reason="checkpoint_timeout"),
        destination_model_id="claude-sonnet-5",
        from_agent_name="analysis-bot",
        to_agent_name="analysis-bot",
        requested_work=None,
    )

    assert prepared.transfer_kind == "transcript"
    assert prepared.extra_resources == (), "nothing to mount"
    assert prepared.transfer_file_id is None
    assert prepared.user_prefix == "", "no transcript means no user prefix"
    system_text = prepared.system_blocks[0]["text"]
    assert "checkpoint_timeout" not in system_text
    assert "ran out of time" in system_text
    assert "running processes, notebook kernels and shells" in system_text


def test_as_prepared_replacement_says_only_the_thread_survived_for_history_only() -> None:
    prepared = as_prepared_replacement(
        HistoryOnly(gap_reason="events_unavailable"),
        destination_model_id="claude-sonnet-5",
        from_agent_name="analysis-bot",
        to_agent_name="report-bot",
        requested_work=None,
    )

    assert prepared.transfer_kind == "history"
    assert prepared.user_prefix == ""
    system_text = prepared.system_blocks[0]["text"]
    assert "only what was posted in this thread came across" in system_text
