"""The continuation dispatch contract: queue once, claim once, and know when not to run.

Real Postgres throughout — the at-most-once guarantee is a row lock, not a
Python flag — and a transport-level fake MA, because re-resolving the concrete
destination is one of the decisions under test.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from anthropic import AsyncAnthropic
from daimon.core.continuity.continuation import (
    ContinuationRequest,
    claim_continuation,
    decide_continuation,
    record_continuation,
    settle_continuation,
)
from daimon.core.errors import DaimonError
from daimon.core.stores.task_continuations import get_continuation
from daimon.testing.factories import make_account, make_tenant
from daimon.testing.ma import MARouter, build_fake_anthropic, not_found_response
from daimon.testing.ma_models import ma_agent
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

_NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)
_TARGET_ID = "agt_stats"


@pytest_asyncio.fixture
async def caller(db_session: AsyncSession) -> tuple[uuid.UUID, uuid.UUID]:
    """A tenant and one account inside it, committed and ready to be referenced."""
    tenant = await make_tenant(db_session)
    account = await make_account(db_session, tenant=tenant)
    await db_session.commit()
    return tenant.id, account.id


def _request(
    *,
    tenant_id: uuid.UUID,
    account_id: uuid.UUID,
    requested_work: str | None = "finish the churn writeup",
    target_ma_agent_id: str = _TARGET_ID,
) -> ContinuationRequest:
    return ContinuationRequest(
        tenant_id=tenant_id,
        platform="discord",
        parent_channel_id="C_PARENT",
        thread_id="T_THREAD",
        requester_account_id=account_id,
        requester_external_user_id="discord-user-1",
        target_ma_agent_id=target_ma_agent_id,
        target_name="stats-bot",
        requested_work=requested_work,
        reason="task_handoff",
        idempotency_key=uuid.uuid4(),
    )


def _live_target(tenant_id: uuid.UUID) -> AsyncAnthropic:
    agent = ma_agent(
        id=_TARGET_ID,
        name="stats-bot",
        model="claude-sonnet-5",
        tenant_id=tenant_id,
        created_at=_NOW,
    )
    router = MARouter()
    router.add_agent(agent)
    return build_fake_anthropic(router.dispatch)


def _missing_target() -> AsyncAnthropic:
    router = MARouter()
    router.add(
        "GET", rf"/v1/agents/{_TARGET_ID}", lambda _r, _m: not_found_response("agent not found")
    )
    return build_fake_anthropic(router.dispatch)


def _archived_target(tenant_id: uuid.UUID) -> AsyncAnthropic:
    archived = ma_agent(
        id=_TARGET_ID,
        name="stats-bot",
        model="claude-sonnet-5",
        tenant_id=tenant_id,
        created_at=_NOW,
        archived_at=_NOW,
    )
    router = MARouter()
    router.add_agent(archived)
    return build_fake_anthropic(router.dispatch)


async def test_decide_skips_save_only_when_the_handoff_carried_no_work(
    db_session_factory: async_sessionmaker[AsyncSession],
    caller: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """A switch alone promised nothing, so it must never spend a billed turn."""
    tenant_id, account_id = caller
    request = _request(tenant_id=tenant_id, account_id=account_id, requested_work=None)
    await record_continuation(db_session_factory, request)

    decision = await decide_continuation(
        db_session_factory,
        _live_target(tenant_id),
        request=request,
        now=_NOW,
        latest_user_message_at=None,
        active_turn=False,
    )

    assert decision.action == "skip_save_only", "no requested work means no dispatch"
    assert decision.message is None, "a silent skip says nothing to the person"
    assert decision.seed_user_message is None, "nothing is seeded when nothing runs"


async def test_decide_dispatches_with_the_requested_work_as_the_seed_message(
    db_session_factory: async_sessionmaker[AsyncSession],
    caller: tuple[uuid.UUID, uuid.UUID],
) -> None:
    tenant_id, account_id = caller
    request = _request(tenant_id=tenant_id, account_id=account_id)
    await record_continuation(db_session_factory, request)

    decision = await decide_continuation(
        db_session_factory,
        _live_target(tenant_id),
        request=request,
        now=_NOW,
        latest_user_message_at=None,
        active_turn=False,
    )

    assert decision.action == "dispatch", "a live target with work queued runs the turn"
    assert decision.seed_user_message == "finish the churn writeup", (
        "the person's own words are what the new agent is asked to continue"
    )
    assert decision.message is None, "a dispatch needs no skip copy"


@pytest.mark.parametrize("target_state", ["missing", "archived", "other_tenant"])
async def test_decide_skips_when_the_target_is_no_longer_that_agent(
    db_session_factory: async_sessionmaker[AsyncSession],
    caller: tuple[uuid.UUID, uuid.UUID],
    target_state: str,
) -> None:
    """Never substitute a namesake: a changed destination is a skip, not a guess."""
    tenant_id, account_id = caller
    request = _request(tenant_id=tenant_id, account_id=account_id)
    await record_continuation(db_session_factory, request)
    if target_state == "missing":
        anthropic = _missing_target()
    elif target_state == "archived":
        anthropic = _archived_target(tenant_id)
    else:
        anthropic = _live_target(uuid.uuid4())

    decision = await decide_continuation(
        db_session_factory,
        anthropic,
        request=request,
        now=_NOW,
        latest_user_message_at=None,
        active_turn=False,
    )

    assert decision.action == "skip_target_changed", (
        f"a {target_state} target must not receive a queued turn"
    )
    assert decision.message is not None, "the person is told the target changed"
    assert "stats-bot is no longer the agent this was set up for." in decision.message, (
        "the copy names the agent the person chose"
    )
    assert "Nothing was lost." in decision.message, "the copy must not imply the work was destroyed"


async def test_decide_skips_when_the_person_has_spoken_again_since_queueing(
    db_session_factory: async_sessionmaker[AsyncSession],
    caller: tuple[uuid.UUID, uuid.UUID],
) -> None:
    tenant_id, account_id = caller
    request = _request(tenant_id=tenant_id, account_id=account_id)
    await record_continuation(db_session_factory, request)
    async with db_session_factory() as session:
        row = await get_continuation(session, idempotency_key=request.idempotency_key)
    assert row is not None, "the recorded row is the clock the supersede rule reads"

    decision = await decide_continuation(
        db_session_factory,
        _live_target(tenant_id),
        request=request,
        now=_NOW,
        latest_user_message_at=row.created_at + timedelta(seconds=1),
        active_turn=False,
    )

    assert decision.action == "skip_superseded", "a newer message replaces the queued work"


async def test_decide_dispatches_when_the_last_message_predates_the_queued_work(
    db_session_factory: async_sessionmaker[AsyncSession],
    caller: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """The boundary the other way: the message that asked for the handoff is older."""
    tenant_id, account_id = caller
    request = _request(tenant_id=tenant_id, account_id=account_id)
    await record_continuation(db_session_factory, request)
    async with db_session_factory() as session:
        row = await get_continuation(session, idempotency_key=request.idempotency_key)
    assert row is not None, "the recorded row must exist to compare against"

    decision = await decide_continuation(
        db_session_factory,
        _live_target(tenant_id),
        request=request,
        now=_NOW,
        latest_user_message_at=row.created_at - timedelta(seconds=1),
        active_turn=False,
    )

    assert decision.action == "dispatch", "the request that caused the handoff cannot supersede it"


async def test_decide_defers_while_a_turn_is_already_running_in_the_thread(
    db_session_factory: async_sessionmaker[AsyncSession],
    caller: tuple[uuid.UUID, uuid.UUID],
) -> None:
    tenant_id, account_id = caller
    request = _request(tenant_id=tenant_id, account_id=account_id)
    await record_continuation(db_session_factory, request)

    decision = await decide_continuation(
        db_session_factory,
        _live_target(tenant_id),
        request=request,
        now=_NOW,
        latest_user_message_at=None,
        active_turn=True,
    )

    assert decision.action == "skip_turn_running", "the in-flight turn finishes first"
    assert decision.message is not None and "stats-bot takes over from your next message here." in (
        decision.message
    ), "the person is told who answers next and that the current message finishes as it is"


async def test_decide_refuses_a_request_that_was_never_recorded(
    db_session_factory: async_sessionmaker[AsyncSession],
    caller: tuple[uuid.UUID, uuid.UUID],
) -> None:
    """Deciding without a row would compare against a clock that does not exist."""
    tenant_id, account_id = caller

    with pytest.raises(DaimonError, match="never recorded"):
        await decide_continuation(
            db_session_factory,
            _live_target(tenant_id),
            request=_request(tenant_id=tenant_id, account_id=account_id),
            now=_NOW,
            latest_user_message_at=None,
            active_turn=False,
        )


async def test_settle_records_delivery_and_skip_distinctly(
    db_session_factory: async_sessionmaker[AsyncSession],
    caller: tuple[uuid.UUID, uuid.UUID],
) -> None:
    tenant_id, account_id = caller
    delivered = _request(tenant_id=tenant_id, account_id=account_id)
    skipped = _request(tenant_id=tenant_id, account_id=account_id)
    await record_continuation(db_session_factory, delivered)
    await record_continuation(db_session_factory, skipped)

    await claim_continuation(
        db_session_factory, idempotency_key=delivered.idempotency_key, now=_NOW
    )
    await claim_continuation(db_session_factory, idempotency_key=skipped.idempotency_key, now=_NOW)
    await settle_continuation(
        db_session_factory,
        idempotency_key=delivered.idempotency_key,
        status="delivered",
        now=_NOW,
    )
    await settle_continuation(
        db_session_factory,
        idempotency_key=skipped.idempotency_key,
        status="skipped",
        now=_NOW,
        skip_reason="skip_superseded",
    )

    async with db_session_factory() as session:
        delivered_row = await get_continuation(session, idempotency_key=delivered.idempotency_key)
        skipped_row = await get_continuation(session, idempotency_key=skipped.idempotency_key)
    assert delivered_row is not None and delivered_row.delivered_at == _NOW, (
        "only a delivered continuation carries a delivery time"
    )
    assert skipped_row is not None and skipped_row.delivered_at is None, (
        "a skipped row must never read as though a turn ran for it"
    )
    assert skipped_row.skip_reason == "skip_superseded", "the skip records why it was skipped"


def _concurrency_dsn() -> str:
    url = os.environ.get("DAIMON_DATABASE__TEST_URL")
    if not url:
        pytest.skip("DAIMON_DATABASE__TEST_URL must be set for the concurrency test")
    return url


async def test_two_dispatchers_racing_one_continuation_claim_it_exactly_once() -> None:
    """Two adapter processes, one queued turn: the database picks the single winner.

    Two real engines, because the guarantee is a Postgres row lock serializing
    separate connections — a shared-connection fixture cannot demonstrate it.
    Runs in the default schema with a freshly minted key so parallel pytest
    workers cannot collide.
    """
    dsn = _concurrency_dsn()
    engine_a = create_async_engine(dsn)
    engine_b = create_async_engine(dsn)
    factory_a = async_sessionmaker(engine_a, expire_on_commit=False)
    factory_b = async_sessionmaker(engine_b, expire_on_commit=False)

    try:
        async with factory_a.begin() as seed:
            tenant = await make_tenant(seed)
            account = await make_account(seed, tenant=tenant)
        request = _request(tenant_id=tenant.id, account_id=account.id)
        await record_continuation(factory_a, request)

        first, second = await asyncio.gather(
            claim_continuation(factory_a, idempotency_key=request.idempotency_key, now=_NOW),
            claim_continuation(factory_b, idempotency_key=request.idempotency_key, now=_NOW),
        )

        assert [first, second].count(True) == 1, (
            "exactly one dispatcher may claim a continuation; "
            f"got {[first, second].count(True)} winners"
        )
        async with factory_a() as session:
            row = await get_continuation(session, idempotency_key=request.idempotency_key)
        assert row is not None and row.status == "claimed", "the winner's claim is committed"
    finally:
        await engine_a.dispose()
        await engine_b.dispose()
