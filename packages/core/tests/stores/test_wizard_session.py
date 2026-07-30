"""Integration tests for the wizard_session store — real Postgres.

Covers the read/write lifecycle, the two single-statement gates
(`update_wizard_state`, `try_claim_submit`), the bounded expiry sweep, and
the platform-user-scoped erasure helpers.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from daimon.core.stores import wizard_session as store
from daimon.core.stores.domain import WizardSessionRow
from daimon.testing.factories import make_account, make_tenant, make_wizard_session
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


async def test_create_wizard_session_returns_pydantic_row_and_get_matches(
    db_session: AsyncSession,
) -> None:
    created = await make_wizard_session(db_session)

    assert isinstance(created, WizardSessionRow), "store must return Pydantic, not ORM"

    fetched = await store.get_wizard_session(db_session, short_id=created.id)

    assert fetched == created


async def test_get_wizard_session_returns_none_for_unknown_id(
    db_session: AsyncSession,
) -> None:
    fetched = await store.get_wizard_session(db_session, short_id="doesnotexist")

    assert fetched is None


async def test_update_wizard_state_on_open_unexpired_row_returns_one_and_persists(
    db_session: AsyncSession,
) -> None:
    created = await make_wizard_session(db_session, answers={}, current_step=0)
    now = created.updated_at + timedelta(seconds=5)

    rowcount = await store.update_wizard_state(
        db_session,
        short_id=created.id,
        answers={"choice": ["a"]},
        current_step=1,
        expected_updated_at=created.updated_at,
        now=now,
    )

    assert rowcount == 1
    refetched = await store.get_wizard_session(db_session, short_id=created.id)
    assert refetched is not None
    assert refetched.answers == {"choice": ["a"]}
    assert refetched.current_step == 1
    assert refetched.updated_at == now
    assert refetched.updated_at > created.updated_at


async def test_update_wizard_state_returns_zero_for_submitted_row(
    db_session: AsyncSession,
) -> None:
    created = await make_wizard_session(db_session, status="submitted")

    rowcount = await store.update_wizard_state(
        db_session,
        short_id=created.id,
        answers={"choice": ["a"]},
        current_step=1,
        expected_updated_at=created.updated_at,
        now=datetime.now(UTC),
    )

    assert rowcount == 0
    refetched = await store.get_wizard_session(db_session, short_id=created.id)
    assert refetched is not None
    assert refetched.answers == created.answers, "a no-op update must leave answers untouched"
    assert refetched.current_step == created.current_step


async def test_update_wizard_state_returns_zero_for_expired_row(
    db_session: AsyncSession,
) -> None:
    now = datetime.now(UTC)
    created = await make_wizard_session(
        db_session,
        now=now - timedelta(hours=2),
        expires_at=now - timedelta(hours=1),
    )

    rowcount = await store.update_wizard_state(
        db_session,
        short_id=created.id,
        answers={"choice": ["a"]},
        current_step=1,
        expected_updated_at=created.updated_at,
        now=now,
    )

    assert rowcount == 0
    refetched = await store.get_wizard_session(db_session, short_id=created.id)
    assert refetched is not None
    assert refetched.answers == created.answers


async def test_update_wizard_state_returns_zero_for_deleted_row(
    db_session: AsyncSession,
) -> None:
    created = await make_wizard_session(db_session, requester_platform_user_id="del-me")
    tenant_id = created.tenant_id
    deleted = await store.delete_wizard_sessions_for_platform_user(
        db_session, platform_user_id="del-me", tenant_id=tenant_id
    )
    assert deleted == 1

    rowcount = await store.update_wizard_state(
        db_session,
        short_id=created.id,
        answers={"choice": ["a"]},
        current_step=1,
        expected_updated_at=created.updated_at,
        now=datetime.now(UTC),
    )

    assert rowcount == 0, "a tap on a purged row must write nothing"
    assert (await store.get_wizard_session(db_session, short_id=created.id)) is None


async def test_update_wizard_state_returns_zero_when_the_row_changed_since_it_was_read(
    db_session: AsyncSession,
) -> None:
    """Two taps computed from the same base row: the second must lose rather
    than overwrite the first's answer with state that predates it."""
    created = await make_wizard_session(db_session, answers={}, current_step=0)
    first_now = created.updated_at + timedelta(seconds=1)
    second_now = created.updated_at + timedelta(seconds=2)

    first = await store.update_wizard_state(
        db_session,
        short_id=created.id,
        answers={"choice": ["a"]},
        current_step=1,
        expected_updated_at=created.updated_at,
        now=first_now,
    )
    second = await store.update_wizard_state(
        db_session,
        short_id=created.id,
        answers={},
        current_step=1,
        expected_updated_at=created.updated_at,
        now=second_now,
    )

    assert first == 1, "the tap that read the current row must win"
    assert second == 0, "a tap computed from a superseded row must write nothing"
    refetched = await store.get_wizard_session(db_session, short_id=created.id)
    assert refetched is not None
    assert refetched.answers == {"choice": ["a"]}, (
        "the losing tap must not erase the answer the winning tap recorded"
    )
    assert refetched.updated_at == first_now


async def test_try_claim_submit_returns_row_on_first_call_and_none_on_second(
    db_session: AsyncSession,
) -> None:
    created = await make_wizard_session(db_session)
    now = datetime.now(UTC)

    first = await store.try_claim_submit(
        db_session, short_id=created.id, answers={"choice": ["a"]}, current_step=1, now=now
    )
    second = await store.try_claim_submit(
        db_session,
        short_id=created.id,
        answers={"choice": ["a"]},
        current_step=1,
        now=now + timedelta(seconds=1),
    )

    assert first is not None
    assert first.status == "submitted"
    assert second is None, "a second claim of an already-submitted row must return None"


async def test_concurrent_try_claim_submit_yields_exactly_one_winner(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Two Submit taps racing the same row: one wins, one gets None.

    A second winner would mean a second billed turn spawned from the same
    form. `try_claim_submit`'s single predicated UPDATE is the entire gate —
    asserting the race empirically is the difference between trusting the
    SQL shape and knowing it holds.
    """
    created = await make_wizard_session(db_session)
    now = datetime.now(UTC)

    async def claim() -> WizardSessionRow | None:
        async with db_session_factory() as session:
            row = await store.try_claim_submit(
                session, short_id=created.id, answers={"choice": ["a"]}, current_step=1, now=now
            )
            await session.commit()
            return row

    first, second = await asyncio.gather(claim(), claim())

    winners = [row for row in (first, second) if row is not None]
    assert len(winners) == 1, (
        "exactly one racing claim may submit a wizard session; a second claim "
        f"winning would mean a second billed turn; got {len(winners)} winners"
    )


async def test_abandon_expired_wizard_sessions_flips_only_open_past_expiry_rows(
    db_session: AsyncSession,
) -> None:
    now = datetime.now(UTC)
    expired_open = await make_wizard_session(
        db_session, now=now - timedelta(hours=2), expires_at=now - timedelta(hours=1)
    )
    still_open = await make_wizard_session(db_session, now=now, expires_at=now + timedelta(hours=1))
    expired_submitted = await make_wizard_session(
        db_session,
        now=now - timedelta(hours=2),
        expires_at=now - timedelta(hours=1),
        status="submitted",
    )

    flipped = await store.abandon_expired_wizard_sessions(db_session, now=now)

    assert flipped == 1
    after_expired_open = await store.get_wizard_session(db_session, short_id=expired_open.id)
    assert after_expired_open is not None
    assert after_expired_open.status == "abandoned"
    after_still_open = await store.get_wizard_session(db_session, short_id=still_open.id)
    assert after_still_open is not None
    assert after_still_open.status == "open", "an unexpired row must not be abandoned"
    after_expired_submitted = await store.get_wizard_session(
        db_session, short_id=expired_submitted.id
    )
    assert after_expired_submitted is not None
    assert after_expired_submitted.status == "submitted", "a submitted row must not be touched"


async def test_abandon_expired_wizard_sessions_honours_limit(
    db_session: AsyncSession,
) -> None:
    now = datetime.now(UTC)
    for _ in range(3):
        await make_wizard_session(
            db_session, now=now - timedelta(hours=2), expires_at=now - timedelta(hours=1)
        )

    flipped = await store.abandon_expired_wizard_sessions(db_session, now=now, limit=2)

    assert flipped == 2


async def test_delete_wizard_sessions_for_platform_user_deletes_regardless_of_status(
    db_session: AsyncSession,
) -> None:
    tenant = await make_tenant(db_session)
    account = await make_account(db_session, tenant=tenant)
    open_row = await make_wizard_session(
        db_session, tenant=tenant, account=account, requester_platform_user_id="erase-me"
    )
    submitted_row = await make_wizard_session(
        db_session,
        tenant=tenant,
        account=account,
        requester_platform_user_id="erase-me",
        status="submitted",
    )
    abandoned_row = await make_wizard_session(
        db_session,
        tenant=tenant,
        account=account,
        requester_platform_user_id="erase-me",
        status="abandoned",
    )

    deleted = await store.delete_wizard_sessions_for_platform_user(
        db_session, platform_user_id="erase-me", tenant_id=tenant.id
    )

    assert deleted == 3, "delete must remove open, submitted, AND abandoned rows"
    assert (await store.get_wizard_session(db_session, short_id=open_row.id)) is None
    assert (await store.get_wizard_session(db_session, short_id=submitted_row.id)) is None
    assert (await store.get_wizard_session(db_session, short_id=abandoned_row.id)) is None


async def test_delete_wizard_sessions_for_platform_user_is_idempotent(
    db_session: AsyncSession,
) -> None:
    tenant = await make_tenant(db_session)
    account = await make_account(db_session, tenant=tenant)
    await make_wizard_session(
        db_session, tenant=tenant, account=account, requester_platform_user_id="idem-me"
    )

    first = await store.delete_wizard_sessions_for_platform_user(
        db_session, platform_user_id="idem-me", tenant_id=tenant.id
    )
    second = await store.delete_wizard_sessions_for_platform_user(
        db_session, platform_user_id="idem-me", tenant_id=tenant.id
    )

    assert first == 1
    assert second == 0, "a re-run delete of an already-erased user must be a no-op"


async def test_delete_wizard_sessions_for_platform_user_does_not_touch_other_tenant(
    db_session: AsyncSession,
) -> None:
    tenant_a = await make_tenant(db_session, workspace_id="guild-a")
    tenant_b = await make_tenant(db_session, workspace_id="guild-b")
    account_a = await make_account(db_session, tenant=tenant_a)
    account_b = await make_account(db_session, tenant=tenant_b)
    row_a = await make_wizard_session(
        db_session,
        tenant=tenant_a,
        account=account_a,
        requester_platform_user_id="shared-platform-id",
    )
    row_b = await make_wizard_session(
        db_session,
        tenant=tenant_b,
        account=account_b,
        requester_platform_user_id="shared-platform-id",
    )

    deleted = await store.delete_wizard_sessions_for_platform_user(
        db_session, platform_user_id="shared-platform-id", tenant_id=tenant_a.id
    )

    assert deleted == 1
    assert (await store.get_wizard_session(db_session, short_id=row_a.id)) is None
    survivor = await store.get_wizard_session(db_session, short_id=row_b.id)
    assert survivor is not None, (
        "another tenant's row sharing the same platform user id must survive"
    )


async def test_count_wizard_sessions_for_platform_user_matches_delete_rowcount(
    db_session: AsyncSession,
) -> None:
    tenant = await make_tenant(db_session)
    account = await make_account(db_session, tenant=tenant)
    for _ in range(3):
        await make_wizard_session(
            db_session, tenant=tenant, account=account, requester_platform_user_id="count-me"
        )

    count_before = await store.count_wizard_sessions_for_platform_user(
        db_session, platform_user_id="count-me", tenant_id=tenant.id
    )
    deleted = await store.delete_wizard_sessions_for_platform_user(
        db_session, platform_user_id="count-me", tenant_id=tenant.id
    )

    assert count_before == deleted == 3
