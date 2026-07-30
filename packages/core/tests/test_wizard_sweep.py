"""Tests for daimon.core.wizard_sweep."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from daimon.core.stores import wizard_session as store
from daimon.core.wizard_sweep import sweep_expired_wizard_sessions
from daimon.testing.factories import make_wizard_session
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

NOW = datetime(2026, 5, 29, 12, 0, 0, tzinfo=UTC)


async def test_sweep_flips_open_row_past_deadline_to_abandoned(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    created = await make_wizard_session(
        db_session, now=NOW - timedelta(hours=2), expires_at=NOW - timedelta(hours=1)
    )
    await db_session.commit()

    count = await sweep_expired_wizard_sessions(db_session_factory, now=NOW)

    assert count == 1
    refetched = await store.get_wizard_session(db_session, short_id=created.id)
    assert refetched is not None
    assert refetched.status == "abandoned"


async def test_sweep_leaves_open_row_before_deadline_untouched(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    created = await make_wizard_session(db_session, now=NOW, expires_at=NOW + timedelta(hours=1))
    await db_session.commit()

    count = await sweep_expired_wizard_sessions(db_session_factory, now=NOW)

    assert count == 0
    refetched = await store.get_wizard_session(db_session, short_id=created.id)
    assert refetched is not None
    assert refetched.status == "open"


async def test_sweep_leaves_submitted_row_past_deadline_untouched(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    created = await make_wizard_session(
        db_session,
        now=NOW - timedelta(hours=2),
        expires_at=NOW - timedelta(hours=1),
        status="submitted",
    )
    await db_session.commit()

    count = await sweep_expired_wizard_sessions(db_session_factory, now=NOW)

    assert count == 0
    refetched = await store.get_wizard_session(db_session, short_id=created.id)
    assert refetched is not None
    assert refetched.status == "submitted"


async def test_sweep_return_value_equals_number_flipped(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    for _ in range(3):
        await make_wizard_session(
            db_session, now=NOW - timedelta(hours=2), expires_at=NOW - timedelta(hours=1)
        )
    await db_session.commit()

    count = await sweep_expired_wizard_sessions(db_session_factory, now=NOW)

    assert count == 3


async def test_second_sweep_at_same_now_is_idempotent(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    await make_wizard_session(
        db_session, now=NOW - timedelta(hours=2), expires_at=NOW - timedelta(hours=1)
    )
    await db_session.commit()

    first = await sweep_expired_wizard_sessions(db_session_factory, now=NOW)
    second = await sweep_expired_wizard_sessions(db_session_factory, now=NOW)

    assert first == 1
    assert second == 0, "a second sweep at the same instant must flip nothing"
