"""Shared fixtures for top-level integration tests."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from decimal import Decimal

import pytest
import pytest_asyncio
from daimon.core._models import Base
from daimon.core.db import build_engine, build_session_factory
from daimon.core.stores import tenant_ledger
from daimon.testing.db import db_engine as db_engine  # noqa: F401
from daimon.testing.db import db_session as db_session  # noqa: F401
from daimon.testing.db import db_session_factory as db_session_factory  # noqa: F401
from daimon.testing.factories import make_tenant
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker


def _schema_engine_test_dsn() -> str:
    url = os.environ.get("DAIMON_DATABASE__TEST_URL")
    if not url:
        pytest.skip("DAIMON_DATABASE__TEST_URL must be set for integration tests")
    return url


@pytest_asyncio.fixture
async def schema_engine() -> AsyncIterator[
    tuple[AsyncEngine, async_sessionmaker[AsyncSession], uuid.UUID]
]:
    """Engine + sessionmaker scoped to a per-test schema, seeded with a funded tenant.

    The engine is configured with ``schema_translate_map`` so all DDL + DML
    lands in ``test_<uuid>``; the schema is created here and dropped on
    teardown. Used by the routines end-to-end suite, whose scheduler ``run``
    is invoked with this engine via ``_engine_override`` so it shares the
    per-test schema with the test body's seed/verify code.
    """
    dsn = _schema_engine_test_dsn()
    raw_engine = build_engine(dsn)
    schema = f"test_{uuid.uuid4().hex}"
    async with raw_engine.connect() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        mapped = await conn.execution_options(schema_translate_map={None: schema})
        await mapped.run_sync(Base.metadata.create_all)
        await mapped.commit()
    await raw_engine.dispose()

    # Build the *real* engine the test body and the scheduler will share.
    # `execution_options(schema_translate_map=...)` returns a wrapper engine
    # whose checked-out connections all carry the translate map.
    engine = build_engine(dsn).execution_options(
        schema_translate_map={None: schema},
    )
    sm = build_session_factory(engine)

    # Seed a Tenant — the routine row's tenant_id FK points at it.
    # Seed a positive balance so the admission gate (is_over_balance)
    # does not block scheduled fires — these e2e tests exercise the fire path, not the gate.
    async with sm() as s, s.begin():
        tenant = await make_tenant(s, platform="discord", workspace_id="e2e-guild-a")
        tenant_id = tenant.id
        await tenant_ledger.insert_entry(
            s,
            tenant_id=tenant_id,
            delta_usd=Decimal("100"),
            reason="test-seed",
            idempotency_key=f"test-seed:{tenant_id}",
        )

    try:
        yield engine, sm, tenant_id
    finally:
        await engine.dispose()
        # Teardown — drop schema on a fresh raw engine.
        cleanup_engine = build_engine(dsn)
        async with cleanup_engine.connect() as conn:
            await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
            await conn.commit()
        await cleanup_engine.dispose()


@pytest.fixture(autouse=True)
def _ephemeral_scheduler_health_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bind the scheduler liveness responder to an OS-assigned free port.

    Integration tests that boot the real scheduler ``run()`` start its liveness
    responder, which binds ``DAIMON_SCHEDULER__HEALTH_PORT`` (default 8082). The
    default is a fixed port, so under pytest-xdist (``-n auto``) two worker
    processes running scheduler tests at once collide with EADDRINUSE. Port 0
    lets the OS assign a distinct free port per worker, keeping the production
    liveness path exercised while staying parallel-safe.
    """
    monkeypatch.setenv("DAIMON_SCHEDULER__HEALTH_PORT", "0")
