"""Shared fixtures for CLI `commands` tests.

`schema_sessionmaker` is used by tests that invoke the real Typer CLI via
`CliRunner` — each command owns its own `asyncio.run` loop, so the fixture
uses `NullPool` to keep asyncpg connections bound to the loop that opened
them (a per-test-schema `db_session`/`db_session_factory` pair from
`daimon.testing.db` shares a single connection across the whole test and
would break across separate `asyncio.run` calls).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from urllib.parse import urlparse

import pytest_asyncio
from daimon.core._models import Base
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool


def _require_test_dsn() -> str:
    url = os.environ.get("DAIMON_DATABASE__TEST_URL")
    if not url:
        raise RuntimeError("DAIMON_DATABASE__TEST_URL must be set to run these tests.")
    if "test" not in urlparse(url).path:
        raise RuntimeError("Refusing to run destructive fixtures against a non-test DB.")
    return url


@pytest_asyncio.fixture
async def schema_sessionmaker() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Fresh schema + a NullPool sessionmaker bound to it."""
    dsn = _require_test_dsn()
    schema = f"test_{uuid.uuid4().hex}"
    engine = create_async_engine(dsn, poolclass=NullPool)

    async with engine.connect() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
        conn2 = await conn.execution_options(schema_translate_map={None: schema})
        await conn2.run_sync(Base.metadata.create_all)
        await conn.commit()

    sessionmaker = async_sessionmaker(
        engine.execution_options(schema_translate_map={None: schema}),
        expire_on_commit=False,
        class_=AsyncSession,
    )
    try:
        yield sessionmaker
    finally:
        async with engine.connect() as conn:
            await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
            await conn.commit()
        await engine.dispose()
