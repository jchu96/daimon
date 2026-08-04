"""Tests for send_message's file_handles parameter.

A handle names a row staged in Postgres by create_file_upload_url and filled
by a direct PUT from the agent's sandbox.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from daimon.adapters.mcp.tools.discord import (
    _build_files_from_handles,  # pyright: ignore[reportPrivateUsage]
)
from daimon.core.stores.file_uploads import create_upload, store_upload_content
from daimon.testing.factories import make_tenant
from fastmcp.exceptions import ToolError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


class _ReusingFactory:
    """Hand the helper the very session the test writes through.

    The helper opens its own session; in a test the rows live in an
    uncommitted, schema-isolated transaction, so it has to reuse this one
    rather than open a second connection that cannot see them.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    def __call__(self) -> _ReusingFactory:
        return self

    async def __aenter__(self) -> AsyncSession:
        return self._session

    async def __aexit__(self, *_: object) -> None:
        return None


def _factory_for(db_session: AsyncSession) -> async_sessionmaker[AsyncSession]:
    return _ReusingFactory(db_session)  # pyright: ignore[reportReturnType]


async def test_build_files_from_handles_reads_uploaded_bytes_byte_identical(
    db_session: AsyncSession,
) -> None:
    """An agent-uploaded file resolves from Postgres and arrives unaltered.

    This is the regression guard for silently truncated attachments: the
    payload carries every byte value, so any encoding or clipping step in the
    path shows up as a mismatch.
    """
    tenant = await make_tenant(db_session)
    payload = bytes(range(256)) * 8
    now = datetime.now(UTC)
    row, token = await create_upload(
        db_session,
        tenant_id=tenant.id,
        title="chart",
        display_filename="chart.png",
        content_type="image/png",
        now=now,
    )
    await store_upload_content(db_session, upload_token=token, data=payload, now=now)

    files = await _build_files_from_handles(
        [row.id],
        session_factory=_factory_for(db_session),
        tenant_id=tenant.id,
    )

    fp = files[0].fp
    fp.seek(0)
    assert fp.read() == payload, "uploaded bytes must reach Discord byte-identical"
    assert files[0].filename == "chart.png", "display filename should ride the upload row"


async def test_build_files_from_handles_rejects_an_upload_with_no_bytes_yet(
    db_session: AsyncSession,
) -> None:
    """A minted-but-never-PUT handle must not post an empty attachment."""
    tenant = await make_tenant(db_session)
    row, _ = await create_upload(
        db_session,
        tenant_id=tenant.id,
        title="chart",
        display_filename="chart.png",
        content_type="image/png",
        now=datetime.now(UTC),
    )

    with pytest.raises(ToolError, match="no bytes yet"):
        await _build_files_from_handles(
            [row.id],
            session_factory=_factory_for(db_session),
            tenant_id=tenant.id,
        )


async def test_build_files_from_handles_raises_with_filename_in_message(
    db_session: AsyncSession,
) -> None:
    """Missing handle surfaces a ToolError whose message names the file —
    agents debug from this string. Spike 028 locked this contract."""

    with pytest.raises(ToolError, match="nope.mp3"):
        await _build_files_from_handles(
            ["nope.mp3"],
            session_factory=_factory_for(db_session),
            tenant_id=uuid.uuid4(),
        )


async def test_build_files_from_handles_rejects_too_many(
    db_session: AsyncSession,
) -> None:

    # 11 names is over the cap; doesn't matter that the files don't exist —
    # the cap check fires first.
    with pytest.raises(ToolError, match="max 10"):
        await _build_files_from_handles(
            [f"f{i}.mp3" for i in range(11)],
            session_factory=_factory_for(db_session),
            tenant_id=uuid.uuid4(),
        )
