"""Integration tests for the credential_requests store — real Postgres.

Covers the atomic single-use consume (the authoritative security gate) and
the platform-user-scoped erasure helpers.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from daimon.core.credential_requests import mint_request_token
from daimon.core.stores import credential_requests as store
from daimon.core.stores.domain import CredentialRequestRow
from daimon.testing.factories import make_account, make_tenant
from sqlalchemy.ext.asyncio import AsyncSession


async def _seed_request(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    account_id: uuid.UUID,
    token: str | None = None,
    requester_platform_user_id: str = "requester-1",
    expires_at: datetime | None = None,
) -> CredentialRequestRow:
    return await store.create_credential_request(
        session,
        token=token or mint_request_token(),
        kind="env",
        tenant_id=tenant_id,
        agent_id=uuid.uuid4(),
        account_id=account_id,
        target="OPENAI_API_KEY",
        mcp_server_url=None,
        requester_platform_user_id=requester_platform_user_id,
        channel_id="chan-1",
        expires_at=expires_at or (datetime.now(tz=UTC) + timedelta(minutes=30)),
    )


async def test_create_credential_request_returns_row_with_used_at_none(
    db_session: AsyncSession,
) -> None:
    tenant = await make_tenant(db_session)
    account = await make_account(db_session, tenant=tenant)

    row = await _seed_request(db_session, tenant_id=tenant.id, account_id=account.id)

    assert row.used_at is None, "a freshly created request must be unused"


async def test_peek_credential_request_returns_row_for_unused_unexpired_token(
    db_session: AsyncSession,
) -> None:
    tenant = await make_tenant(db_session)
    account = await make_account(db_session, tenant=tenant)
    created = await _seed_request(db_session, tenant_id=tenant.id, account_id=account.id)

    peeked = await store.peek_credential_request(db_session, token=created.token)

    assert peeked is not None
    assert peeked.token == created.token


async def test_peek_credential_request_returns_row_even_when_expired(
    db_session: AsyncSession,
) -> None:
    tenant = await make_tenant(db_session)
    account = await make_account(db_session, tenant=tenant)
    created = await _seed_request(
        db_session,
        tenant_id=tenant.id,
        account_id=account.id,
        expires_at=datetime.now(tz=UTC) - timedelta(minutes=1),
    )

    peeked = await store.peek_credential_request(db_session, token=created.token)

    assert peeked is not None, "peek must return an expired row so the caller can tell it apart"


async def test_peek_credential_request_returns_row_even_when_used(
    db_session: AsyncSession,
) -> None:
    tenant = await make_tenant(db_session)
    account = await make_account(db_session, tenant=tenant)
    created = await _seed_request(db_session, tenant_id=tenant.id, account_id=account.id)
    await store.consume_credential_request(
        db_session, token=created.token, now=datetime.now(tz=UTC)
    )

    peeked = await store.peek_credential_request(db_session, token=created.token)

    assert peeked is not None, "peek must return a used row so the caller can tell it apart"
    assert peeked.used_at is not None


async def test_peek_credential_request_returns_none_for_unknown_token(
    db_session: AsyncSession,
) -> None:
    peeked = await store.peek_credential_request(db_session, token=mint_request_token())

    assert peeked is None


async def test_consume_credential_request_sets_used_at_on_first_call(
    db_session: AsyncSession,
) -> None:
    tenant = await make_tenant(db_session)
    account = await make_account(db_session, tenant=tenant)
    created = await _seed_request(db_session, tenant_id=tenant.id, account_id=account.id)
    now = datetime.now(tz=UTC)

    consumed = await store.consume_credential_request(db_session, token=created.token, now=now)

    assert consumed is not None
    assert consumed.used_at == now


async def test_consume_credential_request_second_call_returns_none_and_preserves_used_at(
    db_session: AsyncSession,
) -> None:
    tenant = await make_tenant(db_session)
    account = await make_account(db_session, tenant=tenant)
    created = await _seed_request(db_session, tenant_id=tenant.id, account_id=account.id)
    first_now = datetime.now(tz=UTC)

    first = await store.consume_credential_request(db_session, token=created.token, now=first_now)
    second = await store.consume_credential_request(
        db_session, token=created.token, now=datetime.now(tz=UTC) + timedelta(seconds=5)
    )

    assert first is not None
    assert second is None, "a second consume of an already-used token must return None"

    peeked = await store.peek_credential_request(db_session, token=created.token)
    assert peeked is not None
    assert peeked.used_at == first_now, (
        "used_at must still hold the first call's timestamp, not be overwritten"
    )


async def test_consume_credential_request_returns_none_when_expired(
    db_session: AsyncSession,
) -> None:
    tenant = await make_tenant(db_session)
    account = await make_account(db_session, tenant=tenant)
    created = await _seed_request(
        db_session,
        tenant_id=tenant.id,
        account_id=account.id,
        expires_at=datetime.now(tz=UTC) - timedelta(minutes=1),
    )

    consumed = await store.consume_credential_request(
        db_session, token=created.token, now=datetime.now(tz=UTC)
    )

    assert consumed is None, "an expired token must not be consumable"

    peeked = await store.peek_credential_request(db_session, token=created.token)
    assert peeked is not None
    assert peeked.used_at is None, "a rejected consume must leave used_at NULL"


async def test_consume_credential_request_returns_none_for_unknown_token(
    db_session: AsyncSession,
) -> None:
    consumed = await store.consume_credential_request(
        db_session, token=mint_request_token(), now=datetime.now(tz=UTC)
    )

    assert consumed is None


async def test_delete_credential_requests_for_platform_user_deletes_regardless_of_state(
    db_session: AsyncSession,
) -> None:
    tenant = await make_tenant(db_session)
    account = await make_account(db_session, tenant=tenant)
    unused = await _seed_request(
        db_session,
        tenant_id=tenant.id,
        account_id=account.id,
        requester_platform_user_id="erase-me",
    )
    used = await _seed_request(
        db_session,
        tenant_id=tenant.id,
        account_id=account.id,
        requester_platform_user_id="erase-me",
    )
    await store.consume_credential_request(db_session, token=used.token, now=datetime.now(tz=UTC))
    expired = await _seed_request(
        db_session,
        tenant_id=tenant.id,
        account_id=account.id,
        requester_platform_user_id="erase-me",
        expires_at=datetime.now(tz=UTC) - timedelta(minutes=1),
    )

    deleted = await store.delete_credential_requests_for_platform_user(
        db_session, platform_user_id="erase-me", tenant_id=tenant.id
    )

    assert deleted == 3, "delete must remove unused, used, AND expired rows"
    assert (await store.peek_credential_request(db_session, token=unused.token)) is None
    assert (await store.peek_credential_request(db_session, token=used.token)) is None
    assert (await store.peek_credential_request(db_session, token=expired.token)) is None


async def test_count_credential_requests_for_platform_user_matches_delete_rowcount(
    db_session: AsyncSession,
) -> None:
    tenant = await make_tenant(db_session)
    account = await make_account(db_session, tenant=tenant)
    for _ in range(3):
        await _seed_request(
            db_session,
            tenant_id=tenant.id,
            account_id=account.id,
            requester_platform_user_id="count-me",
        )

    count_before = await store.count_credential_requests_for_platform_user(
        db_session, platform_user_id="count-me", tenant_id=tenant.id
    )
    deleted = await store.delete_credential_requests_for_platform_user(
        db_session, platform_user_id="count-me", tenant_id=tenant.id
    )

    assert count_before == deleted == 3
