"""Message access follows provisioned tenant state, independently of enrollment policy."""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from daimon.adapters.teams.identity import DENIED_MESSAGE, VerifiedTeamsTurnResolver
from daimon.core.defaults.provisioning import provision_tenant
from daimon.core.stores.tenants import set_provision_status
from microsoft_teams.api import MessageActivity  # pyright: ignore[reportMissingTypeStubs]
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .conftest import AAD_OBJECT_ID, make_message_activity

SECOND_TENANT = str(uuid.UUID(int=99))


def context() -> SimpleNamespace:
    return SimpleNamespace(
        activity=MessageActivity.model_validate(
            make_message_activity(tenant_id=SECOND_TENANT, channel_tenant_id=SECOND_TENANT)
        ),
        send=AsyncMock(),
    )


async def test_provisioned_second_tenant_admitted(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    result = await provision_tenant(
        db_session_factory, platform="teams", workspace_id=SECOND_TENANT
    )
    resolver = VerifiedTeamsTurnResolver(db_session_factory)
    ctx = context()
    authorized = await resolver(ctx)
    assert authorized is not None
    assert authorized.tenant_id == result.tenant_id
    assert authorized.external_user_id == AAD_OBJECT_ID
    ctx.send.assert_not_awaited()


async def test_unprovisioned_second_tenant_refused(
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    resolver = VerifiedTeamsTurnResolver(db_session_factory)
    ctx = context()
    assert await resolver(ctx) is None
    ctx.send.assert_awaited_once_with(DENIED_MESSAGE)


@pytest.mark.parametrize("status,archive", [("pending", False), ("failed", False), ("ready", True)])
async def test_second_tenant_must_be_ready_and_unarchived(
    db_session_factory: async_sessionmaker[AsyncSession], status: str, archive: bool
) -> None:
    result = await provision_tenant(
        db_session_factory, platform="teams", workspace_id=SECOND_TENANT
    )
    await set_provision_status(
        db_session_factory, tenant_id=result.tenant_id, status=status, archive=archive
    )
    ctx = context()
    assert await VerifiedTeamsTurnResolver(db_session_factory)(ctx) is None
    ctx.send.assert_awaited_once_with(DENIED_MESSAGE)
