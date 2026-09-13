"""Setup routing remains shared while defaults and caller sessions stay independent."""

from __future__ import annotations

import pytest
from daimon.core.errors import DaimonError
from daimon.core.scope import DeploymentDefault, ScopeContext
from daimon.core.stores.accounts import delete_account
from daimon.core.stores.scoped_config_read import is_agent_reachable_in_tenant, resolve
from daimon.core.stores.thread_agent_bindings import (
    create_binding,
    get_binding,
    list_active_bindings,
    update_channel_lifecycle,
    update_lifecycle,
)
from daimon.testing.factories import make_account, make_tenant
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession


async def test_thread_wins_without_changing_environment_or_reachability(
    db_session: AsyncSession,
) -> None:
    tenant = await make_tenant(db_session)
    caller = await make_account(db_session, tenant=tenant)
    binding = await create_binding(
        db_session,
        tenant_id=tenant.id,
        platform="discord",
        parent_channel_id="channel",
        thread_id="setup",
        responder_ma_agent_id="agent_daimon",
        responder_name="daimon",
        configuration_target_ma_agent_id="agent_specialist",
        configuration_target_name="specialist",
        creator_account_id=caller.id,
    )
    default = DeploymentDefault(agent_name="specialist", environment_name="science")
    context = ScopeContext(
        tenant_id=tenant.id,
        platform="discord",
        channel_id="channel",
        thread_id="setup",
        account_id=caller.id,
    )
    configured = await resolve(db_session, context=context, default=default)
    assert configured.responder_ma_agent_id == "agent_daimon", "thread selects concrete Daimon"
    assert configured.agent_name_tier == "thread", "thread should outrank the parent"
    assert configured.configuration_target_ma_agent_id == "agent_specialist", (
        "target stays distinct"
    )
    assert configured.environment_name == "science", "setup must retain the environment cascade"
    assert configured.environment_name_tier == "deployment", "environment has no thread tier"
    parent = await resolve(
        db_session, context=context.model_copy(update={"thread_id": None}), default=default
    )
    assert parent.agent_name == "specialist", "opening setup must not change the parent responder"
    assert not await is_agent_reachable_in_tenant(
        db_session, tenant_id=tenant.id, agent_name="daimon", default=default
    ), "thread bindings must not affect mutation reachability"
    await delete_account(db_session, account_id=caller.id)
    retained = await get_binding(
        db_session,
        tenant_id=tenant.id,
        platform="discord",
        parent_channel_id="channel",
        thread_id="setup",
    )
    assert retained is not None and retained.id == binding.id, "creator erasure retains shared work"
    assert retained.creator_account_id is None, "erasure removes attribution"


async def test_lifecycle_retains_identity_and_omits_closed_conversations(
    db_session: AsyncSession,
) -> None:
    tenant = await make_tenant(db_session)
    for index in range(12):
        await create_binding(
            db_session,
            tenant_id=tenant.id,
            platform="slack",
            parent_channel_id="channel",
            thread_id=str(index),
            responder_ma_agent_id="agent_daimon",
            responder_name="daimon",
            configuration_target_ma_agent_id="deleted_target",
            configuration_target_name="specialist",
        )
    assert (
        len(
            await list_active_bindings(
                db_session, tenant_id=tenant.id, platform="slack", parent_channel_id="channel"
            )
        )
        == 10
    ), "recent list must stay bounded"
    assert not await list_active_bindings(
        db_session, tenant_id=tenant.id, platform="slack", parent_channel_id="other"
    ), "list cannot cross parent channels"
    await update_channel_lifecycle(
        db_session,
        tenant_id=tenant.id,
        platform="slack",
        parent_channel_id="channel",
        archived=True,
    )
    assert not await list_active_bindings(
        db_session, tenant_id=tenant.id, platform="slack", parent_channel_id="channel"
    ), "channel archive covers more than the visible ten"
    context = ScopeContext(
        tenant_id=tenant.id, platform="slack", channel_id="channel", thread_id="0"
    )
    archived = await resolve(db_session, context=context, default=DeploymentDefault())
    assert archived.configuration_target_ma_agent_id == "deleted_target", (
        "archiving retains exact target"
    )
    await update_channel_lifecycle(
        db_session,
        tenant_id=tenant.id,
        platform="slack",
        parent_channel_id="channel",
        archived=False,
    )
    await update_lifecycle(
        db_session,
        tenant_id=tenant.id,
        platform="slack",
        parent_channel_id="channel",
        thread_id="0",
        deleted=True,
    )
    await update_lifecycle(
        db_session,
        tenant_id=tenant.id,
        platform="slack",
        parent_channel_id="channel",
        thread_id="0",
        deleted=False,
    )
    with pytest.raises(DaimonError, match="deleted"):
        await resolve(db_session, context=context, default=DeploymentDefault(agent_name="other"))
    assert (
        len(
            await list_active_bindings(
                db_session, tenant_id=tenant.id, platform="slack", parent_channel_id="channel"
            )
        )
        == 10
    ), "reopened remaining conversations return"


async def test_binding_location_is_unique_and_tenant_scoped(db_session: AsyncSession) -> None:
    first = await make_tenant(db_session)
    second = await make_tenant(db_session)
    for tenant in (first, second):
        await create_binding(
            db_session,
            tenant_id=tenant.id,
            platform="discord",
            parent_channel_id="channel",
            thread_id="setup",
            responder_ma_agent_id="agent_daimon",
            responder_name="daimon",
        )
    with pytest.raises(IntegrityError):
        async with db_session.begin_nested():
            await create_binding(
                db_session,
                tenant_id=first.id,
                platform="discord",
                parent_channel_id="channel",
                thread_id="setup",
                responder_ma_agent_id="agent_other",
                responder_name="other",
            )
