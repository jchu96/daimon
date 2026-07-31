"""delete_agent must archive the agent's memory store and clear the binding."""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from unittest.mock import MagicMock

import httpx
import pytest
from daimon.adapters.discord.agent_setup.write import delete_agent
from daimon.adapters.discord.runtime import DiscordRuntime
from daimon.core.ma_identity import derive_agent_uuid
from daimon.core.scope import ChannelScopeRef
from daimon.core.stores.agent_memory_stores import (
    get_memory_store_id,
    insert_memory_store,
)
from daimon.core.stores.scoped_config_read import get_scope
from daimon.core.stores.scoped_config_write import set_fields
from daimon.testing.factories import make_tenant
from daimon.testing.ma import (
    FakeMemoryStoreState,
    NotHandled,
    build_fake_anthropic,
    combine_handlers,
    make_fake_ma_handler,
    make_fake_memory_store_handler,
)

pytestmark = pytest.mark.asyncio


def _make_archive_agent_handler() -> Callable[[httpx.Request], httpx.Response]:
    """`make_fake_ma_handler` doesn't implement POST .../archive — add it here."""

    def handler(request: httpx.Request) -> httpx.Response:
        m = re.fullmatch(r"/v1/agents/(?P<id>[^/]+)/archive", request.url.path)
        if request.method != "POST" or not m:
            raise NotHandled
        now = datetime.now(UTC).isoformat()
        return httpx.Response(
            200,
            json={
                "id": m.group("id"),
                "type": "agent",
                "name": "doomed",
                "version": 2,
                "model": {"id": "claude-sonnet-4-6", "speed": "standard"},
                "system": None,
                "metadata": {},
                "mcp_servers": [],
                "tools": [],
                "skills": [],
                "created_at": now,
                "updated_at": now,
                "archived_at": now,
                "description": None,
            },
        )

    return handler


async def test_delete_agent_archives_memory_store(db_session, db_session_factory) -> None:
    tenant = await make_tenant(db_session)
    mem_state = FakeMemoryStoreState()
    client = build_fake_anthropic(
        combine_handlers(
            _make_archive_agent_handler(),
            make_fake_memory_store_handler(mem_state),
            make_fake_ma_handler(),
        )
    )

    # Create an MA agent tagged for this tenant, then bind a memory store to
    # its derived UUID.
    agent = await client.beta.agents.create(
        name="doomed",
        model="claude-sonnet-4-6",
        metadata={"daimon_tenant": str(tenant.id), "daimon_name": "doomed"},
    )
    agent_uuid = derive_agent_uuid(tenant_id=tenant.id, ma_agent_id=str(agent.id))
    store = await client.beta.memory_stores.create(name="m", description="d")
    await insert_memory_store(
        db_session, tenant_id=tenant.id, agent_id=agent_uuid, memory_store_id=store.id
    )
    await db_session.commit()

    runtime = MagicMock(spec=DiscordRuntime)
    runtime.anthropic = client
    runtime.sessionmaker = db_session_factory

    await delete_agent(runtime, tenant_id=tenant.id, name="doomed")

    assert mem_state.stores[store.id]["archived_at"] is not None
    async with db_session_factory() as s:
        assert await get_memory_store_id(s, tenant_id=tenant.id, agent_id=agent_uuid) is None


def _make_failing_store_archive_handler() -> Callable[[httpx.Request], httpx.Response]:
    """500 on memory-store archive — simulates a transient MA outage."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and re.fullmatch(
            r"/v1/memory_stores/[^/]+/archive", request.url.path
        ):
            return httpx.Response(
                500,
                json={"type": "error", "error": {"type": "api_error", "message": "boom"}},
            )
        raise NotHandled

    return handler


async def test_delete_agent_succeeds_when_store_archive_fails(
    db_session, db_session_factory
) -> None:
    """Transient store-archive failure must not fail agent deletion.

    The agent is already archived when the store archive runs; a raise here
    would strand the flow with no retry path (archived agents are filtered
    from lookup). delete_agent degrades best-effort instead.
    """
    tenant = await make_tenant(db_session)
    mem_state = FakeMemoryStoreState()
    client = build_fake_anthropic(
        combine_handlers(
            _make_archive_agent_handler(),
            _make_failing_store_archive_handler(),
            make_fake_memory_store_handler(mem_state),
            make_fake_ma_handler(),
        )
    )

    agent = await client.beta.agents.create(
        name="doomed",
        model="claude-sonnet-4-6",
        metadata={"daimon_tenant": str(tenant.id), "daimon_name": "doomed"},
    )
    agent_uuid = derive_agent_uuid(tenant_id=tenant.id, ma_agent_id=str(agent.id))
    store = await client.beta.memory_stores.create(name="m", description="d")
    await insert_memory_store(
        db_session, tenant_id=tenant.id, agent_id=agent_uuid, memory_store_id=store.id
    )
    await db_session.commit()

    runtime = MagicMock(spec=DiscordRuntime)
    runtime.anthropic = client
    runtime.sessionmaker = db_session_factory

    # Must not raise despite the 500 from the store archive.
    await delete_agent(runtime, tenant_id=tenant.id, name="doomed")

    # Store archive never landed; the binding row remains (inert — the agent
    # is archived, so nothing re-reads it).
    assert mem_state.stores[store.id]["archived_at"] is None
    async with db_session_factory() as s:
        assert await get_memory_store_id(s, tenant_id=tenant.id, agent_id=agent_uuid) == store.id


async def test_delete_agent_clears_channel_default_naming_the_agent(
    db_session, db_session_factory
) -> None:
    """Deleting the channel's current default must not leave a row naming a dead agent.

    A channel_config row pointing at an archived agent makes every turn in that
    channel fail to resolve until an admin re-scopes by hand.
    """
    tenant = await make_tenant(db_session)
    scope = ChannelScopeRef(tenant_id=tenant.id, channel_id="C_DELETED_DEFAULT")
    await set_fields(db_session, scope=scope, tenant_id=tenant.id, agent_name="doomed")
    await db_session.commit()

    client = build_fake_anthropic(
        combine_handlers(
            _make_archive_agent_handler(),
            make_fake_memory_store_handler(FakeMemoryStoreState()),
            make_fake_ma_handler(),
        )
    )
    await client.beta.agents.create(
        name="doomed",
        model="claude-sonnet-4-6",
        metadata={"daimon_tenant": str(tenant.id), "daimon_name": "doomed"},
    )

    runtime = MagicMock(spec=DiscordRuntime)
    runtime.anthropic = client
    runtime.sessionmaker = db_session_factory

    await delete_agent(runtime, tenant_id=tenant.id, name="doomed")

    async with db_session_factory() as s:
        assert await get_scope(s, scope=scope) is None, (
            "the channel row naming the deleted agent must be gone so resolution "
            "falls through the cascade instead of naming a dead agent"
        )


async def test_delete_agent_leaves_scope_rows_naming_another_agent_untouched(
    db_session, db_session_factory
) -> None:
    """Only rows naming the deleted agent move; a sibling channel keeps its default."""
    tenant = await make_tenant(db_session)
    doomed_scope = ChannelScopeRef(tenant_id=tenant.id, channel_id="C_DOOMED")
    survivor_scope = ChannelScopeRef(tenant_id=tenant.id, channel_id="C_SURVIVOR")
    await set_fields(db_session, scope=doomed_scope, tenant_id=tenant.id, agent_name="doomed")
    await set_fields(db_session, scope=survivor_scope, tenant_id=tenant.id, agent_name="keeper")
    await db_session.commit()

    client = build_fake_anthropic(
        combine_handlers(
            _make_archive_agent_handler(),
            make_fake_memory_store_handler(FakeMemoryStoreState()),
            make_fake_ma_handler(),
        )
    )
    await client.beta.agents.create(
        name="doomed",
        model="claude-sonnet-4-6",
        metadata={"daimon_tenant": str(tenant.id), "daimon_name": "doomed"},
    )

    runtime = MagicMock(spec=DiscordRuntime)
    runtime.anthropic = client
    runtime.sessionmaker = db_session_factory

    await delete_agent(runtime, tenant_id=tenant.id, name="doomed")

    async with db_session_factory() as s:
        assert await get_scope(s, scope=doomed_scope) is None, (
            "the channel row naming the deleted agent must be gone"
        )
        survivor = await get_scope(s, scope=survivor_scope)
        assert survivor is not None, "a channel row naming a different agent must survive"
        assert survivor.agent_name == "keeper", (
            "a channel row naming a different agent must keep its agent_name"
        )
