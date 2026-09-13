"""Setup roots keep immutable identities and lifecycle without billed execution."""

from __future__ import annotations

import re
from datetime import UTC, datetime

import httpx
import pytest
from aioresponses import aioresponses
from anthropic.types.beta import BetaManagedAgentsAgent, BetaManagedAgentsModelConfig
from daimon.adapters.slack.app import SlackApp
from daimon.adapters.slack.runtime import SlackRuntime, build_turn_deps
from daimon.adapters.slack.setup_conversations import (
    create_setup_conversation,
    handle_setup_lifecycle,
)
from daimon.core.config import AnthropicSettings, DatabaseSettings, Settings
from daimon.core.errors import DaimonError
from daimon.core.ma_resolver import new_resolver_cache
from daimon.core.scope import DeploymentDefault
from daimon.core.stores.thread_agent_bindings import get_binding
from daimon.testing.factories import make_tenant
from daimon.testing.ma import build_fake_anthropic, list_response
from pydantic import PostgresDsn, SecretStr
from slack_sdk.web.async_client import AsyncWebClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


@pytest.mark.parametrize("is_admin", [False, True])
@pytest.mark.parametrize("target_deleted", [False, True])
async def test_setup_root_routes_daimon_and_retains_target_through_archive_and_delete(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    is_admin: bool,
    target_deleted: bool,
) -> None:
    tenant = await make_tenant(db_session, platform="slack", workspace_id="T_SETUP")
    await db_session.commit()
    now = datetime.now(UTC)
    responder = BetaManagedAgentsAgent(
        id="agent_daimon",
        type="agent",
        name="daimon",
        version=1,
        model=BetaManagedAgentsModelConfig(id="claude-sonnet-4-6", speed="standard"),
        system=None,
        tools=[],
        skills=[],
        mcp_servers=[],
        metadata={
            "daimon_tenant": str(tenant.id),
            "daimon_name": "daimon",
            "daimon_managed": "true",
        },
        created_at=now,
        updated_at=now,
    )
    target = BetaManagedAgentsAgent(
        id="agent_specialist",
        type="agent",
        name="specialist",
        version=1,
        model=BetaManagedAgentsModelConfig(id="claude-sonnet-4-6", speed="standard"),
        system=None,
        tools=[],
        skills=[],
        mcp_servers=[],
        metadata={"daimon_tenant": str(tenant.id), "daimon_name": "specialist"},
        created_at=now,
        updated_at=now,
    )
    requests: list[httpx.Request] = []

    def ma_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v1/agents":
            return list_response([responder.model_dump(mode="json")])
        if request.url.path == "/v1/agents/agent_specialist":
            if target_deleted:
                return httpx.Response(
                    404,
                    json={
                        "type": "error",
                        "error": {"type": "not_found_error", "message": "deleted"},
                    },
                )
            return httpx.Response(200, json=target.model_dump(mode="json"))
        raise AssertionError(f"Unexpected MA call: {request.method} {request.url.path}")

    anthropic = build_fake_anthropic(ma_handler)
    settings = Settings(
        _env_file=None,  # pyright: ignore[reportCallIssue]  # BaseSettings runtime option
        database=DatabaseSettings(url=PostgresDsn("postgresql+asyncpg://test:test@localhost/test")),
        anthropic=AnthropicSettings(api_key=SecretStr("test")),
    )
    cache = new_resolver_cache()
    default = DeploymentDefault(agent_name="specialist")
    async with httpx.AsyncClient() as http_client:
        runtime = SlackRuntime(
            settings=settings,
            anthropic=anthropic,
            sessionmaker=db_session_factory,
            billing_config=None,
            http_client=http_client,
            resolver_cache=cache,
            deployment_default=default,
            turn_deps=build_turn_deps(
                settings,
                anthropic,
                db_session_factory,
                deployment_default=default,
                resolver_cache=cache,
                billing_config=None,
            ),
        )
        with aioresponses() as slack:
            slack.post("https://slack.com/api/auth.test", payload={"ok": True, "user_id": "U_BOT"})
            slack.get(
                re.compile(r"https://slack.com/api/users.info.*"),
                payload={"ok": True, "user": {"is_admin": is_admin}},
            )
            slack.post(
                "https://slack.com/api/chat.postMessage", payload={"ok": True, "ts": "123.456"}
            )
            slack.post("https://slack.com/api/chat.update", payload={"ok": True})
            if target_deleted:
                with pytest.raises(DaimonError, match="no longer exists"):
                    await create_setup_conversation(
                        runtime,
                        AsyncWebClient(token="xoxb-test"),
                        team_id="T_SETUP",
                        channel_id="C_PARENT",
                        user_id="U_OPENER",
                        target_ma_agent_id=target.id,
                    )
                assert not slack.requests, "deleted target must fail before platform creation"
                return
            link = await create_setup_conversation(
                runtime,
                AsyncWebClient(token="xoxb-test"),
                team_id="T_SETUP",
                channel_id="C_PARENT",
                user_id="U_OPENER",
                target_ma_agent_id=target.id,
            )
            assert "C_PARENT-123.456" in link, "entry should return a link to the created thread"
            slack_requests_before = sum(len(calls) for calls in slack.requests.values())
            await SlackApp(runtime=runtime)._orchestrate(  # pyright: ignore[reportPrivateUsage]  # exercise listener turn boundary
                {
                    "type": "app_mention",
                    "ts": "123.456",
                    "user": "U_BOT",
                    "bot_id": "B_BOT",
                    "text": "Mention <@U_BOT>",
                },
                team_id="T_SETUP",
                channel="C_PARENT",
                event_ts="123.456",
                web_client=AsyncWebClient(token="xoxb-test"),
                tenant_id=tenant.id,
            )
            assert sum(len(calls) for calls in slack.requests.values()) == slack_requests_before, (
                "echoed opener must never begin turn admission or Slack role lookup"
            )
            updates = [
                call.kwargs["json"]
                for (method, url), calls in slack.requests.items()
                if method == "POST" and url.path.endswith("chat.update")
                for call in calls
            ]
            assert "specialist" in updates[0]["text"] and "<@U_BOT>" in updates[0]["text"], (
                "opener should name target and actual bot mention"
            )
        assert all(request.method == "GET" for request in requests), (
            "opening setup must not create a session or billed turn"
        )
        assert runtime.deployment_default.agent_name == "specialist", (
            "setup must preserve parent routing"
        )
        for event, archived, deleted in [
            (
                {
                    "type": "message",
                    "subtype": "message_deleted",
                    "channel": "C_PARENT",
                    "deleted_ts": "123.999",
                },
                False,
                False,
            ),
            ({"type": "channel_archive", "channel": "C_PARENT"}, True, False),
            ({"type": "channel_unarchive", "channel": "C_PARENT"}, False, False),
            (
                {
                    "type": "message",
                    "subtype": "message_deleted",
                    "channel": "C_PARENT",
                    "deleted_ts": "123.456",
                },
                False,
                True,
            ),
        ]:
            await handle_setup_lifecycle(runtime, event, team_id="T_SETUP")
            async with db_session_factory() as session:
                binding = await get_binding(
                    session,
                    tenant_id=tenant.id,
                    platform="slack",
                    parent_channel_id="C_PARENT",
                    thread_id="123.456",
                )
            assert binding is not None, "lifecycle retains conversation identity"
            assert binding.responder_ma_agent_id == responder.id, "Daimon must answer setup"
            assert binding.configuration_target_ma_agent_id == target.id, (
                "target must remain a separate concrete identity"
            )
            assert (binding.archived, binding.deleted) == (archived, deleted), (
                "lifecycle should track channel and root events"
            )
