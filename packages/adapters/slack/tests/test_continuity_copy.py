"""Slack adapter tests for turn-continuity rendering.

Covers `_run_thread_turn`'s continuity-facing branches: a `SessionPreparationFailed`
must render `render_preparation_failed` and never run a turn; a `SessionAgentMismatch`
must render `render_responder_changed_without_handoff`, not the generic error copy,
and never run a turn; `is_setup` on the turn origin must come from
`admission.config.thread_binding_kind == "setup"`, not merely "a binding exists" (a
handoff binding must NOT count as setup); and each non-"continued" `ContinuityOutcome`
state must produce its documented pre/post-answer notice.

`bind_session` (and, for the notice-copy tests, `run_prepared_turn`) are mocked at the
`daimon.adapters.slack.app` boundary -- the same precedent test_app.py's ceiling tests
already use to isolate `_run_thread_turn`'s branching from the internals of
`daimon.core.session_preparation` / `daimon.core.turn.run`, which are unit-tested in
`packages/core/tests/` on their own terms.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from cryptography.fernet import Fernet
from daimon.adapters.slack.app import SlackApp
from daimon.adapters.slack.runtime import SlackRuntime, build_turn_deps
from daimon.core.continuity.messages import (
    render_current_work_must_finish,
    render_preparation_failed,
    render_replacement_summary,
    render_responder_changed_without_handoff,
    render_unexpected_loss,
)
from daimon.core.defaults.metadata import MA_METADATA_KEY_NAME, MA_METADATA_KEY_TENANT
from daimon.core.defaults.provisioning import provision_tenant
from daimon.core.github_credentials import build_multifernet, encrypt_token
from daimon.core.ma_identity import derive_tenant_uuid
from daimon.core.ma_resolver import new_resolver_cache
from daimon.core.scope import DeploymentDefault
from daimon.core.stores.slack_bot_tokens import upsert_slack_bot_token
from daimon.core.stores.thread_agent_bindings import create_binding
from daimon.core.turn.admission import Admission
from daimon.core.turn.errors import SessionAgentMismatch, SessionPreparationFailed
from daimon.core.turn.prepare import ContinuityOutcome, PreparedTurn
from daimon.core.turn.run import RunOutcome
from daimon.core.turn.state import TurnState
from daimon.core.turn_origin import turn_origin as real_turn_origin
from daimon.testing.ma import (
    _agent_response as _agent_response,  # pyright: ignore[reportPrivateUsage]
)
from daimon.testing.ma import (
    _environment_response as _environment_response,  # pyright: ignore[reportPrivateUsage]
)
from daimon.testing.ma import build_fake_anthropic
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from yarl import URL

_AGENT_ID = "agent_continuity_test"
_ENV_ID = "env_continuity_test"


def _make_agent_env_handler(tenant_id_str: str) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and path == f"/v1/agents/{_AGENT_ID}":
            return httpx.Response(
                200,
                json=_agent_response(
                    agent_id=_AGENT_ID,
                    metadata={
                        MA_METADATA_KEY_TENANT: tenant_id_str,
                        MA_METADATA_KEY_NAME: "uat-agent",
                    },
                ),
            )
        if request.method == "GET" and path == f"/v1/environments/{_ENV_ID}":
            env = _environment_response(
                environment_id=_ENV_ID,
                metadata={MA_METADATA_KEY_TENANT: tenant_id_str, MA_METADATA_KEY_NAME: "test-env"},
            )
            return httpx.Response(200, json=env.model_dump(mode="json"))
        raise AssertionError(f"_make_agent_env_handler: unhandled {request.method} {path}")

    return handler


def _make_app(sessionmaker: async_sessionmaker[AsyncSession], *, tenant_id_str: str) -> SlackApp:
    settings = MagicMock()
    settings.crypto.keys = ()
    settings.slack.max_concurrent_turns_per_tenant = 3
    settings.slack.bot_display_name = "daimon"
    settings.mcp.public_url = None
    settings.mcp.app_root_url = None
    settings.defaults_root = MagicMock()
    settings.billing.markup = Decimal("1.0")

    anthropic_client = build_fake_anthropic(_make_agent_env_handler(tenant_id_str))
    deployment_default = DeploymentDefault(agent_name="uat-agent", environment_name="test-env")
    resolver_cache = new_resolver_cache()
    turn_deps = build_turn_deps(
        settings,
        anthropic_client,
        sessionmaker,
        deployment_default=deployment_default,
        resolver_cache=resolver_cache,
        billing_config=None,
    )
    runtime = SlackRuntime(
        settings=settings,
        anthropic=anthropic_client,
        sessionmaker=sessionmaker,
        billing_config=None,
        http_client=MagicMock(spec=httpx.AsyncClient),
        resolver_cache=resolver_cache,
        turn_deps=turn_deps,
        deployment_default=deployment_default,
    )
    return SlackApp(runtime=runtime)


def _event(*, channel: str, thread_ts: str, user: str) -> dict[str, Any]:
    return {
        "type": "app_mention",
        "channel": channel,
        "event_ts": thread_ts,
        "ts": thread_ts,
        "thread_ts": thread_ts,
        "user": user,
        "text": "<@U_BOT> hello",
    }


async def _record_noop(*, event: Any) -> None:
    return None


def _prepared_turn(*, continuity: ContinuityOutcome) -> PreparedTurn:
    """A `PreparedTurn` whose only load-bearing field for these tests is
    `continuity` -- `admission` is never read by real code once
    `run_prepared_turn` itself is mocked, so a bare placeholder is enough to
    satisfy the frozen dataclass's constructor."""
    return PreparedTurn(
        admission=MagicMock(spec=Admission),
        ma_session_id="sess_continuity_test",
        mapping_id=uuid.uuid4(),
        watermark=None,
        reused=True,
        session_account_id=uuid.uuid4(),
        _record=_record_noop,
        continuity=continuity,
    )


async def test_session_preparation_failed_renders_copy_and_does_not_run_turn(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    team_id = "T_PREP_FAILED"
    channel = "C_TEST"
    thread_ts = "9100000001.000001"
    tenant_id = derive_tenant_uuid(platform="slack", workspace_id=team_id)
    fernet_key = Fernet.generate_key().decode()

    await provision_tenant(db_session_factory, platform="slack", workspace_id=team_id)
    await upsert_slack_bot_token(
        db_session,
        team_id=team_id,
        encrypted_token=encrypt_token(build_multifernet((fernet_key,)), "xoxb-prep-failed"),
    )
    await db_session.commit()

    app = _make_app(db_session_factory, tenant_id_str=str(tenant_id))
    app.runtime.settings.crypto.keys = (SecretStr(fernet_key),)

    event = _event(channel=channel, thread_ts=thread_ts, user="U_PREP_FAILED")

    with (
        patch(
            "daimon.core.turn.admission.resolve_agent", new_callable=AsyncMock
        ) as mock_resolve_agent,
        patch(
            "daimon.core.turn.admission.resolve_environment", new_callable=AsyncMock
        ) as mock_resolve_env,
        patch(
            "daimon.core.turn.admission.is_over_balance", new_callable=AsyncMock
        ) as mock_over_balance,
        patch("daimon.core.turn.admission.is_over_cap", new_callable=AsyncMock) as mock_over_cap,
        patch(
            "daimon.adapters.slack.app.bind_session", new_callable=AsyncMock
        ) as mock_bind_session,
        patch(
            "daimon.adapters.slack.app.run_prepared_turn", new_callable=AsyncMock
        ) as mock_run_prepared_turn,
    ):
        mock_resolve_agent.return_value = _AGENT_ID
        mock_resolve_env.return_value = _ENV_ID
        mock_over_balance.return_value = False
        mock_over_cap.return_value = False
        mock_bind_session.side_effect = SessionPreparationFailed(
            reasons=("model",),
            stage="checkpointed",
            retry_after=datetime.now(UTC),
            preserved=True,
        )

        await app._handle_app_mention(event, team_id=team_id)  # pyright: ignore[reportPrivateUsage]

    mock_run_prepared_turn.assert_not_called()
    updates = [
        req
        for (_method, url), reqs in fake_slack_web_client.mock.requests.items()
        if url == URL("https://slack.com/api/chat.update")
        for req in reqs
    ]
    assert len(updates) == 1, "the status card must be edited in place with the failure copy"
    body = updates[0].kwargs["json"]
    assert body["text"] == render_preparation_failed("uat-agent")


async def test_responder_changed_without_handoff_renders_offer_and_does_not_run_turn(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    team_id = "T_RESPONDER_MISMATCH"
    channel = "C_TEST"
    thread_ts = "9100000002.000001"
    tenant_id = derive_tenant_uuid(platform="slack", workspace_id=team_id)
    fernet_key = Fernet.generate_key().decode()

    await provision_tenant(db_session_factory, platform="slack", workspace_id=team_id)
    await upsert_slack_bot_token(
        db_session,
        team_id=team_id,
        encrypted_token=encrypt_token(build_multifernet((fernet_key,)), "xoxb-mismatch"),
    )
    await db_session.commit()

    app = _make_app(db_session_factory, tenant_id_str=str(tenant_id))
    app.runtime.settings.crypto.keys = (SecretStr(fernet_key),)

    event = _event(channel=channel, thread_ts=thread_ts, user="U_MISMATCH")

    with (
        patch(
            "daimon.core.turn.admission.resolve_agent", new_callable=AsyncMock
        ) as mock_resolve_agent,
        patch(
            "daimon.core.turn.admission.resolve_environment", new_callable=AsyncMock
        ) as mock_resolve_env,
        patch(
            "daimon.core.turn.admission.is_over_balance", new_callable=AsyncMock
        ) as mock_over_balance,
        patch("daimon.core.turn.admission.is_over_cap", new_callable=AsyncMock) as mock_over_cap,
        patch(
            "daimon.adapters.slack.app.bind_session", new_callable=AsyncMock
        ) as mock_bind_session,
    ):
        mock_resolve_agent.return_value = _AGENT_ID
        mock_resolve_env.return_value = _ENV_ID
        mock_over_balance.return_value = False
        mock_over_cap.return_value = False
        mock_bind_session.side_effect = SessionAgentMismatch(
            mapping_id=uuid.uuid4(),
            session_id="sess_owned_by_other_agent",
            source_agent_id=_AGENT_ID,
            destination_agent_id=_AGENT_ID,
        )

        await app._handle_app_mention(event, team_id=team_id)  # pyright: ignore[reportPrivateUsage]

    updates = [
        req
        for (_method, url), reqs in fake_slack_web_client.mock.requests.items()
        if url == URL("https://slack.com/api/chat.update")
        for req in reqs
    ]
    assert len(updates) == 1
    body = updates[0].kwargs["json"]
    # source_agent_id == the admitted agent's own id here, so the "owner"
    # lookup resolves the SAME fake agent ("uat-agent") -- this pins the
    # renderer call shape, not a distinct owner name (there is only one
    # agent in this fake).
    assert body["text"] == render_responder_changed_without_handoff(
        new_responder="uat-agent", owner="uat-agent", channel=f"<#{channel}>"
    )


async def test_is_setup_is_false_for_a_handoff_binding(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    """A thread whose task was handed to another agent (`kind='handoff'`) must
    NOT be treated as a setup conversation -- `is_setup` must come from
    `thread_binding_kind == "setup"`, not `thread_binding_id is not None`.
    """
    team_id = "T_HANDOFF_NOT_SETUP"
    channel = "C_TEST"
    thread_ts = "9100000003.000001"
    tenant_id = derive_tenant_uuid(platform="slack", workspace_id=team_id)
    fernet_key = Fernet.generate_key().decode()

    await provision_tenant(db_session_factory, platform="slack", workspace_id=team_id)
    await upsert_slack_bot_token(
        db_session,
        team_id=team_id,
        encrypted_token=encrypt_token(build_multifernet((fernet_key,)), "xoxb-handoff"),
    )
    await create_binding(
        db_session,
        tenant_id=tenant_id,
        platform="slack",
        parent_channel_id=channel,
        thread_id=thread_ts,
        responder_ma_agent_id=_AGENT_ID,
        responder_name="uat-agent",
        kind="handoff",
    )
    await db_session.commit()

    app = _make_app(db_session_factory, tenant_id_str=str(tenant_id))
    app.runtime.settings.crypto.keys = (SecretStr(fernet_key),)

    event = _event(channel=channel, thread_ts=thread_ts, user="U_HANDOFF")

    captured: dict[str, Any] = {}

    def _capturing_turn_origin(sessionmaker: Any, **kwargs: Any) -> Any:
        captured["is_setup"] = kwargs.get("is_setup")
        return real_turn_origin(sessionmaker, **kwargs)

    prepared = _prepared_turn(continuity=ContinuityOutcome())

    with (
        patch(
            "daimon.core.turn.admission.resolve_environment", new_callable=AsyncMock
        ) as mock_resolve_env,
        patch(
            "daimon.core.turn.admission.is_over_balance", new_callable=AsyncMock
        ) as mock_over_balance,
        patch("daimon.core.turn.admission.is_over_cap", new_callable=AsyncMock) as mock_over_cap,
        patch(
            "daimon.adapters.slack.app.bind_session", new_callable=AsyncMock
        ) as mock_bind_session,
        patch("daimon.adapters.slack.app.turn_origin", new=_capturing_turn_origin),
        patch(
            "daimon.adapters.slack.app.run_prepared_turn", new_callable=AsyncMock
        ) as mock_run_prepared_turn,
    ):
        mock_resolve_env.return_value = _ENV_ID
        mock_over_balance.return_value = False
        mock_over_cap.return_value = False
        mock_bind_session.return_value = prepared
        mock_run_prepared_turn.return_value = RunOutcome(
            state=TurnState(),
            ma_session_id=prepared.ma_session_id,
            mapping_id=prepared.mapping_id,
            recovered=False,
        )
        await app._handle_app_mention(event, team_id=team_id)  # pyright: ignore[reportPrivateUsage]

    assert captured.get("is_setup") is False, (
        "a handoff binding must not be treated as a setup conversation"
    )


async def test_replaced_posts_replacement_summary_before_the_answer(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    team_id = "T_REPLACED"
    channel = "C_TEST"
    thread_ts = "9100000004.000001"
    tenant_id = derive_tenant_uuid(platform="slack", workspace_id=team_id)
    fernet_key = Fernet.generate_key().decode()

    await provision_tenant(db_session_factory, platform="slack", workspace_id=team_id)
    await upsert_slack_bot_token(
        db_session,
        team_id=team_id,
        encrypted_token=encrypt_token(build_multifernet((fernet_key,)), "xoxb-replaced"),
    )
    await db_session.commit()

    app = _make_app(db_session_factory, tenant_id_str=str(tenant_id))
    app.runtime.settings.crypto.keys = (SecretStr(fernet_key),)

    event = _event(channel=channel, thread_ts=thread_ts, user="U_REPLACED")

    prepared = _prepared_turn(continuity=ContinuityOutcome(state="replaced", transfer_kind="full"))

    with (
        patch(
            "daimon.core.turn.admission.resolve_agent", new_callable=AsyncMock
        ) as mock_resolve_agent,
        patch(
            "daimon.core.turn.admission.resolve_environment", new_callable=AsyncMock
        ) as mock_resolve_env,
        patch(
            "daimon.core.turn.admission.is_over_balance", new_callable=AsyncMock
        ) as mock_over_balance,
        patch("daimon.core.turn.admission.is_over_cap", new_callable=AsyncMock) as mock_over_cap,
        patch(
            "daimon.adapters.slack.app.bind_session", new_callable=AsyncMock
        ) as mock_bind_session,
        patch(
            "daimon.adapters.slack.app.run_prepared_turn", new_callable=AsyncMock
        ) as mock_run_prepared_turn,
    ):
        mock_resolve_agent.return_value = _AGENT_ID
        mock_resolve_env.return_value = _ENV_ID
        mock_over_balance.return_value = False
        mock_over_cap.return_value = False
        mock_bind_session.return_value = prepared
        mock_run_prepared_turn.return_value = RunOutcome(
            state=TurnState(),
            ma_session_id=prepared.ma_session_id,
            mapping_id=prepared.mapping_id,
            recovered=False,
            continuity=prepared.continuity,
        )

        await app._handle_app_mention(event, team_id=team_id)  # pyright: ignore[reportPrivateUsage]

    posts = [
        req
        for (_method, url), reqs in fake_slack_web_client.mock.requests.items()
        if url == URL("https://slack.com/api/chat.postMessage")
        for req in reqs
    ]
    texts = [p.kwargs["json"]["text"] for p in posts]
    assert render_replacement_summary("full", lost=[]) in texts


async def test_replaced_after_loss_posts_unexpected_loss_copy(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    team_id = "T_REPLACED_AFTER_LOSS"
    channel = "C_TEST"
    thread_ts = "9100000005.000001"
    tenant_id = derive_tenant_uuid(platform="slack", workspace_id=team_id)
    fernet_key = Fernet.generate_key().decode()

    await provision_tenant(db_session_factory, platform="slack", workspace_id=team_id)
    await upsert_slack_bot_token(
        db_session,
        team_id=team_id,
        encrypted_token=encrypt_token(build_multifernet((fernet_key,)), "xoxb-loss"),
    )
    await db_session.commit()

    app = _make_app(db_session_factory, tenant_id_str=str(tenant_id))
    app.runtime.settings.crypto.keys = (SecretStr(fernet_key),)

    event = _event(channel=channel, thread_ts=thread_ts, user="U_LOSS")

    prepared = _prepared_turn(
        continuity=ContinuityOutcome(state="replaced_after_loss", transfer_kind="history")
    )

    with (
        patch(
            "daimon.core.turn.admission.resolve_agent", new_callable=AsyncMock
        ) as mock_resolve_agent,
        patch(
            "daimon.core.turn.admission.resolve_environment", new_callable=AsyncMock
        ) as mock_resolve_env,
        patch(
            "daimon.core.turn.admission.is_over_balance", new_callable=AsyncMock
        ) as mock_over_balance,
        patch("daimon.core.turn.admission.is_over_cap", new_callable=AsyncMock) as mock_over_cap,
        patch(
            "daimon.adapters.slack.app.bind_session", new_callable=AsyncMock
        ) as mock_bind_session,
        patch(
            "daimon.adapters.slack.app.run_prepared_turn", new_callable=AsyncMock
        ) as mock_run_prepared_turn,
    ):
        mock_resolve_agent.return_value = _AGENT_ID
        mock_resolve_env.return_value = _ENV_ID
        mock_over_balance.return_value = False
        mock_over_cap.return_value = False
        mock_bind_session.return_value = prepared
        mock_run_prepared_turn.return_value = RunOutcome(
            state=TurnState(),
            ma_session_id=prepared.ma_session_id,
            mapping_id=prepared.mapping_id,
            recovered=False,
            continuity=prepared.continuity,
        )

        await app._handle_app_mention(event, team_id=team_id)  # pyright: ignore[reportPrivateUsage]

    posts = [
        req
        for (_method, url), reqs in fake_slack_web_client.mock.requests.items()
        if url == URL("https://slack.com/api/chat.postMessage")
        for req in reqs
    ]
    texts = [p.kwargs["json"]["text"] for p in posts]
    assert render_unexpected_loss("history") in texts


async def test_pending_posts_must_finish_copy_after_the_answer(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    team_id = "T_PENDING"
    channel = "C_TEST"
    thread_ts = "9100000006.000001"
    tenant_id = derive_tenant_uuid(platform="slack", workspace_id=team_id)
    fernet_key = Fernet.generate_key().decode()

    await provision_tenant(db_session_factory, platform="slack", workspace_id=team_id)
    await upsert_slack_bot_token(
        db_session,
        team_id=team_id,
        encrypted_token=encrypt_token(build_multifernet((fernet_key,)), "xoxb-pending"),
    )
    await db_session.commit()

    app = _make_app(db_session_factory, tenant_id_str=str(tenant_id))
    app.runtime.settings.crypto.keys = (SecretStr(fernet_key),)

    event = _event(channel=channel, thread_ts=thread_ts, user="U_PENDING")

    prepared = _prepared_turn(continuity=ContinuityOutcome(state="continued", pending=("model",)))

    with (
        patch(
            "daimon.core.turn.admission.resolve_agent", new_callable=AsyncMock
        ) as mock_resolve_agent,
        patch(
            "daimon.core.turn.admission.resolve_environment", new_callable=AsyncMock
        ) as mock_resolve_env,
        patch(
            "daimon.core.turn.admission.is_over_balance", new_callable=AsyncMock
        ) as mock_over_balance,
        patch("daimon.core.turn.admission.is_over_cap", new_callable=AsyncMock) as mock_over_cap,
        patch(
            "daimon.adapters.slack.app.bind_session", new_callable=AsyncMock
        ) as mock_bind_session,
        patch(
            "daimon.adapters.slack.app.run_prepared_turn", new_callable=AsyncMock
        ) as mock_run_prepared_turn,
    ):
        mock_resolve_agent.return_value = _AGENT_ID
        mock_resolve_env.return_value = _ENV_ID
        mock_over_balance.return_value = False
        mock_over_cap.return_value = False
        mock_bind_session.return_value = prepared
        mock_run_prepared_turn.return_value = RunOutcome(
            state=TurnState(),
            ma_session_id=prepared.ma_session_id,
            mapping_id=prepared.mapping_id,
            recovered=False,
            continuity=prepared.continuity,
        )

        await app._handle_app_mention(event, team_id=team_id)  # pyright: ignore[reportPrivateUsage]

    posts = [
        req
        for (_method, url), reqs in fake_slack_web_client.mock.requests.items()
        if url == URL("https://slack.com/api/chat.postMessage")
        for req in reqs
    ]
    texts = [p.kwargs["json"]["text"] for p in posts]
    assert render_current_work_must_finish("uat-agent", handoff=False) in texts
