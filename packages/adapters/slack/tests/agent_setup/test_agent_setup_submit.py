"""Tests for agent_setup/submit.py.

Covers:
- Pure evaluators: response_action keyed to correct input block_id
- Secret-paste: key-name validation, cap, byte limit, value-absence guarantee
- edit-repo: blank PAT = keep (proceed=True, pat_replace=False)
- run_* handlers via FakeSlackWebClient:
  - run_new_agent_submission (admin, write succeeds) posts :white_check_mark: ephemeral; no views_update
  - run_paste_secrets_submission (admin, 2 pairs) posts count ephemeral without secret values; no views_update
  - the four always-open paths (new/fork/edit_repo/paste_secrets): a non-admin
    submission succeeds with no permission-refusal ephemeral, including against
    the workspace's currently-default agent for the two per-agent attachments
  - the three field-conditional paths (edit_agent/add_skill/add_mcp): a
    non-admin submission proceeds on an unreachable agent and is refused
    before any MA request on a reachable one, via the shared
    ``agent_setup.gate`` helper
"""

from __future__ import annotations

import json
import re
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
import yarl
from daimon.adapters.slack.agent_setup import submit as submit_mod
from daimon.adapters.slack.agent_setup.submit import (
    _SECRET_CAP,
    SubmitDecision,
    evaluate_edit_agent_submission,
    evaluate_edit_repo_submission,
    evaluate_fork_agent_submission,
    evaluate_new_agent_submission,
    evaluate_paste_secrets_submission,
    run_add_mcp_submission,
    run_add_skill_submission,
    run_edit_agent_submission,
    run_edit_repo_submission,
    run_fork_agent_submission,
    run_new_agent_submission,
    run_paste_secrets_submission,
)
from daimon.adapters.slack.runtime import SlackRuntime
from daimon.core.scope import TenantScopeRef
from daimon.core.skill_sync import SyncReport
from daimon.core.stores.scoped_config_write import set_fields
from daimon.testing.factories import make_tenant
from daimon.testing.ma import build_fake_anthropic, make_fake_ma_handler
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

# ---------------------------------------------------------------------------
# Helpers for building minimal Slack view_submission payloads
# ---------------------------------------------------------------------------

_TEAM_ID = "T_TEST"
_USER_ID = "U_TEST"
_CHANNEL_ID = "C_TEST"
_AGENT_NAME = "my-agent"

_USERS_INFO_PATTERN = re.compile(r"https://slack\.com/api/users\.info.*")

_ADMIN_USERS_INFO_PAYLOAD = {
    "ok": True,
    "user": {
        "id": _USER_ID,
        "name": "admin",
        "is_admin": True,
        "is_owner": False,
        "is_primary_owner": False,
    },
}


def _override_users_info_admin(mock: Any) -> None:
    """Replace the conftest non-admin users.info stub with an admin one.

    aioresponses stores matchers by uuid key in insertion order — the first
    matching entry wins. The conftest registers the non-admin baseline with
    repeat=True so a plain .get() append never takes effect. This helper removes
    existing pattern-matched users.info entries and re-registers an admin payload.
    """
    to_remove = [
        k
        for k, v in mock._matches.items()  # type: ignore[attr-defined]
        if getattr(v, "url_or_pattern", None) == _USERS_INFO_PATTERN
    ]
    for k in to_remove:
        del mock._matches[k]  # type: ignore[attr-defined]
    mock.get(  # pyright: ignore[reportUnknownMemberType]
        _USERS_INFO_PATTERN,
        payload=_ADMIN_USERS_INFO_PAYLOAD,
        repeat=True,
    )


def _build_runtime_no_db(fernet_key: str = "dummy") -> SlackRuntime:
    """Build a SlackRuntime with fake MA transport and a dummy (unused) sessionmaker.

    Suitable for run_* handlers that do not use runtime.sessionmaker
    (e.g. run_new_agent_submission).
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker

    settings: MagicMock = MagicMock()
    settings.crypto.keys = (SecretStr(fernet_key),)
    settings.mcp.public_url = None
    settings.mcp.jwt_secret = None
    settings.github = MagicMock()
    settings.github.app_id = None
    return SlackRuntime(
        settings=settings,
        anthropic=build_fake_anthropic(make_fake_ma_handler()),
        sessionmaker=async_sessionmaker(),  # pyright: ignore[reportArgumentType]
        billing_config=None,
        http_client=MagicMock(spec=httpx.AsyncClient),
        resolver_cache=MagicMock(),  # pyright: ignore[reportArgumentType]  # stub, turn path not exercised
        turn_deps=MagicMock(),  # pyright: ignore[reportArgumentType]  # stub, turn path not exercised
    )


def _meta(
    *,
    team_id: str = _TEAM_ID,
    agent_name: str = _AGENT_NAME,
    active_section: str = "agent",
) -> str:
    return json.dumps(
        {
            "team_id": team_id,
            "channel_id": _CHANNEL_ID,
            "agent_name": agent_name,
            "active_section": active_section,
        },
        separators=(",", ":"),
    )


def _payload(
    *,
    callback_id: str,
    values: dict[str, Any],
    team_id: str = _TEAM_ID,
    user_id: str = _USER_ID,
    agent_name: str = _AGENT_NAME,
) -> dict[str, Any]:
    """Build a minimal view_submission payload with the given state values."""
    return {
        "user": {"id": user_id},
        "view": {
            "callback_id": callback_id,
            "private_metadata": _meta(team_id=team_id, agent_name=agent_name),
            "state": {"values": values},
        },
    }


def _input_value(block_id: str, action_id: str, value: str) -> dict[str, Any]:
    """Build a minimal state.values entry for a plain_text_input."""
    return {block_id: {action_id: {"type": "plain_text_input", "value": value}}}


# ---------------------------------------------------------------------------
# Pure evaluator tests — evaluate_new_agent_submission
# ---------------------------------------------------------------------------


def test_evaluate_new_agent_submission_when_name_invalid_returns_errors_keyed_new_agent_name() -> (
    None
):
    values = _input_value("new_agent__name", "new_agent__name", "bad name!")  # spaces + bang
    payload = _payload(callback_id="agent_setup__new_agent", values=values)

    decision = evaluate_new_agent_submission(payload)

    assert isinstance(decision, SubmitDecision), "should return SubmitDecision"
    assert decision.proceed is False, "invalid name should not proceed"
    assert decision.response_payload.get("response_action") == "errors", (
        "should return response_action: errors"
    )
    errors: dict[str, str] = decision.response_payload.get("errors", {})
    assert "new_agent__name" in errors, (
        "error must be keyed to new_agent__name (the input block_id)"
    )


def test_evaluate_new_agent_submission_when_name_valid_returns_clear_and_proceed() -> None:
    values = _input_value("new_agent__name", "new_agent__name", "my-agent")
    payload = _payload(callback_id="agent_setup__new_agent", values=values)

    decision = evaluate_new_agent_submission(payload)

    assert decision.proceed is True, "valid name should proceed"
    assert decision.response_payload.get("response_action") == "clear", (
        "successful new-agent submit should clear (pop to L1)"
    )
    assert decision.extra.get("name") == "my-agent", "name should be carried to extra"


def test_evaluate_new_agent_submission_when_model_invalid_returns_errors_keyed_new_agent_model() -> (
    None
):
    values = {
        **_input_value("new_agent__name", "new_agent__name", "valid-name"),
        **_input_value("new_agent__model", "new_agent__model", "gpt-4-turbo"),
    }
    payload = _payload(callback_id="agent_setup__new_agent", values=values)

    decision = evaluate_new_agent_submission(payload)

    assert decision.proceed is False, "unknown model should not proceed"
    errors = decision.response_payload.get("errors", {})
    assert "new_agent__model" in errors, (
        "error must be keyed to new_agent__model (the input block_id)"
    )


# ---------------------------------------------------------------------------
# Pure evaluator tests — evaluate_fork_agent_submission
# ---------------------------------------------------------------------------


def test_evaluate_fork_agent_submission_when_new_name_invalid_returns_errors_keyed_fork_agent_name() -> (
    None
):
    values = _input_value("fork_agent__name", "fork_agent__name", "bad name!")
    payload = _payload(callback_id="agent_setup__fork_agent", values=values)

    decision = evaluate_fork_agent_submission(payload)

    assert decision.proceed is False, "invalid fork name should not proceed"
    errors = decision.response_payload.get("errors", {})
    assert "fork_agent__name" in errors, "error must be keyed to fork_agent__name"


def test_evaluate_fork_agent_submission_when_name_valid_returns_proceed() -> None:
    values = _input_value("fork_agent__name", "fork_agent__name", "my-fork")
    payload = _payload(callback_id="agent_setup__fork_agent", values=values)

    decision = evaluate_fork_agent_submission(payload)

    assert decision.proceed is True, "valid fork name should proceed"
    assert decision.extra.get("new_name") == "my-fork", "new_name should be in extra"
    assert decision.extra.get("source_name") == _AGENT_NAME, (
        "source_name should come from private_metadata agent_name"
    )


# ---------------------------------------------------------------------------
# Pure evaluator tests — evaluate_edit_agent_submission
# ---------------------------------------------------------------------------


def test_evaluate_edit_agent_submission_when_model_invalid_returns_errors_keyed_edit_agent_model() -> (
    None
):
    values = _input_value("edit_agent__model", "edit_agent__model", "gpt-4-turbo")
    payload = _payload(callback_id="agent_setup__edit_agent", values=values)

    decision = evaluate_edit_agent_submission(payload)

    assert decision.proceed is False, "unknown model should not proceed"
    errors = decision.response_payload.get("errors", {})
    assert "edit_agent__model" in errors, "error must be keyed to edit_agent__model"


def test_evaluate_edit_agent_submission_when_model_blank_returns_proceed() -> None:
    values = _input_value("edit_agent__model", "edit_agent__model", "")
    payload = _payload(callback_id="agent_setup__edit_agent", values=values)

    decision = evaluate_edit_agent_submission(payload)

    assert decision.proceed is True, "blank model (keep current) should proceed"


# ---------------------------------------------------------------------------
# Pure evaluator tests — evaluate_edit_repo_submission
# ---------------------------------------------------------------------------


def test_evaluate_edit_repo_submission_when_pat_blank_proceeds_with_keep_flag() -> None:
    """Blank PAT = keep stored token: proceed=True, pat_replace=False."""
    values = {
        **_input_value("edit_repo__url", "edit_repo__url", "https://github.com/org/repo"),
        **_input_value("edit_repo__pat", "edit_repo__pat", ""),
    }
    payload = _payload(callback_id="agent_setup__edit_repo", values=values)

    decision = evaluate_edit_repo_submission(payload)

    assert decision.proceed is True, "blank PAT should proceed (empty=keep)"
    assert decision.extra.get("pat_replace") is False, (
        "blank PAT must not set pat_replace (never overwrite stored token on blank)"
    )
    assert decision.extra.get("pat") is None, "blank PAT should produce None in extra"


def test_evaluate_edit_repo_submission_when_pat_provided_sets_replace_flag() -> None:
    values = {
        **_input_value("edit_repo__url", "edit_repo__url", "https://github.com/org/repo"),
        **_input_value("edit_repo__pat", "edit_repo__pat", "ghp_test1234"),
    }
    payload = _payload(callback_id="agent_setup__edit_repo", values=values)

    decision = evaluate_edit_repo_submission(payload)

    assert decision.proceed is True, "valid PAT should proceed"
    assert decision.extra.get("pat_replace") is True, "non-blank PAT should set pat_replace"


# ---------------------------------------------------------------------------
# Pure evaluator tests — evaluate_paste_secrets_submission
# ---------------------------------------------------------------------------


def test_evaluate_paste_secrets_when_key_invalid_returns_errors_keyed_paste_secrets_content() -> (
    None
):
    content = "123_STARTS_WITH_DIGIT=value"  # invalid: starts with digit
    values = _input_value("paste_secrets__content", "paste_secrets__content", content)
    payload = _payload(callback_id="agent_setup__paste_secrets", values=values)

    decision = evaluate_paste_secrets_submission(payload)

    assert decision.proceed is False, "invalid key name should not proceed"
    errors = decision.response_payload.get("errors", {})
    assert "paste_secrets__content" in errors, (
        "error must be keyed to paste_secrets__content (the input block_id)"
    )


def test_evaluate_paste_secrets_when_count_exceeds_cap_returns_cap_error() -> None:
    # Build _SECRET_CAP + 1 keys
    lines = "\n".join(f"KEY_{i}=value_{i}" for i in range(_SECRET_CAP + 1))
    values = _input_value("paste_secrets__content", "paste_secrets__content", lines)
    payload = _payload(callback_id="agent_setup__paste_secrets", values=values)

    decision = evaluate_paste_secrets_submission(payload)

    assert decision.proceed is False, f">{_SECRET_CAP} secrets should not proceed"
    errors = decision.response_payload.get("errors", {})
    assert "paste_secrets__content" in errors, "cap error must be keyed to paste_secrets__content"
    assert str(_SECRET_CAP) in errors["paste_secrets__content"], (
        "error text should mention the cap limit"
    )


def test_evaluate_paste_secrets_when_value_oversized_returns_byte_cap_error() -> None:
    from daimon.adapters.slack.agent_setup.submit import _MAX_SECRET_VALUE_BYTES

    oversized_value = "x" * (_MAX_SECRET_VALUE_BYTES + 1)
    content = f"MY_KEY={oversized_value}"
    values = _input_value("paste_secrets__content", "paste_secrets__content", content)
    payload = _payload(callback_id="agent_setup__paste_secrets", values=values)

    decision = evaluate_paste_secrets_submission(payload)

    assert decision.proceed is False, "oversized value should not proceed"
    errors = decision.response_payload.get("errors", {})
    assert "paste_secrets__content" in errors, (
        "byte-cap error must be keyed to paste_secrets__content"
    )
    # CRITICAL: the error message must reference the KEY name, not the value.
    error_text = errors["paste_secrets__content"]
    assert "MY_KEY" in error_text, "error text should name the offending key"
    assert oversized_value not in error_text, "secret VALUE must never appear in the error message"


def test_evaluate_paste_secrets_when_valid_response_payload_does_not_contain_values() -> None:
    """Serialized response_payload must not contain any secret value."""
    secret_value = "s3cr3t_val_that_should_not_leak"
    content = f"API_KEY={secret_value}"
    values = _input_value("paste_secrets__content", "paste_secrets__content", content)
    payload = _payload(callback_id="agent_setup__paste_secrets", values=values)

    decision = evaluate_paste_secrets_submission(payload)

    assert decision.proceed is True, "valid secret should proceed"
    # Serialize the response_payload and assert the value is absent.
    serialized = json.dumps(decision.response_payload)
    assert secret_value not in serialized, (
        "secret VALUE must never appear in the response_action payload"
    )


def test_evaluate_paste_secrets_when_valid_extra_carries_pairs() -> None:
    content = "FOO=bar\nBAZ=qux"
    values = _input_value("paste_secrets__content", "paste_secrets__content", content)
    payload = _payload(callback_id="agent_setup__paste_secrets", values=values)

    decision = evaluate_paste_secrets_submission(payload)

    assert decision.proceed is True, "valid secrets should proceed"
    pairs: list[tuple[str, str]] = decision.extra.get("pairs", [])
    assert len(pairs) == 2, "should parse 2 key-value pairs"
    assert ("FOO", "bar") in pairs, "FOO=bar should be in parsed pairs"
    assert ("BAZ", "qux") in pairs, "BAZ=qux should be in parsed pairs"


# ---------------------------------------------------------------------------
# run_* handler tests via FakeSlackWebClient
# ---------------------------------------------------------------------------
# These tests require DAIMON_DATABASE__TEST_URL (real Postgres) for the DB write.
# ---------------------------------------------------------------------------


def _iso_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _agent_payload_for_gate_tests(
    *, tenant_id: Any, agent_id: str, agent_name: str = _AGENT_NAME
) -> dict[str, object]:
    """Build a minimal MA agent payload tagged for this tenant/name."""
    from daimon.core.defaults.metadata import MA_METADATA_KEY_NAME, MA_METADATA_KEY_TENANT

    now = _iso_now()
    return {
        "id": agent_id,
        "type": "agent",
        "name": agent_name,
        "version": 1,
        "model": {"id": "claude-sonnet-4-6", "speed": "standard"},
        "system": None,
        "metadata": {
            MA_METADATA_KEY_TENANT: str(tenant_id),
            MA_METADATA_KEY_NAME: agent_name,
        },
        "mcp_servers": [],
        "tools": [],
        "skills": [],
        "created_at": now,
        "updated_at": now,
        "archived_at": None,
        "description": None,
    }


def _make_recording_ma_handler_with_agents_and_update(
    agents: list[dict[str, object]],
    calls: list[tuple[str, str]],
) -> Any:
    """Like a PATCH-capable fake MA handler, but records every (method, path)
    it serves so a refusal test can assert zero MA traffic occurred."""
    agent_store: dict[str, dict[str, object]] = {
        str(ag["id"]): ag
        for ag in agents  # type: ignore[index]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        calls.append((method, path))

        if method == "GET" and path == "/v1/agents":
            return httpx.Response(200, json={"data": list(agent_store.values()), "has_more": False})
        m = re.match(r"^/v1/agents/(?P<id>[^/]+)$", path)
        if m and method == "GET":
            agent_id_req = m.group("id")
            if agent_id_req in agent_store:
                return httpx.Response(200, json=agent_store[agent_id_req])
            return httpx.Response(
                404,
                json={
                    "type": "error",
                    "error": {"type": "not_found_error", "message": "not found"},
                },
            )
        if m and method in {"PATCH", "POST"}:
            agent_id_req = m.group("id")
            if agent_id_req not in agent_store:
                return httpx.Response(
                    404,
                    json={
                        "type": "error",
                        "error": {"type": "not_found_error", "message": "not found"},
                    },
                )
            body: dict[str, Any] = json.loads(request.content)
            existing = agent_store[agent_id_req]
            merged: dict[str, object] = {**existing, **body}
            merged["version"] = int(existing.get("version", 1)) + 1  # type: ignore[arg-type]
            agent_store[agent_id_req] = merged
            return httpx.Response(200, json=merged)
        if method == "GET" and path == "/v1/environments":
            return httpx.Response(200, json={"data": [], "has_more": False})

        return httpx.Response(404, json={"error": f"unhandled {method} {path}"})

    return handler


def _build_runtime_with_db(
    db_factory: async_sessionmaker[AsyncSession],
    *,
    fernet_key: str = "dummy",
    anthropic_handler: Any = None,
) -> SlackRuntime:
    """Build a SlackRuntime with a real DB factory and a fake MA transport."""
    handler = anthropic_handler or make_fake_ma_handler()
    settings: MagicMock = MagicMock()
    settings.crypto.keys = (SecretStr(fernet_key),)
    settings.mcp.public_url = None
    settings.mcp.jwt_secret = None
    settings.github = MagicMock()
    settings.github.app_id = None
    settings.slack.dev_allow_all_admin = False
    return SlackRuntime(
        settings=settings,
        anthropic=build_fake_anthropic(handler),
        sessionmaker=db_factory,
        billing_config=None,
        http_client=MagicMock(spec=httpx.AsyncClient),
        resolver_cache=MagicMock(),  # pyright: ignore[reportArgumentType]  # stub, turn path not exercised
        turn_deps=MagicMock(),  # pyright: ignore[reportArgumentType]  # stub, turn path not exercised
    )


async def _mark_reachable(
    db_session_factory: async_sessionmaker[AsyncSession],
    *,
    tenant_id: Any,
    agent_name: str,
) -> None:
    """Write a real tenant-scope propagation row so the named agent is
    currently reachable — never patch the reachability predicate."""
    async with db_session_factory() as session, session.begin():
        await set_fields(
            session,
            scope=TenantScopeRef(tenant_id=tenant_id),
            tenant_id=tenant_id,
            agent_name=agent_name,
            mode="agent",
        )


def _ephemeral_texts(client_fake: Any) -> list[str]:
    ephemeral_key = ("POST", yarl.URL("https://slack.com/api/chat.postEphemeral"))
    return [
        call.kwargs["json"]["text"] for call in client_fake.mock.requests.get(ephemeral_key, [])
    ]


# ---------------------------------------------------------------------------
# Task 1 — the four always-open paths: non-admin succeeds, no refusal ephemeral
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_new_agent_submission_when_non_admin_creates_agent_with_no_refusal(
    fake_slack_web_client: Any,
) -> None:
    """Creation is open to every workspace member — no admin re-check remains."""
    client_fake: Any = fake_slack_web_client
    # conftest default users.info is non-admin; no override needed.
    runtime = _build_runtime_no_db()

    await run_new_agent_submission(
        runtime,
        client_fake.client,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        view_id="V_SUBMIT_TEST",
        extra={"name": "member-created-agent", "model": "claude-sonnet-4-6", "system": None},
    )

    texts = _ephemeral_texts(client_fake)
    assert any(":white_check_mark:" in t for t in texts), (
        "a non-admin's new-agent submission must succeed"
    )
    assert not any("permission" in t for t in texts), (
        "creation must not post a permission-refusal ephemeral"
    )


@pytest.mark.asyncio
async def test_run_new_agent_submission_when_duplicate_name_still_refused(
    fake_slack_web_client: Any,
) -> None:
    """A name collision is still refused by create_blank_agent's own guard,
    independent of the now-removed admin check."""
    client_fake: Any = fake_slack_web_client
    runtime = _build_runtime_no_db()

    extra = {"name": "collide-agent", "model": "claude-sonnet-4-6", "system": None}
    await run_new_agent_submission(
        runtime,
        client_fake.client,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        view_id="V1",
        extra=extra,
    )
    await run_new_agent_submission(
        runtime,
        client_fake.client,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        view_id="V2",
        extra=extra,
    )

    texts = _ephemeral_texts(client_fake)
    assert any(":white_check_mark:" in t for t in texts), "the first create should succeed"
    assert any("Failed to create agent" in t for t in texts), (
        "the second, colliding create must still be refused by the collision guard"
    )


@pytest.mark.asyncio
async def test_run_fork_agent_submission_when_non_admin_forks_agent_with_no_refusal(
    fake_slack_web_client: Any,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Forking is open to every workspace member — no admin re-check remains."""
    from cryptography.fernet import Fernet

    client_fake: Any = fake_slack_web_client
    runtime = _build_runtime_with_db(db_session_factory, fernet_key=Fernet.generate_key().decode())

    await run_new_agent_submission(
        runtime,
        client_fake.client,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        view_id="V1",
        extra={"name": _AGENT_NAME, "model": "claude-sonnet-4-6", "system": None},
    )
    await run_fork_agent_submission(
        runtime,
        client_fake.client,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        view_id="V2",
        extra={"source_name": _AGENT_NAME, "new_name": "forked-agent"},
    )

    texts = _ephemeral_texts(client_fake)
    assert any("Forked" in t for t in texts), "a non-admin's fork submission must succeed"
    assert not any("permission" in t for t in texts), (
        "fork must not post a permission-refusal ephemeral"
    )


@pytest.mark.asyncio
async def test_run_edit_repo_submission_when_non_admin_and_agent_is_workspace_default_binds_repo(
    fake_slack_web_client: Any,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Repo binding is a per-agent attachment, so it stays open for a
    non-admin even against the workspace's currently-default agent."""
    from daimon.core.ma_identity import derive_agent_uuid, derive_tenant_uuid
    from daimon.core.stores.agent_repo_binding import get_binding

    tenant_id = derive_tenant_uuid(platform="slack", workspace_id=_TEAM_ID)
    ma_agent_id = f"agent_{'f' * 24}"
    agent_uuid = derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=ma_agent_id)

    async with db_session_factory() as session:
        await make_tenant(session, platform="slack", workspace_id=_TEAM_ID, id=tenant_id)
        await session.commit()
    await _mark_reachable(db_session_factory, tenant_id=tenant_id, agent_name=_AGENT_NAME)

    client_fake: Any = fake_slack_web_client
    # conftest default users.info is non-admin; no override.

    runtime = SlackRuntime(
        settings=_build_edit_repo_settings(app_id=None),
        anthropic=build_fake_anthropic(
            _build_edit_repo_ma_handler(tenant_id=tenant_id, ma_agent_id=ma_agent_id)
        ),
        sessionmaker=db_session_factory,
        billing_config=None,
        http_client=MagicMock(spec=httpx.AsyncClient),
        resolver_cache=MagicMock(),  # pyright: ignore[reportArgumentType]  # stub, turn path not exercised
        turn_deps=MagicMock(),  # pyright: ignore[reportArgumentType]  # stub, turn path not exercised
    )

    await run_edit_repo_submission(
        runtime,
        client_fake.client,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        view_id="V_SUBMIT_TEST",
        agent_name=_AGENT_NAME,
        parent_section="repo",
        extra={
            "repo_url": "https://github.com/example/default-agent.git",
            "pat": "",
            "pat_replace": False,
        },
    )

    async with db_session_factory() as session:
        row = await get_binding(session, tenant_id=tenant_id, agent_id=agent_uuid)

    assert row is not None, "binding should be written even against a workspace-default agent"
    assert row.repo_url == "example/default-agent"

    texts = _ephemeral_texts(client_fake)
    assert any(":white_check_mark:" in t for t in texts), (
        "repo binding must succeed for a non-admin on a workspace-default agent"
    )
    assert not any("workspace-admin" in t for t in texts), (
        "no reachability refusal ephemeral should be posted for a per-agent attachment"
    )


@pytest.mark.asyncio
async def test_run_paste_secrets_submission_when_non_admin_and_agent_is_workspace_default_writes_secrets(
    fake_slack_web_client: Any,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Env variables are a per-agent attachment, so they stay open for a
    non-admin even against the workspace's currently-default agent."""
    async with db_session_factory() as session:
        tenant = await make_tenant(session, platform="slack", workspace_id=_TEAM_ID)
        tenant_id = tenant.id
        await session.commit()
    await _mark_reachable(db_session_factory, tenant_id=tenant_id, agent_name=_AGENT_NAME)

    client_fake: Any = fake_slack_web_client
    # conftest default users.info is non-admin; no override.

    ma_agent_id = f"agent_{'g' * 24}"
    agent_data = _agent_payload_for_gate_tests(tenant_id=tenant_id, agent_id=ma_agent_id)

    def _handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        if method == "GET" and path == "/v1/agents":
            return httpx.Response(200, json={"data": [agent_data], "has_more": False})
        if method == "GET" and path == "/v1/environments":
            return httpx.Response(200, json={"data": [], "has_more": False})
        return httpx.Response(404, json={"error": f"unhandled {method} {path}"})

    runtime = _build_runtime_with_db(db_session_factory, anthropic_handler=_handler)

    await run_paste_secrets_submission(
        runtime,
        client_fake.client,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        view_id="V_SUBMIT_TEST",
        agent_name=_AGENT_NAME,
        parent_section="secrets",
        extra={"pairs": [("OPEN_KEY", "open_val")]},
    )

    texts = _ephemeral_texts(client_fake)
    assert any(":white_check_mark:" in t for t in texts), (
        "paste-secrets must succeed for a non-admin on a workspace-default agent"
    )
    assert not any("workspace-admin" in t for t in texts), (
        "no reachability refusal ephemeral should be posted for a per-agent attachment"
    )


# ---------------------------------------------------------------------------
# Task 2 — the three field-conditional paths: reachability-gated for non-admin
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_edit_agent_submission_when_non_admin_and_unreachable_proceeds(
    fake_slack_web_client: Any,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Editing model/prompt is open when nobody has scoped this agent."""
    from daimon.core.ma_identity import derive_tenant_uuid

    tenant_id = derive_tenant_uuid(platform="slack", workspace_id=_TEAM_ID)
    ma_agent_id = f"agent_{'h' * 24}"
    calls: list[tuple[str, str]] = []
    agent_payload = _agent_payload_for_gate_tests(tenant_id=tenant_id, agent_id=ma_agent_id)
    handler = _make_recording_ma_handler_with_agents_and_update([agent_payload], calls)
    runtime = _build_runtime_with_db(db_session_factory, anthropic_handler=handler)

    client_fake: Any = fake_slack_web_client
    # conftest default users.info is non-admin; no override.

    await run_edit_agent_submission(
        runtime,
        client_fake.client,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        view_id="V_SUBMIT_TEST",
        agent_name=_AGENT_NAME,
        parent_section="agent",
        extra={"model": None, "system": "You are helpful."},
    )

    assert any(method in {"PATCH", "POST"} for method, path in calls if "/v1/agents/" in path), (
        "an unreachable agent's edit must reach the MA update"
    )

    texts = _ephemeral_texts(client_fake)
    assert any(":white_check_mark:" in t for t in texts), (
        "edit-agent must succeed for a non-admin on an unreachable agent"
    )


@pytest.mark.asyncio
async def test_run_edit_agent_submission_when_non_admin_and_reachable_refused(
    fake_slack_web_client: Any,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A non-admin editing a currently-default agent must be refused before
    any MA request."""
    async with db_session_factory() as session:
        tenant = await make_tenant(session, platform="slack", workspace_id=_TEAM_ID)
        tenant_id = tenant.id
        await session.commit()
    await _mark_reachable(db_session_factory, tenant_id=tenant_id, agent_name=_AGENT_NAME)

    calls: list[tuple[str, str]] = []
    ma_agent_id = f"agent_{'i' * 24}"
    agent_payload = _agent_payload_for_gate_tests(tenant_id=tenant_id, agent_id=ma_agent_id)
    handler = _make_recording_ma_handler_with_agents_and_update([agent_payload], calls)
    runtime = _build_runtime_with_db(db_session_factory, anthropic_handler=handler)

    client_fake: Any = fake_slack_web_client

    await run_edit_agent_submission(
        runtime,
        client_fake.client,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        view_id="V_SUBMIT_TEST",
        agent_name=_AGENT_NAME,
        parent_section="agent",
        extra={"model": None, "system": "Attempted change."},
    )

    writes = [c for c in calls if c[0] != "GET"]
    assert writes == [], "a refused edit must never write to the MA"

    texts = _ephemeral_texts(client_fake)
    assert len(texts) == 1, "exactly one ephemeral should be posted on refusal"
    assert "workspace-admin" in texts[0], (
        "the refusal ephemeral should name why the write was refused"
    )


@pytest.mark.asyncio
async def test_run_add_skill_submission_when_non_admin_and_unreachable_proceeds(
    fake_slack_web_client: Any,
    db_session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adding a skill is open when nobody has scoped this agent."""
    calls: list[str] = []

    async def fake_kickoff(
        runtime: Any, *, tenant_id: Any, account_id: Any, agent_name: str, repo_url: str
    ) -> SyncReport:
        calls.append(repo_url)
        return SyncReport(synced=1)

    monkeypatch.setattr(submit_mod, "kick_off_skill_sync", fake_kickoff)

    runtime = _build_runtime_with_db(db_session_factory)
    client_fake: Any = fake_slack_web_client
    # conftest default users.info is non-admin; no override.

    await run_add_skill_submission(
        runtime,
        client_fake.client,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        view_id="V_SUBMIT_TEST",
        agent_name=_AGENT_NAME,
        parent_section="skills",
        extra={"repo_url": "https://github.com/example/skills.git", "branch": "main"},
    )

    assert calls == ["https://github.com/example/skills.git"], (
        "an unreachable agent's add-skill must reach kick_off_skill_sync"
    )

    texts = _ephemeral_texts(client_fake)
    assert any(":white_check_mark:" in t for t in texts), (
        "add-skill must succeed for a non-admin on an unreachable agent"
    )


@pytest.mark.asyncio
async def test_run_add_skill_submission_when_non_admin_and_reachable_refused_queues_no_sync(
    fake_slack_web_client: Any,
    db_session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-admin adding a skill to a currently-default agent must be
    refused before the sync is queued."""

    async def unexpected_kickoff(*args: Any, **kwargs: Any) -> SyncReport:
        raise AssertionError("skill sync must not be kicked off when the write is refused")

    monkeypatch.setattr(submit_mod, "kick_off_skill_sync", unexpected_kickoff)

    async with db_session_factory() as session:
        tenant = await make_tenant(session, platform="slack", workspace_id=_TEAM_ID)
        tenant_id = tenant.id
        await session.commit()
    await _mark_reachable(db_session_factory, tenant_id=tenant_id, agent_name=_AGENT_NAME)

    runtime = _build_runtime_with_db(db_session_factory)
    client_fake: Any = fake_slack_web_client

    await run_add_skill_submission(
        runtime,
        client_fake.client,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        view_id="V_SUBMIT_TEST",
        agent_name=_AGENT_NAME,
        parent_section="skills",
        extra={"repo_url": "https://github.com/example/skills.git", "branch": "main"},
    )

    texts = _ephemeral_texts(client_fake)
    assert len(texts) == 1, "exactly one ephemeral should be posted on refusal, no sync queued"
    assert "workspace-admin" in texts[0], (
        "the refusal ephemeral should name why the write was refused"
    )


@pytest.mark.asyncio
async def test_run_add_mcp_submission_when_non_admin_and_unreachable_proceeds(
    fake_slack_web_client: Any,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Adding an MCP server is open when nobody has scoped this agent."""
    from daimon.core.ma_identity import derive_tenant_uuid

    tenant_id = derive_tenant_uuid(platform="slack", workspace_id=_TEAM_ID)
    ma_agent_id = f"agent_{'j' * 24}"
    calls: list[tuple[str, str]] = []
    agent_payload = _agent_payload_for_gate_tests(tenant_id=tenant_id, agent_id=ma_agent_id)
    handler = _make_recording_ma_handler_with_agents_and_update([agent_payload], calls)
    runtime = _build_runtime_with_db(db_session_factory, anthropic_handler=handler)

    client_fake: Any = fake_slack_web_client
    # conftest default users.info is non-admin; no override.

    await run_add_mcp_submission(
        runtime,
        client_fake.client,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        view_id="V_SUBMIT_TEST",
        agent_name=_AGENT_NAME,
        parent_section="mcps",
        extra={
            "mcp_name": "an-mcp",
            "mcp_url": "https://mcp.example.com",
            "token": None,
            "token_replace": False,
        },
    )

    assert any(method in {"PATCH", "POST"} for method, path in calls if "/v1/agents/" in path), (
        "an unreachable agent's add-mcp must reach the MA update"
    )

    texts = _ephemeral_texts(client_fake)
    assert any(":white_check_mark:" in t for t in texts), (
        "add-mcp must succeed for a non-admin on an unreachable agent"
    )


@pytest.mark.asyncio
async def test_run_add_mcp_submission_when_non_admin_and_reachable_refused(
    fake_slack_web_client: Any,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A non-admin adding an MCP server to a currently-default agent must be
    refused before any MA request."""
    async with db_session_factory() as session:
        tenant = await make_tenant(session, platform="slack", workspace_id=_TEAM_ID)
        tenant_id = tenant.id
        await session.commit()
    await _mark_reachable(db_session_factory, tenant_id=tenant_id, agent_name=_AGENT_NAME)

    ma_agent_id = f"agent_{'k' * 24}"
    calls: list[tuple[str, str]] = []
    agent_payload = _agent_payload_for_gate_tests(tenant_id=tenant_id, agent_id=ma_agent_id)
    handler = _make_recording_ma_handler_with_agents_and_update([agent_payload], calls)
    runtime = _build_runtime_with_db(db_session_factory, anthropic_handler=handler)

    client_fake: Any = fake_slack_web_client

    await run_add_mcp_submission(
        runtime,
        client_fake.client,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        view_id="V_SUBMIT_TEST",
        agent_name=_AGENT_NAME,
        parent_section="mcps",
        extra={
            "mcp_name": "an-mcp",
            "mcp_url": "https://mcp.example.com",
            "token": None,
            "token_replace": False,
        },
    )

    writes = [c for c in calls if c[0] != "GET"]
    assert writes == [], "a refused add-mcp must never write to the MA"

    texts = _ephemeral_texts(client_fake)
    assert len(texts) == 1, "exactly one ephemeral should be posted on refusal"
    assert "workspace-admin" in texts[0], (
        "the refusal ephemeral should name why the write was refused"
    )


@pytest.mark.asyncio
async def test_run_add_mcp_submission_when_admin_and_reachable_proceeds(
    fake_slack_web_client: Any,
    db_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """An admin caller is unaffected by reachability on a field-conditional branch."""
    async with db_session_factory() as session:
        tenant = await make_tenant(session, platform="slack", workspace_id=_TEAM_ID)
        tenant_id = tenant.id
        await session.commit()
    await _mark_reachable(db_session_factory, tenant_id=tenant_id, agent_name=_AGENT_NAME)

    ma_agent_id = f"agent_{'l' * 24}"
    calls: list[tuple[str, str]] = []
    agent_payload = _agent_payload_for_gate_tests(tenant_id=tenant_id, agent_id=ma_agent_id)
    handler = _make_recording_ma_handler_with_agents_and_update([agent_payload], calls)
    runtime = _build_runtime_with_db(db_session_factory, anthropic_handler=handler)

    client_fake: Any = fake_slack_web_client
    _override_users_info_admin(client_fake.mock)

    await run_add_mcp_submission(
        runtime,
        client_fake.client,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        view_id="V_SUBMIT_TEST",
        agent_name=_AGENT_NAME,
        parent_section="mcps",
        extra={
            "mcp_name": "an-mcp",
            "mcp_url": "https://mcp.example.com",
            "token": None,
            "token_replace": False,
        },
    )

    assert any(method in {"PATCH", "POST"} for method, path in calls if "/v1/agents/" in path), (
        "an admin's add-mcp must still reach the MA update even on a reachable agent"
    )

    texts = _ephemeral_texts(client_fake)
    assert any(":white_check_mark:" in t for t in texts), (
        "add-mcp must succeed for an admin regardless of reachability"
    )


# ---------------------------------------------------------------------------
# CR-02 regression: success ephemerals and NO views_update on cleared view
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_new_agent_submission_when_admin_and_write_succeeds_posts_success_ephemeral_and_no_views_update(
    fake_slack_web_client: Any,
) -> None:
    """Admin create succeeds → :white_check_mark: ephemeral posted; views_update NOT called.

    CR-02 fix: _refresh_l1 was removed; views_update on the cleared L3 view_id
    would return not_found and produce a spurious :x: failure. The fix posts a
    :white_check_mark: chat_postEphemeral instead.
    """
    client_fake: Any = fake_slack_web_client
    _override_users_info_admin(client_fake.mock)

    runtime = _build_runtime_no_db()

    await run_new_agent_submission(
        runtime,
        client_fake.client,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        view_id="V_SUBMIT_TEST",
        extra={"name": "fresh-agent", "model": "claude-sonnet-4-6", "system": None},
    )

    ephemeral_key = ("POST", yarl.URL("https://slack.com/api/chat.postEphemeral"))
    views_update_key = ("POST", yarl.URL("https://slack.com/api/views.update"))

    assert ephemeral_key in client_fake.mock.requests, (
        "successful create should post a chat_postEphemeral"
    )
    ephemeral_calls: list[Any] = client_fake.mock.requests[ephemeral_key]
    assert len(ephemeral_calls) == 1, "exactly one ephemeral should be posted"

    # The Slack SDK sends chat.postEphemeral as JSON (kwargs["json"]).
    ephemeral_text: str = ephemeral_calls[0].kwargs["json"]["text"]
    assert ":white_check_mark:" in ephemeral_text, (
        "success ephemeral text must contain :white_check_mark:"
    )

    assert views_update_key not in client_fake.mock.requests, (
        "run_new_agent_submission must NOT call views_update on the cleared L3 view (CR-02)"
    )


@pytest.mark.asyncio
async def test_run_paste_secrets_submission_when_admin_and_two_pairs_posts_count_ephemeral_without_values_and_no_views_update(
    fake_slack_web_client: Any,
    db_session_factory: Any,
) -> None:
    """Admin paste-secrets (2 pairs) → count confirmation ephemeral without secret values; no views_update.

    Threat T-83-22: the confirmation references only key names/count — never pair values.
    CR-02 fix: no views_update call on the cleared L3 view.
    """
    from cryptography.fernet import Fernet
    from daimon.adapters.slack.runtime import SlackRuntime
    from daimon.testing.factories import make_tenant

    # We need a Tenant row so put_agent_file's FK resolves. Seed it via a
    # one-shot session from the factory (per-test schema isolation is active).
    async with db_session_factory() as session:
        tenant = await make_tenant(session, platform="slack", workspace_id=_TEAM_ID)
        tenant_id = tenant.id
        await session.commit()

    client_fake: Any = fake_slack_web_client
    _override_users_info_admin(client_fake.mock)

    fernet_key = Fernet.generate_key().decode()
    settings: MagicMock = MagicMock()
    settings.crypto.keys = (SecretStr(fernet_key),)
    settings.mcp.public_url = None
    settings.mcp.jwt_secret = None
    settings.github = MagicMock()
    settings.github.app_id = None

    # MA handler: the agent must exist so run_paste_secrets can find it via find_agent_by_daimon_tag.
    from datetime import UTC, datetime

    from daimon.core.defaults.metadata import MA_METADATA_KEY_NAME, MA_METADATA_KEY_TENANT

    _ma_agent_id = f"agent_{'b' * 24}"
    now = datetime.now(UTC).isoformat()
    _agent_data: dict[str, object] = {
        "id": _ma_agent_id,
        "type": "agent",
        "name": _AGENT_NAME,
        "version": 1,
        "model": {"id": "claude-sonnet-4-6", "speed": "standard"},
        "system": None,
        "metadata": {
            MA_METADATA_KEY_TENANT: str(tenant_id),
            MA_METADATA_KEY_NAME: _AGENT_NAME,
        },
        "mcp_servers": [],
        "tools": [],
        "skills": [],
        "created_at": now,
        "updated_at": now,
        "archived_at": None,
        "description": None,
    }

    import httpx

    def _handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        if method == "GET" and path == "/v1/agents":
            return httpx.Response(200, json={"data": [_agent_data], "has_more": False})
        if method == "GET" and path == "/v1/environments":
            return httpx.Response(200, json={"data": [], "has_more": False})
        return httpx.Response(404, json={"error": f"unhandled {method} {path}"})

    runtime = SlackRuntime(
        settings=settings,
        anthropic=build_fake_anthropic(_handler),
        sessionmaker=db_session_factory,
        billing_config=None,
        http_client=MagicMock(spec=httpx.AsyncClient),
        resolver_cache=MagicMock(),  # pyright: ignore[reportArgumentType]  # stub, turn path not exercised
        turn_deps=MagicMock(),  # pyright: ignore[reportArgumentType]  # stub, turn path not exercised
    )

    secret_val_1 = "s3cr3t_one"
    secret_val_2 = "s3cr3t_two"

    await run_paste_secrets_submission(
        runtime,
        client_fake.client,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        view_id="V_SUBMIT_TEST",
        agent_name=_AGENT_NAME,
        parent_section="secrets",
        extra={"pairs": [("KEY_ONE", secret_val_1), ("KEY_TWO", secret_val_2)]},
    )

    ephemeral_key = ("POST", yarl.URL("https://slack.com/api/chat.postEphemeral"))
    views_update_key = ("POST", yarl.URL("https://slack.com/api/views.update"))

    assert ephemeral_key in client_fake.mock.requests, (
        "successful paste-secrets should post a chat_postEphemeral"
    )
    ephemeral_calls = client_fake.mock.requests[ephemeral_key]
    assert len(ephemeral_calls) >= 1, "at least one ephemeral should be posted"

    # Find the success ephemeral (the Slack SDK sends JSON body; text is in kwargs["json"]["text"]).
    ephemeral_texts = [call.kwargs["json"]["text"] for call in ephemeral_calls]
    success_texts = [t for t in ephemeral_texts if ":white_check_mark:" in t]
    assert len(success_texts) >= 1, "success confirmation ephemeral must include :white_check_mark:"

    success_text = success_texts[0]
    assert "2" in success_text or "secrets" in success_text, (
        "success text for 2 pairs must reference the count (e.g. '2 secrets')"
    )

    # T-83-22: secret values must NOT appear in the confirmation text.
    assert secret_val_1 not in success_text, (
        f"secret value '{secret_val_1}' must not appear in the confirmation (T-83-22)"
    )
    assert secret_val_2 not in success_text, (
        f"secret value '{secret_val_2}' must not appear in the confirmation (T-83-22)"
    )

    assert views_update_key not in client_fake.mock.requests, (
        "run_paste_secrets_submission must NOT call views_update on the cleared L3 view (CR-02)"
    )


# ---------------------------------------------------------------------------
# run_edit_repo_submission preserves ma_secret_ref
# ---------------------------------------------------------------------------


def _build_edit_repo_ma_handler(*, tenant_id: Any, ma_agent_id: str) -> Any:
    """Build an httpx handler exposing a single agent for find_agent_by_daimon_tag."""
    import httpx
    from daimon.core.defaults.metadata import MA_METADATA_KEY_NAME, MA_METADATA_KEY_TENANT

    now = _iso_now()
    agent_data: dict[str, object] = {
        "id": ma_agent_id,
        "type": "agent",
        "name": _AGENT_NAME,
        "version": 1,
        "model": {"id": "claude-sonnet-4-6", "speed": "standard"},
        "system": None,
        "metadata": {
            MA_METADATA_KEY_TENANT: str(tenant_id),
            MA_METADATA_KEY_NAME: _AGENT_NAME,
        },
        "mcp_servers": [],
        "tools": [],
        "skills": [],
        "created_at": now,
        "updated_at": now,
        "archived_at": None,
        "description": None,
    }

    def _handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        if method == "GET" and path == "/v1/agents":
            return httpx.Response(200, json={"data": [agent_data], "has_more": False})
        if method == "GET" and path == "/v1/environments":
            return httpx.Response(200, json={"data": [], "has_more": False})
        return httpx.Response(404, json={"error": f"unhandled {method} {path}"})

    return _handler


def _iso_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


@pytest.mark.asyncio
async def test_run_edit_repo_submission_when_pat_replace_false_preserves_inline_pat(
    fake_slack_web_client: Any,
    db_session_factory: Any,
) -> None:
    """PAT-CLOBBER regression: blank PAT edit-repo preserves the stored ma_secret_ref.

    Store an inline PAT (ma_secret_ref=inline-pat:{agent_uuid}) directly on the
    binding, then submit edit-repo with a new URL + pat_replace=False (blank PAT).
    The binding's ma_secret_ref must still equal inline-pat:{agent_uuid} and
    repo_url must reflect the new URL.
    """
    from daimon.adapters.slack.runtime import SlackRuntime
    from daimon.core.ma_identity import derive_agent_uuid, derive_tenant_uuid
    from daimon.core.stores.agent_repo_binding import get_binding, set_binding
    from daimon.testing.factories import make_tenant

    tenant_id = derive_tenant_uuid(platform="slack", workspace_id=_TEAM_ID)
    ma_agent_id = f"agent_{'c' * 24}"
    agent_uuid = derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=ma_agent_id)

    async with db_session_factory() as session:
        await make_tenant(session, platform="slack", workspace_id=_TEAM_ID, id=tenant_id)
        await session.commit()

    async with db_session_factory() as session:
        await set_binding(
            session,
            tenant_id=tenant_id,
            agent_id=agent_uuid,
            repo_url="https://github.com/example/old.git",
            default_branch="main",
            ma_secret_ref=f"inline-pat:{agent_uuid}",
        )
        await session.commit()

    client_fake: Any = fake_slack_web_client
    _override_users_info_admin(client_fake.mock)

    settings: MagicMock = MagicMock()
    settings.crypto.keys = (SecretStr("dummy"),)
    settings.mcp.public_url = None
    settings.mcp.jwt_secret = None
    settings.github = MagicMock()
    settings.github.app_id = None

    runtime = SlackRuntime(
        settings=settings,
        anthropic=build_fake_anthropic(
            _build_edit_repo_ma_handler(tenant_id=tenant_id, ma_agent_id=ma_agent_id)
        ),
        sessionmaker=db_session_factory,
        billing_config=None,
        http_client=MagicMock(spec=httpx.AsyncClient),
        resolver_cache=MagicMock(),  # pyright: ignore[reportArgumentType]  # stub, turn path not exercised
        turn_deps=MagicMock(),  # pyright: ignore[reportArgumentType]  # stub, turn path not exercised
    )

    await run_edit_repo_submission(
        runtime,
        client_fake.client,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        view_id="V_SUBMIT_TEST",
        agent_name=_AGENT_NAME,
        parent_section="repo",
        extra={"repo_url": "https://github.com/example/new.git", "pat": "", "pat_replace": False},
    )

    async with db_session_factory() as session:
        row = await get_binding(session, tenant_id=tenant_id, agent_id=agent_uuid)

    assert row is not None, "binding should still exist after edit"
    assert row.ma_secret_ref == f"inline-pat:{agent_uuid}", (
        "blank-PAT edit-repo must preserve the stored ma_secret_ref, not clobber it to anon:"
    )
    assert row.repo_url == "example/new", "repo_url should be updated to the new value"

    ephemeral_key = ("POST", yarl.URL("https://slack.com/api/chat.postEphemeral"))
    assert ephemeral_key in client_fake.mock.requests, "success ephemeral should be posted"
    ephemeral_texts = [
        call.kwargs["json"]["text"] for call in client_fake.mock.requests[ephemeral_key]
    ]
    assert any(":white_check_mark:" in t for t in ephemeral_texts), (
        "success ephemeral must be posted for the blank-PAT edit"
    )


@pytest.mark.asyncio
async def test_run_edit_repo_submission_when_pat_replace_true_stores_new_inline_pat(
    fake_slack_web_client: Any,
    db_session_factory: Any,
) -> None:
    """pat_replace=True + a typed PAT still replaces (existing behavior preserved)."""
    from cryptography.fernet import Fernet
    from daimon.adapters.slack.runtime import SlackRuntime
    from daimon.core.ma_identity import derive_agent_uuid, derive_tenant_uuid
    from daimon.core.stores.agent_repo_binding import get_binding
    from daimon.testing.factories import make_tenant

    tenant_id = derive_tenant_uuid(platform="slack", workspace_id=_TEAM_ID)
    ma_agent_id = f"agent_{'d' * 24}"
    agent_uuid = derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=ma_agent_id)

    async with db_session_factory() as session:
        await make_tenant(session, platform="slack", workspace_id=_TEAM_ID, id=tenant_id)
        await session.commit()

    client_fake: Any = fake_slack_web_client
    _override_users_info_admin(client_fake.mock)

    fernet_key = Fernet.generate_key().decode()
    settings: MagicMock = MagicMock()
    settings.crypto.keys = (SecretStr(fernet_key),)
    settings.mcp.public_url = None
    settings.mcp.jwt_secret = None
    settings.github = MagicMock()
    settings.github.app_id = None
    settings.github.oauth_scopes = ("repo",)

    runtime = SlackRuntime(
        settings=settings,
        anthropic=build_fake_anthropic(
            _build_edit_repo_ma_handler(tenant_id=tenant_id, ma_agent_id=ma_agent_id)
        ),
        sessionmaker=db_session_factory,
        billing_config=None,
        http_client=MagicMock(spec=httpx.AsyncClient),
        resolver_cache=MagicMock(),  # pyright: ignore[reportArgumentType]  # stub, turn path not exercised
        turn_deps=MagicMock(),  # pyright: ignore[reportArgumentType]  # stub, turn path not exercised
    )

    await run_edit_repo_submission(
        runtime,
        client_fake.client,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        view_id="V_SUBMIT_TEST",
        agent_name=_AGENT_NAME,
        parent_section="repo",
        extra={
            "repo_url": "https://github.com/example/private.git",
            "pat": "ghp_newtoken1234",
            "pat_replace": True,
        },
    )

    async with db_session_factory() as session:
        row = await get_binding(session, tenant_id=tenant_id, agent_id=agent_uuid)

    assert row is not None, "binding should exist after first-time bind with a replaced PAT"
    assert row.ma_secret_ref == f"inline-pat:{agent_uuid}", (
        "pat_replace=True must store the inline PAT reference"
    )
    assert row.repo_url == "example/private", "repo_url should be bound to the new value"


# ---------------------------------------------------------------------------
# run_edit_repo_submission first-time no-PAT bind writes anon:
# ---------------------------------------------------------------------------


def _build_edit_repo_settings(*, app_id: str | None) -> MagicMock:
    settings: MagicMock = MagicMock()
    settings.crypto.keys = (SecretStr("dummy"),)
    settings.mcp.public_url = None
    settings.mcp.jwt_secret = None
    settings.github = MagicMock()
    settings.github.app_id = app_id
    return settings


@pytest.mark.asyncio
async def test_run_edit_repo_submission_no_pat_binds_anon(
    fake_slack_web_client: Any,
    db_session_factory: Any,
) -> None:
    """First-time no-PAT bind writes an anon: binding and reports success.

    No App-coverage probe on Slack: the Slack create_session call does not
    thread session_factory, so the repo-clone path never runs on Slack today —
    the panel must not advertise App coverage. Wiring Slack repo clone is a
    tracked follow-up.
    """
    from daimon.adapters.slack.runtime import SlackRuntime
    from daimon.core.ma_identity import derive_agent_uuid, derive_tenant_uuid
    from daimon.core.stores.agent_repo_binding import get_binding
    from daimon.testing.factories import make_tenant

    tenant_id = derive_tenant_uuid(platform="slack", workspace_id=_TEAM_ID)
    ma_agent_id = f"agent_{'e' * 24}"
    agent_uuid = derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=ma_agent_id)

    async with db_session_factory() as session:
        await make_tenant(session, platform="slack", workspace_id=_TEAM_ID, id=tenant_id)
        await session.commit()

    client_fake: Any = fake_slack_web_client
    _override_users_info_admin(client_fake.mock)

    runtime = SlackRuntime(
        settings=_build_edit_repo_settings(app_id=None),
        anthropic=build_fake_anthropic(
            _build_edit_repo_ma_handler(tenant_id=tenant_id, ma_agent_id=ma_agent_id)
        ),
        sessionmaker=db_session_factory,
        billing_config=None,
        http_client=MagicMock(spec=httpx.AsyncClient),
        resolver_cache=MagicMock(),  # pyright: ignore[reportArgumentType]  # stub, turn path not exercised
        turn_deps=MagicMock(),  # pyright: ignore[reportArgumentType]  # stub, turn path not exercised
    )

    await run_edit_repo_submission(
        runtime,
        client_fake.client,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        view_id="V_SUBMIT_TEST",
        agent_name=_AGENT_NAME,
        parent_section="repo",
        extra={
            "repo_url": "https://github.com/example/covered.git",
            "pat": "",
            "pat_replace": False,
        },
    )

    async with db_session_factory() as session:
        row = await get_binding(session, tenant_id=tenant_id, agent_id=agent_uuid)

    assert row is not None, "binding should exist after first-time no-PAT bind"
    assert row.ma_secret_ref == "anon:", "no-PAT bind writes an anon: binding"

    ephemeral_key = ("POST", yarl.URL("https://slack.com/api/chat.postEphemeral"))
    ephemeral_texts = [
        call.kwargs["json"]["text"] for call in client_fake.mock.requests[ephemeral_key]
    ]
    assert any("Saved repo + auth" in t for t in ephemeral_texts), (
        "user should see the plain save confirmation"
    )
    assert not any("App-covered" in t for t in ephemeral_texts), (
        "Slack panel must not advertise App coverage (Slack does not clone yet)"
    )
