"""Tests for the Slack chat-initiated credential-request surface.

Behavioral assertions — grouped by unit:

Pure builders / evaluation:
  - build_credential_modal states every fixed fact of the request as context
    lines and collects exactly ONE input per kind (a file for env_file, a
    value or token otherwise), keeps every title inside Slack's 24-character
    cap, and carries token/channel/message_ts through private_metadata under
    the kind's callback_id.
  - evaluate_credential_submission rejects an empty or oversized value with a
    response_action errors payload keyed to the input block, rejects an
    env_file submission that is not exactly one file within the size cap, and
    never copies the secret anywhere but the decision's own field.

handle_credential_request_click (real Postgres + FakeSlackWebClient):
  - A live request clicked by its requester opens the kind's modal.
  - Wrong requester / expired / already-used / unknown token / wrong
    workspace each answer with an ephemeral and never open a modal, and a
    row that is both expired and used answers "expired" (Discord's order).
  - The repo kind refuses a non-admin whose agent cannot be resolved
    (fail closed) and lets a workspace admin through before any MA read.

run_* submissions:
  - env: consume + agent_files write commit together; the button message is
    updated in place and the requester gets an ephemeral.
  - env_file: an unreadable file leaves the request unspent and the card
    untouched; a key that already exists refuses the whole import and writes
    nothing; a good file writes every key atomically and the card becomes the
    only receipt.
  - A second submission of a consumed row writes nothing.
  - mcp: missing daimon-mcp configuration refuses BEFORE the consume.
  - repo: non-admin refused before the consume; an admin binds a public repo.
  - skill_repo: the pasted token binds the repo and the imported skills are
    attached to the agent the request named.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import yarl
from cryptography.fernet import Fernet
from daimon.adapters.slack import credential_requests as credential_requests_mod
from daimon.adapters.slack.credential_requests import (
    CRED_CALLBACK_PREFIX,
    ContinuationTrigger,
    build_credential_modal,
    evaluate_credential_submission,
    handle_credential_request_click,
    run_env_credential_submission,
    run_env_file_credential_submission,
    run_mcp_credential_submission,
    run_repo_bind_credential_submission,
    run_skill_repo_credential_submission,
)
from daimon.adapters.slack.runtime import SlackRuntime
from daimon.core.credential_requests import (
    ENV_FILE_TARGET,
    build_skill_repo_target,
    mint_request_token,
)
from daimon.core.defaults.metadata import MA_METADATA_KEY_MANAGED
from daimon.core.defaults.provisioning import derive_guild_account_uuid
from daimon.core.defaults.report import Action, ResourceOutcome
from daimon.core.env_file import MAX_ENV_FILE_BYTES
from daimon.core.errors import DaimonError
from daimon.core.github_credentials import build_multifernet, encrypt_token, get_pat
from daimon.core.ma_identity import derive_agent_uuid, derive_tenant_uuid
from daimon.core.posted_controls import (
    ALREADY_USED_MESSAGE,
    NO_LONGER_VALID_MESSAGE,
    WRONG_REQUESTER_MESSAGE,
    expired_message,
)
from daimon.core.stores.agent_files import put_agent_file
from daimon.core.stores.agent_repo_binding import get_binding
from daimon.core.stores.agent_skill_repo_credentials import get_skill_repo_credential
from daimon.core.stores.credential_requests import (
    consume_credential_request,
    create_credential_request,
    peek_credential_request,
)
from daimon.core.stores.slack_bot_tokens import upsert_slack_bot_token
from daimon.core.stores.task_continuations import list_pending_continuations
from daimon.testing import build_fake_anthropic, list_response, ma_agent, make_fake_ma_handler
from daimon.testing.factories import make_account, make_tenant
from pydantic import SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .harness import build_slack_runtime

_TEAM_ID = "T_CRED"
_USER_ID = "U_REQUESTER"
_CHANNEL_ID = "C_CRED"
_MESSAGE_TS = "1700000002.000200"

_EPHEMERAL_URL = yarl.URL("https://slack.com/api/chat.postEphemeral")
_VIEWS_OPEN_URL = yarl.URL("https://slack.com/api/views.open")
_CHAT_UPDATE_URL = yarl.URL("https://slack.com/api/chat.update")

_USERS_INFO_PATTERN = re.compile(r"https://slack\.com/api/users\.info.*")


def _override_users_info_admin(mock: Any) -> None:
    """Replace the conftest non-admin users.info stub with an admin one.

    aioresponses matches by insertion order and the conftest baseline is
    registered with repeat=True, so a plain append never wins — the existing
    users.info matchers have to be dropped first.
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
        payload={
            "ok": True,
            "user": {
                "id": _USER_ID,
                "name": "admin",
                "is_admin": True,
                "is_owner": False,
                "is_primary_owner": False,
            },
        },
        repeat=True,
    )


# ---------------------------------------------------------------------------
# build_credential_modal
# ---------------------------------------------------------------------------


_FORM_TARGETS: dict[str, str] = {
    "env": "OPENAI_API_KEY",
    "env_file": ENV_FILE_TARGET,
    "mcp": "my-server",
    "repo": build_skill_repo_target("https://github.com/owner/repo", "main", ""),
    "skill_repo": build_skill_repo_target("https://github.com/owner/skills", "main", "skills"),
}
_ALL_KINDS = ["env", "env_file", "mcp", "repo", "skill_repo"]


def _modal(
    kind: str,
    *,
    target: str | None = None,
    mcp_server_url: str | None = "https://mcp.example.com",
) -> dict[str, Any]:
    return build_credential_modal(
        kind=kind,  # type: ignore[arg-type]
        token="tok_test",
        channel_id=_CHANNEL_ID,
        message_ts=_MESSAGE_TS,
        target=target if target is not None else _FORM_TARGETS[kind],
        agent_name="tester",
        mcp_server_url=mcp_server_url,
    )


def _blocks_of(view: dict[str, Any], block_type: str) -> list[dict[str, Any]]:
    return [b for b in view["blocks"] if b["type"] == block_type]


@pytest.mark.parametrize("kind", _ALL_KINDS)
def test_every_form_states_its_facts_and_collects_one_input(kind: str) -> None:
    view = _modal(kind)
    inputs = _blocks_of(view, "input")
    contexts = _blocks_of(view, "context")
    assert view["callback_id"] == f"{CRED_CALLBACK_PREFIX}{kind}", (
        "the callback id routes the submission back to this kind"
    )
    assert len(inputs) == 1, "a private form collects exactly one thing"
    assert not inputs[0].get("optional", False), "the one input a form has is required"
    assert len(contexts) + len(inputs) == len(view["blocks"]), (
        "a form is fixed facts plus one input, nothing else"
    )
    assert "tester" in json.dumps(contexts), "the facts name the agent the request is for"


@pytest.mark.parametrize("kind", _ALL_KINDS)
def test_every_form_title_fits_slacks_cap(kind: str) -> None:
    title = _modal(kind)["title"]["text"]
    assert 0 < len(title) <= 24, "Slack rejects a view whose title exceeds 24 characters"


def test_long_key_name_title_is_truncated_rather_than_rejected() -> None:
    title = _modal("env", target="A_VERY_LONG_KEY_NAME_THAT_OVERFLOWS")["title"]["text"]
    assert title == "A_VERY_LONG_KEY_NAME_THA", "the key name is truncated to the cap, not dropped"


def test_kinds_without_a_short_target_take_their_fixed_title() -> None:
    assert _modal("env_file")["title"]["text"] == "Keys from a file", (
        "the .env sentinel target is no title"
    )
    assert _modal("repo")["title"]["text"] == "Your GitHub token", "a repo URL is no title"
    assert _modal("mcp", target="linear")["title"]["text"] == "linear token", (
        "the server name names the token being asked for"
    )


def test_env_form_collects_the_value_itself() -> None:
    element = _blocks_of(_modal("env"), "input")[0]["element"]
    assert element["type"] == "plain_text_input", "a key value is typed, not uploaded"
    assert element["multiline"] is True, "long keys must not need a single-line field"
    assert element["max_length"] == 3000, "Slack's own maximum for a plain-text input"


def test_env_file_form_collects_exactly_one_uploaded_file() -> None:
    element = _blocks_of(_modal("env_file"), "input")[0]["element"]
    assert element["type"] == "file_input", "the .env form takes an upload, not a pasted value"
    assert element["max_files"] == 1, "a whole-file import reads one file"
    assert element["filetypes"] == ["env", "txt"], "Slack filters the picker to .env-shaped files"


def test_env_file_form_says_the_file_itself_is_not_kept() -> None:
    facts = json.dumps(_blocks_of(_modal("env_file"), "context"))
    assert "one KEY=VALUE per line" in facts, "the form states the format it can read"
    assert "not a retained copy of your uploaded file" in facts, (
        "the person is told what is stored: the keys, not their file"
    )


def test_repo_form_has_no_branch_input_and_states_the_branch_instead() -> None:
    view = _modal("repo", target=build_skill_repo_target("https://github.com/o/r", "release", ""))
    inputs = _blocks_of(view, "input")
    assert len(inputs) == 1, "the repo form collects the token only"
    assert inputs[0]["label"]["text"] == "Token", "the one field is the token"
    assert "Branch" not in json.dumps(inputs), (
        "a branch field would let the form retarget the request the card described"
    )
    assert "release" in json.dumps(_blocks_of(view, "context")), (
        "the branch from the packed target is stated as a fact"
    )


def test_skill_repo_form_says_the_working_repo_is_untouched() -> None:
    facts = json.dumps(_blocks_of(_modal("skill_repo"), "context"))
    assert "working repo does not change" in facts, (
        "the skill repo is the one thing this import binds"
    )


def test_mcp_form_states_the_server_url_it_is_connecting() -> None:
    facts = json.dumps(_blocks_of(_modal("mcp"), "context"))
    assert "https://mcp.example.com" in facts, "the person sees which endpoint the token is for"


def test_modal_metadata_carries_token_channel_and_message_ts() -> None:
    view = _modal("env")
    meta = json.loads(view["private_metadata"])
    assert meta["token"] == "tok_test"
    assert meta["channel_id"] == _CHANNEL_ID
    assert meta["message_ts"] == _MESSAGE_TS


def test_modal_metadata_never_carries_the_target_secret_field() -> None:
    """The modal is built before any secret exists — nothing but routing
    handles may appear in private_metadata, ever."""
    view = _modal("env")
    meta = json.loads(view["private_metadata"])
    assert set(meta) <= {"token", "channel_id", "message_ts"}


# ---------------------------------------------------------------------------
# evaluate_credential_submission
# ---------------------------------------------------------------------------


def _submission(kind: str, values: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "view_submission",
        "team": {"id": _TEAM_ID},
        "user": {"id": _USER_ID},
        "view": {
            "callback_id": f"{CRED_CALLBACK_PREFIX}{kind}",
            "private_metadata": json.dumps(
                {"token": "tok_test", "channel_id": _CHANNEL_ID, "message_ts": _MESSAGE_TS},
                separators=(",", ":"),
            ),
            "state": {"values": values},
        },
    }


def _value_input(value: str) -> dict[str, Any]:
    return {
        "credential__value": {"credential__value": {"type": "plain_text_input", "value": value}}
    }


def test_empty_value_is_rejected_with_field_error() -> None:
    decision = evaluate_credential_submission(_submission("env", _value_input("   ")))
    assert decision.proceed is False
    assert decision.response_payload is not None
    assert decision.response_payload["response_action"] == "errors"
    assert "credential__value" in decision.response_payload["errors"]


def test_oversized_value_is_rejected_with_field_error() -> None:
    decision = evaluate_credential_submission(
        _submission("env", _value_input("é" * 3000))  # 6000 bytes, 3000 chars
    )
    assert decision.proceed is False
    assert decision.response_payload is not None
    assert "credential__value" in decision.response_payload["errors"]


def test_valid_value_proceeds_and_carries_routing_fields() -> None:
    decision = evaluate_credential_submission(_submission("env", _value_input("s3cr3t")))
    assert decision.proceed is True
    assert decision.response_payload is None
    assert decision.kind == "env"
    assert decision.value == "s3cr3t"
    assert decision.token == "tok_test"
    assert decision.channel_id == _CHANNEL_ID
    assert decision.message_ts == _MESSAGE_TS


def _file_input(files: list[dict[str, Any]]) -> dict[str, Any]:
    return {"credential__file": {"credential__file": {"type": "file_input", "files": files}}}


def _env_file_error(files: list[dict[str, Any]]) -> str:
    decision = evaluate_credential_submission(_submission("env_file", _file_input(files)))
    assert decision.proceed is False, "a file submission that cannot be read must not proceed"
    assert decision.response_payload is not None, "a refusal needs an ack payload"
    assert decision.response_payload["response_action"] == "errors", (
        "response_action errors keeps the form open instead of closing it"
    )
    errors: dict[str, str] = decision.response_payload["errors"]
    assert set(errors) == {"credential__file"}, "the error names the file field, nothing else"
    return errors["credential__file"]


def test_env_file_submission_without_a_file_is_refused_on_the_file_field() -> None:
    message = _env_file_error([])
    assert ".env" in message, "the message says what to attach"


def test_env_file_submission_with_two_files_is_refused_on_the_file_field() -> None:
    message = _env_file_error(
        [
            {"id": "F_ONE", "name": "a.env", "size": 10},
            {"id": "F_TWO", "name": "b.env", "size": 10},
        ]
    )
    assert "one file" in message, "the message says one file at a time"
    assert "F_ONE" not in message and "F_TWO" not in message, (
        "a field error names the field, never the submitted content"
    )


def test_env_file_submission_over_the_size_cap_is_refused_before_any_download() -> None:
    message = _env_file_error([{"id": "F_BIG", "name": ".env", "size": MAX_ENV_FILE_BYTES + 1}])
    assert "too big" in message and "KB" in message, "the message states the cap"
    assert "F_BIG" not in message, "a field error names the field, never the submitted content"


def test_env_file_submission_with_one_file_proceeds_with_its_id() -> None:
    decision = evaluate_credential_submission(
        _submission("env_file", _file_input([{"id": "F_ENV", "name": ".env", "size": 128}]))
    )
    assert decision.proceed is True, "one file within the cap is what the form asked for"
    assert decision.response_payload is None, "an accepted submission acks empty and closes"
    assert decision.file_id == "F_ENV", "the decision carries the handle the runner fetches with"
    assert decision.value == "", "a file submission carries a handle, never a value"


# ---------------------------------------------------------------------------
# handle_credential_request_click
# ---------------------------------------------------------------------------


async def _seed_team(session: AsyncSession, *, team_id: str = _TEAM_ID) -> tuple[uuid.UUID, str]:
    fernet_key = Fernet.generate_key().decode()
    fernet = build_multifernet((fernet_key,))
    tenant = await make_tenant(session, platform="slack", workspace_id=team_id)
    # The guild account row the request rows point at: `set_binding` writes
    # `RepoAccessProof.account_id` with an FK to accounts.id, so any test that
    # reaches a binding write needs it to exist.
    await make_account(session, tenant=tenant, id=derive_guild_account_uuid(tenant_id=tenant.id))
    await upsert_slack_bot_token(
        session, team_id=team_id, encrypted_token=encrypt_token(fernet, "xoxb-test")
    )
    await session.flush()
    return tenant.id, fernet_key


async def _seed_request(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    kind: str = "env",
    target: str = "OPENAI_API_KEY",
    requester: str = _USER_ID,
    expires_in: timedelta = timedelta(minutes=30),
    mcp_server_url: str | None = None,
    agent_id: uuid.UUID | None = None,
    posted_message_id: str | None = _MESSAGE_TS,
    origin_thread_id: str | None = None,
    requested_work: str | None = None,
    replaces_updated_at: datetime | None = None,
    target_ma_agent_id: str = "ag_test",
) -> str:
    token = mint_request_token()
    await create_credential_request(
        session,
        token=token,
        kind=kind,  # type: ignore[arg-type]
        tenant_id=tenant_id,
        agent_id=agent_id if agent_id is not None else uuid.uuid4(),
        account_id=derive_guild_account_uuid(tenant_id=tenant_id),
        target=target,
        mcp_server_url=mcp_server_url,
        requester_platform_user_id=requester,
        channel_id=_CHANNEL_ID,
        platform="slack",
        parent_channel_id=_CHANNEL_ID,
        origin_thread_id=origin_thread_id,
        posted_message_id=posted_message_id,
        expires_at=datetime.now(UTC) + expires_in,
        idempotency_key=uuid.uuid4(),
        target_ma_agent_id=target_ma_agent_id,
        target_name="tester",
        responder_name="Daimon",
        requested_work=requested_work,
        replaces_updated_at=replaces_updated_at,
    )
    await session.flush()
    return token


async def _noop_dispatch() -> None:
    """The continuation trigger for the tests that are not about dispatching."""


def _recording_trigger(
    seen: list[int], *, client: Any = None, raises: Exception | None = None
) -> ContinuationTrigger:
    """A trigger that records how many card edits had landed when it ran.

    Recording the edit count is what lets a test assert the ORDER — the
    receipt is on the card before the turn it unblocks is dispatched — and
    `raises` stages a dispatch that fails after a save that already committed.
    """

    async def _trigger() -> None:
        seen.append(len(_chat_updates(client)) if client is not None else 0)
        if raises is not None:
            raise raises

    return _trigger


def _build_runtime(
    fernet_key: str,
    db_factory: async_sessionmaker[AsyncSession],
    *,
    anthropic_handler: Any = None,
    mcp_configured: bool = False,
) -> SlackRuntime:
    settings = MagicMock()
    settings.mcp.public_url = "https://mcp.example.com/mcp" if mcp_configured else None
    settings.mcp.jwt_secret = SecretStr("x" * 32) if mcp_configured else None
    settings.github.oauth_scopes = ("repo",)
    return build_slack_runtime(
        fernet_key,
        db_factory,
        anthropic=build_fake_anthropic(anthropic_handler or make_fake_ma_handler()),
        settings=settings,
    )


def _click_payload(token: str, *, user_id: str = _USER_ID) -> dict[str, Any]:
    return {
        "type": "block_actions",
        "team": {"id": _TEAM_ID},
        "user": {"id": user_id},
        "channel": {"id": _CHANNEL_ID},
        "container": {"message_ts": _MESSAGE_TS},
        "message": {"ts": _MESSAGE_TS},
        "trigger_id": "TRIGGER_TEST",
        "actions": [{"action_id": "credential_request", "value": token}],
    }


@pytest.mark.asyncio
async def test_live_request_clicked_by_requester_opens_the_kind_modal(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    tenant_id, fernet_key = await _seed_team(db_session)
    token = await _seed_request(db_session, tenant_id=tenant_id, kind="env")
    await db_session.commit()
    runtime = _build_runtime(fernet_key, db_session_factory)

    await handle_credential_request_click(runtime, _click_payload(token))

    opens = fake_slack_web_client.mock.requests.get(("POST", _VIEWS_OPEN_URL), [])
    assert len(opens) == 1, "a live request clicked by its requester must open the modal"
    view = opens[0].kwargs["json"]["view"]
    assert view["callback_id"] == f"{CRED_CALLBACK_PREFIX}env"
    for block in [b for b in view["blocks"] if b["type"] == "input"]:
        assert 1 <= block["element"]["max_length"] <= 3000, (
            "Slack rejects plain-text inputs whose maximum length exceeds 3000"
        )
    assert "tester" in json.dumps(view["blocks"]), (
        "the form states the agent the request named, from the row"
    )
    meta = json.loads(view["private_metadata"])
    assert meta["token"] == token


@pytest.mark.asyncio
async def test_wrong_requester_gets_ephemeral_and_no_modal(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    tenant_id, fernet_key = await _seed_team(db_session)
    token = await _seed_request(db_session, tenant_id=tenant_id, requester="U_SOMEONE_ELSE")
    await db_session.commit()
    runtime = _build_runtime(fernet_key, db_session_factory)

    await handle_credential_request_click(runtime, _click_payload(token))

    assert ("POST", _VIEWS_OPEN_URL) not in fake_slack_web_client.mock.requests
    assert _ephemeral_texts(fake_slack_web_client) == [WRONG_REQUESTER_MESSAGE], (
        "the click refusal is the shared copy, character for character"
    )


@pytest.mark.asyncio
async def test_env_file_request_click_opens_the_upload_form(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    tenant_id, fernet_key = await _seed_team(db_session)
    token = await _seed_request(
        db_session, tenant_id=tenant_id, kind="env_file", target=ENV_FILE_TARGET
    )
    await db_session.commit()
    runtime = _build_runtime(fernet_key, db_session_factory)

    await handle_credential_request_click(runtime, _click_payload(token))

    opens = fake_slack_web_client.mock.requests.get(("POST", _VIEWS_OPEN_URL), [])
    assert len(opens) == 1, "an env_file request opens a form like every other kind"
    view = opens[0].kwargs["json"]["view"]
    assert view["callback_id"] == f"{CRED_CALLBACK_PREFIX}env_file"
    elements = [b["element"]["type"] for b in view["blocks"] if b["type"] == "input"]
    assert elements == ["file_input"], "the env_file form asks for an upload"


@pytest.mark.asyncio
async def test_expired_request_gets_ephemeral_and_no_modal(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    tenant_id, fernet_key = await _seed_team(db_session)
    token = await _seed_request(db_session, tenant_id=tenant_id, expires_in=timedelta(minutes=-1))
    await db_session.commit()
    runtime = _build_runtime(fernet_key, db_session_factory)

    await handle_credential_request_click(runtime, _click_payload(token))

    assert ("POST", _VIEWS_OPEN_URL) not in fake_slack_web_client.mock.requests
    assert _ephemeral_texts(fake_slack_web_client) == [
        expired_message(
            kind="env",
            agent_name="tester",
            responder_name="Daimon",
            target="OPENAI_API_KEY",
        )
    ], "an expired click says exactly what the expired card beside it says"


@pytest.mark.asyncio
async def test_unknown_token_gets_ephemeral_and_no_modal(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    _tenant_id, fernet_key = await _seed_team(db_session)
    await db_session.commit()
    runtime = _build_runtime(fernet_key, db_session_factory)

    await handle_credential_request_click(runtime, _click_payload("never-minted"))

    assert ("POST", _VIEWS_OPEN_URL) not in fake_slack_web_client.mock.requests
    assert _ephemeral_texts(fake_slack_web_client) == [NO_LONGER_VALID_MESSAGE], (
        "the click refusal is the shared copy, character for character"
    )


def _ephemeral_texts(fake_slack_web_client: Any) -> list[str]:
    posts = fake_slack_web_client.mock.requests.get(("POST", _EPHEMERAL_URL), [])
    return [str((p.kwargs.get("json") or {}).get("text") or "") for p in posts]


@pytest.mark.asyncio
async def test_already_used_request_gets_ephemeral_and_no_modal(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    tenant_id, fernet_key = await _seed_team(db_session)
    token = await _seed_request(db_session, tenant_id=tenant_id)
    consumed = await consume_credential_request(db_session, token=token, now=datetime.now(UTC))
    assert consumed is not None
    await db_session.commit()
    runtime = _build_runtime(fernet_key, db_session_factory)

    await handle_credential_request_click(runtime, _click_payload(token))

    assert ("POST", _VIEWS_OPEN_URL) not in fake_slack_web_client.mock.requests
    assert _ephemeral_texts(fake_slack_web_client) == [ALREADY_USED_MESSAGE], (
        "the click refusal is the shared copy, character for character"
    )


@pytest.mark.asyncio
async def test_expired_request_answers_expired_even_when_also_used(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    """Pins the check order against Discord's `interaction_check`: a row that
    is both expired and used answers "expired", not "already used"."""
    tenant_id, fernet_key = await _seed_team(db_session)
    token = await _seed_request(db_session, tenant_id=tenant_id, expires_in=timedelta(minutes=-1))
    await db_session.execute(
        text("UPDATE credential_requests SET used_at = now() WHERE token = :token"),
        {"token": token},
    )
    await db_session.commit()
    runtime = _build_runtime(fernet_key, db_session_factory)

    await handle_credential_request_click(runtime, _click_payload(token))

    texts = _ephemeral_texts(fake_slack_web_client)
    assert any("expired" in t for t in texts)
    assert not any("already used" in t for t in texts)


@pytest.mark.asyncio
async def test_cross_workspace_click_gets_ephemeral_and_no_modal(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    """A token minted for one workspace clicked from another must refuse even
    for the right requester — the tenant check, not message routing, is what
    keeps a leaked token unusable elsewhere."""
    _tenant_id, fernet_key = await _seed_team(db_session)
    other = await make_tenant(db_session, platform="slack", workspace_id="T_ELSEWHERE")
    token = await _seed_request(db_session, tenant_id=other.id)
    await db_session.commit()
    runtime = _build_runtime(fernet_key, db_session_factory)

    await handle_credential_request_click(runtime, _click_payload(token))

    assert ("POST", _VIEWS_OPEN_URL) not in fake_slack_web_client.mock.requests
    assert any("isn't for this workspace" in t for t in _ephemeral_texts(fake_slack_web_client))


@pytest.mark.asyncio
async def test_repo_kind_refuses_non_admin_when_agent_cannot_be_resolved(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    """Fail closed: the row's derived agent uuid resolving to nothing means
    the agent was archived or deleted since the mint — a non-admin must not
    reach the modal."""
    tenant_id, fernet_key = await _seed_team(db_session)
    token = await _seed_request(
        db_session, tenant_id=tenant_id, kind="repo", target="https://github.com/o/r"
    )
    await db_session.commit()
    runtime = _build_runtime(fernet_key, db_session_factory)

    await handle_credential_request_click(runtime, _click_payload(token))

    assert ("POST", _VIEWS_OPEN_URL) not in fake_slack_web_client.mock.requests
    assert ("POST", _EPHEMERAL_URL) in fake_slack_web_client.mock.requests


# ---------------------------------------------------------------------------
# run_env_credential_submission
# ---------------------------------------------------------------------------


async def _agent_file_rows(session: AsyncSession) -> list[Any]:
    result = await session.execute(text("SELECT key, content FROM agent_files ORDER BY key"))
    return list(result.mappings())


@pytest.mark.asyncio
async def test_env_submission_consumes_row_and_writes_the_secret(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    tenant_id, fernet_key = await _seed_team(db_session)
    live_agent = ma_agent(id="agent_credentials", name="specialist", tenant_id=tenant_id)

    def ma_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/agents":
            return list_response([live_agent.model_dump(mode="json")])
        raise AssertionError(f"Unexpected MA request: {request.method} {request.url.path}")

    token = await _seed_request(
        db_session,
        agent_id=derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=live_agent.id),
        tenant_id=tenant_id,
        kind="env",
    )
    await db_session.commit()
    runtime = _build_runtime(fernet_key, db_session_factory, anthropic_handler=ma_handler)

    await run_env_credential_submission(
        runtime,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        message_ts=_MESSAGE_TS,
        token=token,
        value="s3cr3t-value",
        dispatch_continuations=_noop_dispatch,
    )

    async with db_session_factory() as s:
        rows = await _agent_file_rows(s)
        row = await peek_credential_request(s, token=token)
    assert rows and rows[0]["key"] == "OPENAI_API_KEY"
    assert rows[0]["content"] == "s3cr3t-value"
    assert row is not None and row.used_at is not None, "the consume must have committed"
    edit = fake_slack_web_client.mock.requests[("POST", _CHAT_UPDATE_URL)][0].kwargs["json"]
    assert edit["text"] == "🔑 Add OPENAI_API_KEY to tester", (
        "the consumed card keeps its headline instead of collapsing to a marker"
    )
    assert not [b for b in edit["blocks"] if b["type"] == "actions"], (
        "the consumed card must not keep a live button"
    )
    assert ("POST", _EPHEMERAL_URL) not in fake_slack_web_client.mock.requests, (
        "the card is the receipt on a success path — an ephemeral beside it would "
        "announce the same save twice in the same conversation"
    )


@pytest.mark.asyncio
async def test_env_submission_of_consumed_row_writes_nothing(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    tenant_id, fernet_key = await _seed_team(db_session)
    live_agent = ma_agent(id="agent_credentials", name="specialist", tenant_id=tenant_id)

    def ma_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/agents":
            return list_response([live_agent.model_dump(mode="json")])
        raise AssertionError(f"Unexpected MA request: {request.method} {request.url.path}")

    token = await _seed_request(
        db_session,
        tenant_id=tenant_id,
        kind="env",
        agent_id=derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=live_agent.id),
    )
    await db_session.commit()
    runtime = _build_runtime(fernet_key, db_session_factory, anthropic_handler=ma_handler)

    common: dict[str, Any] = {
        "team_id": _TEAM_ID,
        "user_id": _USER_ID,
        "channel_id": _CHANNEL_ID,
        "message_ts": _MESSAGE_TS,
        "token": token,
        "dispatch_continuations": _noop_dispatch,
    }
    await run_env_credential_submission(runtime, value="first", **common)
    await run_env_credential_submission(runtime, value="second", **common)

    async with db_session_factory() as s:
        rows = await _agent_file_rows(s)
    assert len(rows) == 1 and rows[0]["content"] == "first", (
        "a consumed row must never produce a second write"
    )


# ---------------------------------------------------------------------------
# run_env_file_credential_submission
# ---------------------------------------------------------------------------

_FILE_ID = "F_ENV_UPLOAD"
_DOWNLOAD_URL = "https://files.slack.com/files-pri/T_CRED-F_ENV_UPLOAD/download/.env"


def _agents_handler(live_agent: Any) -> Callable[[httpx.Request], httpx.Response]:
    """Serve one live MA agent on /v1/agents; every other MA call is a bug."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/agents":
            return list_response([live_agent.model_dump(mode="json")])
        raise AssertionError(f"Unexpected MA request: {request.method} {request.url.path}")

    return handler


def _chat_updates(fake_slack_web_client: Any) -> list[dict[str, Any]]:
    return [
        post.kwargs["json"]
        for post in fake_slack_web_client.mock.requests.get(("POST", _CHAT_UPDATE_URL), [])
    ]


def _patch_file_download(monkeypatch: pytest.MonkeyPatch, body: bytes) -> None:
    """Serve `files.info` and the private download URL at the HTTP boundary."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("Authorization") == "Bearer xoxb-test", (
            "a private Slack file is only readable with the workspace's own bot token"
        )
        if request.url.path == "/api/files.info":
            assert request.url.params["file"] == _FILE_ID, (
                "the runner must fetch the file the submission named"
            )
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "file": {
                        "id": _FILE_ID,
                        "name": ".env",
                        "mimetype": "text/plain",
                        "size": len(body),
                        "url_private_download": _DOWNLOAD_URL,
                    },
                },
            )
        assert str(request.url) == _DOWNLOAD_URL, f"unexpected download url: {request.url}"
        return httpx.Response(200, content=body)

    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        credential_requests_mod,
        "_download_client",
        lambda: httpx.AsyncClient(transport=transport),
    )


async def _seed_env_file_request(
    db_session: AsyncSession, *, tenant_id: uuid.UUID, live_agent: Any
) -> str:
    return await _seed_request(
        db_session,
        tenant_id=tenant_id,
        kind="env_file",
        target=ENV_FILE_TARGET,
        agent_id=derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=live_agent.id),
    )


async def _run_env_file_submission(runtime: SlackRuntime, token: str) -> None:
    await run_env_file_credential_submission(
        runtime,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        message_ts=_MESSAGE_TS,
        token=token,
        file_id=_FILE_ID,
        dispatch_continuations=_noop_dispatch,
    )


@pytest.mark.asyncio
async def test_env_file_submission_writes_every_key_and_makes_the_card_the_receipt(
    monkeypatch: pytest.MonkeyPatch,
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    tenant_id, fernet_key = await _seed_team(db_session)
    live_agent = ma_agent(id="agent_env_file", name="specialist", tenant_id=tenant_id)
    token = await _seed_env_file_request(db_session, tenant_id=tenant_id, live_agent=live_agent)
    await db_session.commit()
    _patch_file_download(monkeypatch, b"# keys\nALPHA=alpha-private\nexport BETA='beta-private'\n")
    runtime = _build_runtime(
        fernet_key, db_session_factory, anthropic_handler=_agents_handler(live_agent)
    )

    await _run_env_file_submission(runtime, token)

    async with db_session_factory() as s:
        rows = await _agent_file_rows(s)
        row = await peek_credential_request(s, token=token)
    assert [(r["key"], r["content"]) for r in rows] == [
        ("ALPHA", "alpha-private"),
        ("BETA", "beta-private"),
    ], "every key the file declared is stored, parsed by the documented grammar"
    assert row is not None and row.used_at is not None, "the consume commits with the writes"
    assert row.outcome == "applied", "the row records how the import ended"
    edits = _chat_updates(fake_slack_web_client)
    assert len(edits) == 1, "the card goes straight to its final state"
    assert edits[0]["text"] == "✅ 2 keys saved for tester.", "the card counts what it saved"
    assert "alpha-private" not in json.dumps(edits[0]), "no value ever reaches a posted message"
    assert ("POST", _EPHEMERAL_URL) not in fake_slack_web_client.mock.requests, (
        "the card is the receipt; a second one would say the same thing twice"
    )


@pytest.mark.asyncio
async def test_env_file_submission_refuses_the_whole_import_when_a_key_already_exists(
    monkeypatch: pytest.MonkeyPatch,
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    """The card promised these keys were new. A file that would replace one is
    refused whole — the other keys in it are not written either."""
    tenant_id, fernet_key = await _seed_team(db_session)
    live_agent = ma_agent(id="agent_env_file", name="specialist", tenant_id=tenant_id)
    agent_id = derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=live_agent.id)
    await put_agent_file(
        db_session,
        tenant_id=tenant_id,
        agent_id=agent_id,
        key="ALPHA",
        content="already-here",
        set_by_account_id=None,
    )
    token = await _seed_env_file_request(db_session, tenant_id=tenant_id, live_agent=live_agent)
    await db_session.commit()
    _patch_file_download(monkeypatch, b"BETA=beta-private\nALPHA=alpha-private\n")
    runtime = _build_runtime(
        fernet_key, db_session_factory, anthropic_handler=_agents_handler(live_agent)
    )

    await _run_env_file_submission(runtime, token)

    async with db_session_factory() as s:
        rows = await _agent_file_rows(s)
        row = await peek_credential_request(s, token=token)
    assert [(r["key"], r["content"]) for r in rows] == [("ALPHA", "already-here")], (
        "a collision writes nothing at all, not even the keys that would have been new"
    )
    assert row is not None and row.used_at is not None, "the click is spent either way"
    assert row.outcome == "stale_replacement", "the row records why nothing was written"
    edits = _chat_updates(fake_slack_web_client)
    assert len(edits) == 1, "the card is edited once, to the refusal"
    assert edits[0]["text"] == "🛡️ No keys were saved for tester.", (
        "the refused card says what did not happen"
    )
    rendered = json.dumps(edits[0]) + " ".join(_ephemeral_texts(fake_slack_web_client))
    assert "line 2: ALPHA is already set." in rendered, (
        "the refusal names the key and the line it came from"
    )
    assert "alpha-private" not in rendered and "beta-private" not in rendered, (
        "a refusal names lines and keys, never values"
    )
    assert _ephemeral_texts(fake_slack_web_client), "the person who submitted is told directly too"


@pytest.mark.asyncio
async def test_env_file_submission_of_an_unparsable_file_leaves_the_request_live(
    monkeypatch: pytest.MonkeyPatch,
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    tenant_id, fernet_key = await _seed_team(db_session)
    live_agent = ma_agent(id="agent_env_file", name="specialist", tenant_id=tenant_id)
    token = await _seed_env_file_request(db_session, tenant_id=tenant_id, live_agent=live_agent)
    await db_session.commit()
    _patch_file_download(monkeypatch, b"NOT A KEY LINE\n")
    runtime = _build_runtime(
        fernet_key, db_session_factory, anthropic_handler=_agents_handler(live_agent)
    )

    await _run_env_file_submission(runtime, token)

    async with db_session_factory() as s:
        rows = await _agent_file_rows(s)
        row = await peek_credential_request(s, token=token)
    assert not rows, "a rejected file writes nothing"
    assert row is not None and row.used_at is None, (
        "a file we could not read must not spend the request — the corrected one reuses it"
    )
    assert ("POST", _CHAT_UPDATE_URL) not in fake_slack_web_client.mock.requests, (
        "the card stays in requested, button and all"
    )
    texts = _ephemeral_texts(fake_slack_web_client)
    assert texts and texts[0].startswith("No keys were saved for tester."), (
        "the person is told nothing changed"
    )
    assert "line 1:" in texts[0], "the rejection names the line that could not be read"


@pytest.mark.asyncio
async def test_env_file_submission_refuses_a_file_that_is_oversized_after_download(
    monkeypatch: pytest.MonkeyPatch,
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    """The size checked before the download is the submitting client's claim.
    The bytes are what count, and they are measured here."""
    tenant_id, fernet_key = await _seed_team(db_session)
    live_agent = ma_agent(id="agent_env_file", name="specialist", tenant_id=tenant_id)
    token = await _seed_env_file_request(db_session, tenant_id=tenant_id, live_agent=live_agent)
    await db_session.commit()
    _patch_file_download(monkeypatch, b"BIG=" + b"x" * MAX_ENV_FILE_BYTES + b"\n")
    runtime = _build_runtime(
        fernet_key, db_session_factory, anthropic_handler=_agents_handler(live_agent)
    )

    await _run_env_file_submission(runtime, token)

    async with db_session_factory() as s:
        rows = await _agent_file_rows(s)
        row = await peek_credential_request(s, token=token)
    assert not rows, "an oversized file is never parsed, so nothing is written"
    assert row is not None and row.used_at is None, "an unread file does not spend the request"
    assert ("POST", _CHAT_UPDATE_URL) not in fake_slack_web_client.mock.requests
    assert any("too big" in text for text in _ephemeral_texts(fake_slack_web_client)), (
        "the person is told the file was too big, not just that it failed"
    )


# ---------------------------------------------------------------------------
# run_mcp_credential_submission
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_submission_with_unconfigured_mcp_refuses_before_the_consume(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    tenant_id, fernet_key = await _seed_team(db_session)
    live_agent = ma_agent(id="agent_credentials", name="specialist", tenant_id=tenant_id)

    def ma_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/agents":
            return list_response([live_agent.model_dump(mode="json")])
        raise AssertionError(f"Unexpected MA request: {request.method} {request.url.path}")

    token = await _seed_request(
        db_session,
        agent_id=derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=live_agent.id),
        tenant_id=tenant_id,
        kind="mcp",
        target="my-server",
        mcp_server_url="https://mcp.example.com",
    )
    await db_session.commit()
    runtime = _build_runtime(
        fernet_key, db_session_factory, anthropic_handler=ma_handler
    )  # mcp settings are None

    await run_mcp_credential_submission(
        runtime,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        message_ts=_MESSAGE_TS,
        token=token,
        value="mcp-token",
        dispatch_continuations=_noop_dispatch,
    )

    async with db_session_factory() as s:
        row = await peek_credential_request(s, token=token)
    assert row is not None and row.used_at is None, (
        "a config refusal must land before the consume so the request survives"
    )
    assert ("POST", _EPHEMERAL_URL) in fake_slack_web_client.mock.requests
    receipt = " ".join(_ephemeral_texts(fake_slack_web_client))
    assert "Ask the operator" in receipt and "Nothing was saved" in receipt, (
        "an unconfigured deployment should give an operator handoff and truthful save status"
    )
    assert "public_url" not in receipt and "jwt_secret" not in receipt, (
        "the person should not receive internal deployment setting names"
    )


# ---------------------------------------------------------------------------
# run_repo_bind_credential_submission
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repo_submission_refuses_non_admin_before_the_consume(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    tenant_id, fernet_key = await _seed_team(db_session)
    token = await _seed_request(
        db_session, tenant_id=tenant_id, kind="repo", target="https://github.com/o/r"
    )
    await db_session.commit()
    runtime = _build_runtime(fernet_key, db_session_factory)

    await run_repo_bind_credential_submission(
        runtime,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        message_ts=_MESSAGE_TS,
        token=token,
        value="",
        dispatch_continuations=_noop_dispatch,
    )

    async with db_session_factory() as s:
        row = await peek_credential_request(s, token=token)
    assert row is not None and row.used_at is None, (
        "the admin gate is the authorization boundary and must precede the consume"
    )
    assert ("POST", _EPHEMERAL_URL) in fake_slack_web_client.mock.requests
    assert ("POST", _CHAT_UPDATE_URL) not in fake_slack_web_client.mock.requests


@pytest.mark.asyncio
async def test_repo_submission_by_admin_binds_a_public_repo(
    monkeypatch: pytest.MonkeyPatch,
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    """A workspace admin (per users.info) passes the gate and binds the repo."""
    monkeypatch.setattr(credential_requests_mod, "is_public_repo", AsyncMock(return_value=True))
    _override_users_info_admin(fake_slack_web_client.mock)
    tenant_id, fernet_key = await _seed_team(db_session)
    live_agent = ma_agent(id="agent_credentials", name="specialist", tenant_id=tenant_id)

    def ma_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/agents":
            return list_response([live_agent.model_dump(mode="json")])
        raise AssertionError(f"Unexpected MA request: {request.method} {request.url.path}")

    token = await _seed_request(
        db_session,
        agent_id=derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=live_agent.id),
        tenant_id=tenant_id,
        kind="repo",
        target=build_skill_repo_target("https://github.com/o/r", "release", ""),
    )
    await db_session.commit()
    runtime = _build_runtime(fernet_key, db_session_factory, anthropic_handler=ma_handler)

    await run_repo_bind_credential_submission(
        runtime,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        message_ts=_MESSAGE_TS,
        token=token,
        value="",
        dispatch_continuations=_noop_dispatch,
    )

    async with db_session_factory() as s:
        row = await peek_credential_request(s, token=token)
        assert row is not None and row.used_at is not None
        binding = await get_binding(s, tenant_id=row.tenant_id, agent_id=row.agent_id)
    assert binding is not None and binding.ma_secret_ref == "anon:"
    assert binding.proof_kind == "public"
    assert binding.repo_url == "o/r", (
        "the repo URL is unpacked from the target before it is stored — a packed "
        "target would store the branch as part of the repo name"
    )
    assert binding.default_branch == "release", (
        "the branch comes from the target the card was posted for, not from the form"
    )
    assert ("POST", _CHAT_UPDATE_URL) in fake_slack_web_client.mock.requests


# ---------------------------------------------------------------------------
# run_skill_repo_credential_submission
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_skill_repo_submission_writes_the_skill_credential_not_the_working_repo_binding(
    monkeypatch: pytest.MonkeyPatch,
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    """Importing puts skills in the tenant library; the request names an agent,
    so the submission must also attach them — import-without-attach leaves the
    user with an agent that has no skills and a success message saying
    otherwise. The pasted token lands in the skill-repo credential store and
    NOWHERE else: somebody offering a token so an agent can read skills out of
    a repo has not asked for that repo to become the agent's checkout, and the
    card they clicked said the working repo does not change."""
    monkeypatch.setattr(
        credential_requests_mod, "pat_can_access_repo", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        credential_requests_mod,
        "run_skill_sync",
        AsyncMock(
            return_value=[
                ResourceOutcome(
                    kind="skill",
                    name="imported-skill",
                    action=Action.CREATED,
                    anthropic_id="skill_01imported",
                )
            ]
        ),
    )
    tenant_id, fernet_key = await _seed_team(db_session)
    ma_agent_id = "agent_slack_skill_attach"
    assert tenant_id == derive_tenant_uuid(platform="slack", workspace_id=_TEAM_ID)
    agent_id = derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=ma_agent_id)
    token = await _seed_request(
        db_session,
        tenant_id=tenant_id,
        kind="skill_repo",
        target=build_skill_repo_target("https://github.com/o/attach-repo", "main", ""),
        agent_id=agent_id,
    )
    await db_session.commit()

    agent = ma_agent(id=ma_agent_id, name="daimon", tenant_id=tenant_id)
    updates: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith(agent.id):
            # Both the version-retry re-fetch and the update itself address the
            # agent directly and must parse as ONE agent; only the list route
            # gets the list envelope.
            if request.method in ("POST", "PATCH"):
                updates.append(json.loads(request.content))
            return httpx.Response(200, json=agent.model_dump(mode="json"))
        return list_response([agent.model_dump(mode="json")])

    runtime = _build_runtime(fernet_key, db_session_factory, anthropic_handler=handler)

    await run_skill_repo_credential_submission(
        runtime,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        message_ts=_MESSAGE_TS,
        token=token,
        value="ghp_slack_attach_token",
        dispatch_continuations=_noop_dispatch,
    )

    async with db_session_factory() as s:
        binding = await get_binding(s, tenant_id=tenant_id, agent_id=agent_id)
        credential = await get_skill_repo_credential(
            s,
            tenant_id=tenant_id,
            agent_id=agent_id,
            repo_url="https://github.com/o/attach-repo",
        )
    assert binding is None, "a skill-repo token must not become the agent's working repo binding"
    assert credential is not None, (
        "the skill-repo credential store is where a later sync looks for this token"
    )
    assert credential.proof_kind == "pat"
    assert (credential.default_branch, credential.path) == ("main", ""), (
        "branch and path come from the target the card was posted for"
    )
    assert updates, "the submission must call agents.update to attach the imported skills"
    attached_ids = {entry["skill_id"] for entry in updates[-1]["skills"]}
    assert "skill_01imported" in attached_ids, (
        "the newly imported skill must be attached to the agent the request named"
    )
    assert ("POST", _EPHEMERAL_URL) not in fake_slack_web_client.mock.requests, (
        "the card is the receipt on a success path"
    )
    assert _chat_updates(fake_slack_web_client)[-1]["text"] == (
        "✅ 1 skill added to tester from o/attach-repo."
    ), "the applied card must report the import through the shared change-confirmation copy"


@pytest.mark.parametrize("fails_after_storage", [False, True])
async def test_skill_repo_failure_receipt_reflects_confirmed_token_storage(
    fails_after_storage: bool,
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    tenant_id, fernet_key = await _seed_team(db_session)
    live_agent = ma_agent(id="agent_credentials", name="specialist", tenant_id=tenant_id)

    def ma_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/agents":
            return list_response([live_agent.model_dump(mode="json")])
        raise AssertionError(f"Unexpected MA request: {request.method} {request.url.path}")

    agent_id = derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=live_agent.id)
    token = await _seed_request(
        db_session,
        tenant_id=tenant_id,
        kind="skill_repo",
        agent_id=agent_id,
        target=build_skill_repo_target("https://github.com/o/skills", "main", ""),
    )
    await db_session.commit()
    checks = 0

    def github_response(request: httpx.Request) -> httpx.Response:
        nonlocal checks
        checks += 1
        if checks == 1 or (checks == 2 and fails_after_storage):
            return httpx.Response(200, json={})
        return httpx.Response(403, text="sensitive-upstream-detail")

    async with httpx.AsyncClient(transport=httpx.MockTransport(github_response)) as http_client:
        runtime = replace(
            _build_runtime(fernet_key, db_session_factory, anthropic_handler=ma_handler),
            http_client=http_client,
        )
        await run_skill_repo_credential_submission(
            runtime,
            team_id=_TEAM_ID,
            user_id=_USER_ID,
            channel_id=_CHANNEL_ID,
            message_ts=_MESSAGE_TS,
            token=token,
            value="ghp_test_private_value",
            dispatch_continuations=_noop_dispatch,
        )
    stored = await get_pat(
        principal_id=derive_guild_account_uuid(tenant_id=tenant_id),
        agent_id=agent_id,
        sessionmaker=db_session_factory,
        fernet=build_multifernet((fernet_key,)),
    )
    receipt = " ".join(_ephemeral_texts(fake_slack_web_client))
    assert (stored is not None) == fails_after_storage, (
        "fixture must fail at the intended write stage"
    )
    assert ("Token stored" in receipt) == fails_after_storage, (
        "receipt must report only confirmed storage"
    )
    assert "retry" in receipt, "failed setup must offer a reachable retry"
    assert "sensitive-upstream-detail" not in receipt, (
        "upstream details must stay out of the receipt"
    )
    assert "ghp_test_private_value" not in receipt, "the private token must stay out of the receipt"


@pytest.mark.parametrize("wrong_dimension", ["requester", "tenant", "platform", "deleted_target"])
async def test_env_submission_rechecks_request_identity_before_consuming(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
    wrong_dimension: str,
) -> None:
    tenant_id, fernet_key = await _seed_team(db_session)
    request_tenant_id = tenant_id
    if wrong_dimension == "tenant":
        other_tenant = await make_tenant(db_session, platform="slack", workspace_id="T_OTHER")
        await make_account(
            db_session, tenant=other_tenant, id=derive_guild_account_uuid(tenant_id=other_tenant.id)
        )
        request_tenant_id = other_tenant.id
    token = mint_request_token()
    await create_credential_request(
        db_session,
        token=token,
        tenant_id=request_tenant_id,
        account_id=derive_guild_account_uuid(tenant_id=request_tenant_id),
        agent_id=uuid.uuid4(),
        kind="env",
        mcp_server_url=None,
        target="TOKEN",
        requester_platform_user_id="U_OTHER" if wrong_dimension == "requester" else _USER_ID,
        channel_id=_CHANNEL_ID,
        platform="discord" if wrong_dimension == "platform" else "slack",
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
        idempotency_key=uuid.uuid4(),
        target_ma_agent_id="ag_test",
        target_name="tester",
        requested_work=None,
    )
    await db_session.commit()
    await run_env_credential_submission(
        _build_runtime(fernet_key, db_session_factory),
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        message_ts=_MESSAGE_TS,
        token=token,
        value="private-value",
        dispatch_continuations=_noop_dispatch,
    )
    async with db_session_factory() as session:
        row = await peek_credential_request(session, token=token)
        files = await _agent_file_rows(session)
    assert row is not None and row.used_at is None, "wrong identity must not consume request"
    assert not files, "wrong identity must not write private input"


async def test_env_submission_uses_durable_destination_after_origin_turn_ends(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    tenant_id, fernet_key = await _seed_team(db_session)
    live_agent = ma_agent(id="agent_credentials", name="specialist", tenant_id=tenant_id)

    def ma_handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/v1/agents":
            return list_response([live_agent.model_dump(mode="json")])
        raise AssertionError(f"Unexpected MA request: {request.method} {request.url.path}")

    token = mint_request_token()
    await create_credential_request(
        db_session,
        token=token,
        tenant_id=tenant_id,
        account_id=derive_guild_account_uuid(tenant_id=tenant_id),
        agent_id=derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=live_agent.id),
        kind="env",
        mcp_server_url=None,
        target="TOKEN",
        requester_platform_user_id=_USER_ID,
        channel_id="C_ORIGIN",
        platform="slack",
        parent_channel_id="C_ORIGIN",
        origin_thread_id="123.456",
        posted_message_id="123.789",
        expires_at=datetime.now(UTC) + timedelta(minutes=5),
        idempotency_key=uuid.uuid4(),
        target_ma_agent_id="ag_test",
        target_name="tester",
        requested_work=None,
    )
    await db_session.commit()
    await run_env_credential_submission(
        _build_runtime(fernet_key, db_session_factory, anthropic_handler=ma_handler),
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id="C_REDIRECT",
        message_ts="999.999",
        token=token,
        value="private-value",
        dispatch_continuations=_noop_dispatch,
    )
    edit = _chat_updates(fake_slack_web_client)[-1]
    assert (edit["channel"], edit["ts"]) == ("C_ORIGIN", "123.789"), (
        "card identity must come from durable request"
    )
    assert ("POST", _EPHEMERAL_URL) not in fake_slack_web_client.mock.requests, (
        "the card carries the receipt, so nothing is said in the channel the form came from"
    )
    assert "private-value" not in json.dumps(edit), "the card must not disclose private input"
    async with db_session_factory() as s:
        pending = await list_pending_continuations(
            s, tenant_id=tenant_id, platform="slack", thread_id="123.456"
        )
    assert [row.parent_channel_id for row in pending] == ["C_ORIGIN"], (
        "the queued turn is addressed to the thread the request was minted in, "
        "not to wherever the form happened to be submitted from"
    )


# ---------------------------------------------------------------------------
# Truthful card states, continuations, and the dispatch trigger
# ---------------------------------------------------------------------------

_ORIGIN_THREAD = "1700000001.000100"
_WORK = "pull last week's Toggl hours"


async def _pending_continuations(
    db_session_factory: async_sessionmaker[AsyncSession],
    *,
    tenant_id: uuid.UUID,
    thread_id: str = _ORIGIN_THREAD,
) -> list[Any]:
    async with db_session_factory() as session:
        return await list_pending_continuations(
            session, tenant_id=tenant_id, platform="slack", thread_id=thread_id
        )


async def _run_env(
    runtime: SlackRuntime, token: str, *, value: str = "s3cr3t", trigger: Any = None
) -> None:
    await run_env_credential_submission(
        runtime,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        message_ts=_MESSAGE_TS,
        token=token,
        value=value,
        dispatch_continuations=trigger if trigger is not None else _noop_dispatch,
    )


async def test_env_submission_records_a_private_input_continuation_in_the_same_transaction(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    tenant_id, fernet_key = await _seed_team(db_session)
    live_agent = ma_agent(id="agent_credentials", name="specialist", tenant_id=tenant_id)
    token = await _seed_request(
        db_session,
        tenant_id=tenant_id,
        kind="env",
        agent_id=derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=live_agent.id),
        origin_thread_id=_ORIGIN_THREAD,
        requested_work=_WORK,
        target_ma_agent_id=live_agent.id,
    )
    await db_session.commit()
    seen: list[int] = []

    await _run_env(
        _build_runtime(
            fernet_key, db_session_factory, anthropic_handler=_agents_handler(live_agent)
        ),
        token,
        trigger=_recording_trigger(seen, client=fake_slack_web_client),
    )

    async with db_session_factory() as session:
        request_row = await peek_credential_request(session, token=token)
    pending = await _pending_continuations(db_session_factory, tenant_id=tenant_id)
    assert request_row is not None and request_row.outcome == "applied", (
        "the request must be recorded as applied once the key is stored"
    )
    assert len(pending) == 1, "a saved value with work waiting on it owes exactly one turn"
    queued = pending[0]
    assert queued.reason == "private_input_applied", (
        "the queued turn must be attributed to the private input, not to a handoff"
    )
    assert queued.idempotency_key == request_row.idempotency_key, (
        "the request's own key is what stops a retried submission queueing a second turn"
    )
    assert queued.requested_work == _WORK, "the work the person was promised must survive"
    assert queued.target_ma_agent_id == live_agent.id, (
        "the continuation is addressed to the agent the request froze, never to a name"
    )
    assert seen == [2], "the trigger runs after the card is edited to received and then applied"


async def test_env_submission_records_none_requested_work_for_a_save_only_request(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    tenant_id, fernet_key = await _seed_team(db_session)
    live_agent = ma_agent(id="agent_credentials", name="specialist", tenant_id=tenant_id)
    token = await _seed_request(
        db_session,
        tenant_id=tenant_id,
        kind="env",
        agent_id=derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=live_agent.id),
        origin_thread_id=_ORIGIN_THREAD,
        requested_work=None,
        target_ma_agent_id=live_agent.id,
    )
    await db_session.commit()

    await _run_env(
        _build_runtime(
            fernet_key, db_session_factory, anthropic_handler=_agents_handler(live_agent)
        ),
        token,
    )

    pending = await _pending_continuations(db_session_factory, tenant_id=tenant_id)
    assert [row.requested_work for row in pending] == [None], (
        "a save-only request records the click and promises no turn"
    )
    card = json.dumps(_chat_updates(fake_slack_web_client)[-1])
    assert "OPENAI_API_KEY saved for tester." in card, "the card must confirm the save"
    assert "from your next message here" not in card, (
        "nothing was waiting on this key, so the card must not promise a next message"
    )


async def test_stale_replacement_leaves_the_stored_value_alone_and_renders_superseded(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    tenant_id, fernet_key = await _seed_team(db_session)
    live_agent = ma_agent(id="agent_credentials", name="specialist", tenant_id=tenant_id)
    agent_id = derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=live_agent.id)
    await put_agent_file(
        db_session,
        tenant_id=tenant_id,
        agent_id=agent_id,
        key="OPENAI_API_KEY",
        content="someone-elses-value",
        set_by_account_id=None,
    )
    token = await _seed_request(
        db_session,
        tenant_id=tenant_id,
        kind="env",
        agent_id=agent_id,
        origin_thread_id=_ORIGIN_THREAD,
        requested_work=_WORK,
        target_ma_agent_id=live_agent.id,
        # The card promised to replace a value last written an hour ago; the
        # row in the table has moved on since.
        replaces_updated_at=datetime.now(UTC) - timedelta(hours=1),
    )
    await db_session.commit()

    await _run_env(
        _build_runtime(
            fernet_key, db_session_factory, anthropic_handler=_agents_handler(live_agent)
        ),
        token,
        value="my-replacement",
    )

    async with db_session_factory() as session:
        rows = await _agent_file_rows(session)
        request_row = await peek_credential_request(session, token=token)
    assert [row["content"] for row in rows] == ["someone-elses-value"], (
        "a failed precondition must leave the stored value exactly as it was"
    )
    assert request_row is not None and request_row.outcome == "stale_replacement", (
        "the spent request must record why it wrote nothing"
    )
    assert not await _pending_continuations(db_session_factory, tenant_id=tenant_id), (
        "nothing was saved, so no turn may be queued on it"
    )
    card = _chat_updates(fake_slack_web_client)[-1]
    assert card["text"] == "⚠️ OPENAI_API_KEY was not replaced for tester.", (
        "the card must say the replacement did not happen"
    )
    assert "The current value is unchanged." in json.dumps(card)
    assert any(
        "was not replaced for tester" in text for text in _ephemeral_texts(fake_slack_web_client)
    ), "the submitter is told, in the card's own words, that a fresh request is needed"


async def test_replacement_needs_admin_at_submit_renders_refused_and_writes_nothing(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    """The role is re-decided at submit: a shared agent's key is not a private
    contribution, and the form sat open long enough for the answer to change."""
    tenant_id, fernet_key = await _seed_team(db_session)
    live_agent = ma_agent(
        id="agent_credentials",
        name="specialist",
        tenant_id=tenant_id,
        metadata={MA_METADATA_KEY_MANAGED: "true"},
    )
    agent_id = derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=live_agent.id)
    await put_agent_file(
        db_session,
        tenant_id=tenant_id,
        agent_id=agent_id,
        key="OPENAI_API_KEY",
        content="shared-value",
        set_by_account_id=None,
    )
    token = await _seed_request(
        db_session,
        tenant_id=tenant_id,
        kind="env",
        agent_id=agent_id,
        origin_thread_id=_ORIGIN_THREAD,
        requested_work=_WORK,
        target_ma_agent_id=live_agent.id,
        replaces_updated_at=datetime.now(UTC),
    )
    await db_session.commit()

    await _run_env(
        _build_runtime(
            fernet_key, db_session_factory, anthropic_handler=_agents_handler(live_agent)
        ),
        token,
        value="my-replacement",
    )

    async with db_session_factory() as session:
        rows = await _agent_file_rows(session)
        request_row = await peek_credential_request(session, token=token)
    assert [row["content"] for row in rows] == ["shared-value"], (
        "a refused replacement must not write"
    )
    assert request_row is not None and request_row.outcome == "write_failed", (
        "the refusal is part of the request's durable trace"
    )
    assert not await _pending_continuations(db_session_factory, tenant_id=tenant_id), (
        "a refused write promises no turn"
    )
    assert _chat_updates(fake_slack_web_client)[-1]["text"] == (
        "🛡️ OPENAI_API_KEY was not replaced for tester."
    ), "the card must carry the refusal, not just the ephemeral"


async def test_dispatch_trigger_runs_after_the_card_edit_and_its_failure_is_swallowed(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    tenant_id, fernet_key = await _seed_team(db_session)
    live_agent = ma_agent(id="agent_credentials", name="specialist", tenant_id=tenant_id)
    token = await _seed_request(
        db_session,
        tenant_id=tenant_id,
        kind="env",
        agent_id=derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=live_agent.id),
        origin_thread_id=_ORIGIN_THREAD,
        requested_work=_WORK,
        target_ma_agent_id=live_agent.id,
    )
    await db_session.commit()
    seen: list[int] = []

    await _run_env(
        _build_runtime(
            fernet_key, db_session_factory, anthropic_handler=_agents_handler(live_agent)
        ),
        token,
        trigger=_recording_trigger(
            seen, client=fake_slack_web_client, raises=DaimonError("thread is busy")
        ),
    )

    assert seen == [2], "the receipt is on the card before any turn is dispatched"
    assert _chat_updates(fake_slack_web_client)[-1]["text"] == (
        "✅ OPENAI_API_KEY saved for tester."
    ), "a dispatch that cannot start must not unsay a save that already committed"
    async with db_session_factory() as session:
        request_row = await peek_credential_request(session, token=token)
    assert request_row is not None and request_row.outcome == "applied", (
        "the save is durable regardless of what the dispatch does"
    )


async def test_expired_click_flips_the_card(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    """Nothing sweeps expired requests, so the first late click is the only
    chance to stop the card advertising a form that can no longer open."""
    tenant_id, fernet_key = await _seed_team(db_session)
    token = await _seed_request(
        db_session, tenant_id=tenant_id, kind="env", expires_in=timedelta(minutes=-5)
    )
    await db_session.commit()

    await handle_credential_request_click(
        _build_runtime(fernet_key, db_session_factory), _click_payload(token)
    )

    assert ("POST", _VIEWS_OPEN_URL) not in fake_slack_web_client.mock.requests, (
        "an expired request must never open a form"
    )
    card = _chat_updates(fake_slack_web_client)[-1]
    assert card["text"] == "⌛ This form expired.", "the card must stop offering the form"
    assert not [b for b in card["blocks"] if b["type"] == "actions"], (
        "the expired card must not keep a live button"
    )


async def test_wrong_requester_click_never_edits_the_card(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    tenant_id, fernet_key = await _seed_team(db_session)
    token = await _seed_request(db_session, tenant_id=tenant_id, kind="env")
    await db_session.commit()

    await handle_credential_request_click(
        _build_runtime(fernet_key, db_session_factory),
        _click_payload(token, user_id="U_SOMEONE_ELSE"),
    )

    assert ("POST", _CHAT_UPDATE_URL) not in fake_slack_web_client.mock.requests, (
        "a bystander's click must not be able to change what the requester's card says"
    )


# ---------------------------------------------------------------------------
# mcp: the attach half decides applied vs partial
# ---------------------------------------------------------------------------

_MCP_VAULT_ID = "vlt_slack_cred"
_MCP_SERVER_URL = "https://ext.example.com/mcp"


def _mcp_handler(
    live_agent: Any,
    *,
    account_id: uuid.UUID,
    agent_id: uuid.UUID,
    attach_fails: bool = False,
) -> Callable[[httpx.Request], httpx.Response]:
    """Serve the MA routes one mcp submission walks: agent, vault, credentials.

    `attach_fails` refuses the agent update the attach half needs — the token
    is stored and the server is still unreachable, which is the partial state
    the card has to tell the truth about.
    """
    display_name = f"daimon-mcp:{account_id}:{agent_id}"

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and path == "/v1/agents":
            return list_response([live_agent.model_dump(mode="json")])
        if path == f"/v1/agents/{live_agent.id}":
            if request.method in ("POST", "PATCH") and attach_fails:
                return httpx.Response(
                    400,
                    json={
                        "type": "error",
                        "error": {"type": "invalid_request_error", "message": "nope"},
                    },
                )
            return httpx.Response(200, json=live_agent.model_dump(mode="json"))
        if request.method == "GET" and path == "/v1/vaults":
            return list_response(
                [
                    {
                        "id": _MCP_VAULT_ID,
                        "type": "vault",
                        "display_name": display_name,
                        "metadata": None,
                        "archived_at": None,
                        "created_at": "2026-04-01T00:00:00Z",
                    }
                ]
            )
        if path == f"/v1/vaults/{_MCP_VAULT_ID}/credentials":
            if request.method == "POST":
                body = json.loads(request.content)
                return httpx.Response(
                    200,
                    json={
                        "id": "vcrd_slack",
                        "type": "credential",
                        "vault_id": _MCP_VAULT_ID,
                        "auth": {
                            "type": "static_bearer",
                            "mcp_server_url": body["auth"]["mcp_server_url"],
                        },
                    },
                )
            return list_response([])
        raise AssertionError(f"Unexpected MA request: {request.method} {path}")

    return handler


async def _run_mcp_submission(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    *,
    attach_fails: bool,
) -> tuple[uuid.UUID, str]:
    """Seed a live mcp request and run it; returns `(tenant_id, token)`."""
    tenant_id, fernet_key = await _seed_team(db_session)
    live_agent = ma_agent(id="agent_credentials", name="specialist", tenant_id=tenant_id)
    agent_id = derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=live_agent.id)
    account_id = derive_guild_account_uuid(tenant_id=tenant_id)
    token = await _seed_request(
        db_session,
        tenant_id=tenant_id,
        kind="mcp",
        target="my-server",
        mcp_server_url=_MCP_SERVER_URL,
        agent_id=agent_id,
        origin_thread_id=_ORIGIN_THREAD,
        requested_work=_WORK,
        target_ma_agent_id=live_agent.id,
    )
    await db_session.commit()
    runtime = _build_runtime(
        fernet_key,
        db_session_factory,
        anthropic_handler=_mcp_handler(
            live_agent, account_id=account_id, agent_id=agent_id, attach_fails=attach_fails
        ),
        mcp_configured=True,
    )
    runtime.turn_deps.fernet = build_multifernet((fernet_key,))
    await run_mcp_credential_submission(
        runtime,
        team_id=_TEAM_ID,
        user_id=_USER_ID,
        channel_id=_CHANNEL_ID,
        message_ts=_MESSAGE_TS,
        token=token,
        value="mcp-token-value",
        dispatch_continuations=_noop_dispatch,
    )
    return tenant_id, token


async def test_mcp_attach_failure_renders_partial_and_records_an_audit_continuation(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    tenant_id, token = await _run_mcp_submission(db_session, db_session_factory, attach_fails=True)

    async with db_session_factory() as session:
        request_row = await peek_credential_request(session, token=token)
    assert request_row is not None and request_row.outcome == "write_failed", (
        "a token stored against a server the agent never declares is not success"
    )
    card = _chat_updates(fake_slack_web_client)[-1]
    assert card["text"] == "⚠️ my-server token saved for tester.", (
        "the card must report the half that worked and the half that did not"
    )
    assert "The connection did not finish" in json.dumps(card)
    pending = await _pending_continuations(db_session_factory, tenant_id=tenant_id)
    assert [row.requested_work for row in pending] == [None], (
        "the click is recorded for the audit trail, but a connection that is not "
        "usable yet must not promise the turn that was waiting on it"
    )


async def test_success_paths_send_no_ephemeral_receipt(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    """The card is the receipt (both would say the same thing twice)."""
    tenant_id, token = await _run_mcp_submission(db_session, db_session_factory, attach_fails=False)

    assert ("POST", _EPHEMERAL_URL) not in fake_slack_web_client.mock.requests, (
        "a successful submission says it once, on the card"
    )
    async with db_session_factory() as session:
        request_row = await peek_credential_request(session, token=token)
    assert request_row is not None and request_row.outcome == "applied"
    assert _chat_updates(fake_slack_web_client)[-1]["text"] == (
        "✅ tester is connected to my-server."
    ), "the applied card carries the shared change-confirmation copy"
    pending = await _pending_continuations(db_session_factory, tenant_id=tenant_id)
    assert [row.requested_work for row in pending] == [_WORK], (
        "a connected server resumes the work that was waiting on it"
    )
