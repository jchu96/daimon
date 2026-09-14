"""SlackDriver -- the Slack half of the PlatformDriver Protocol.

Builds a SlackRuntime + SlackApp from an injected MARouter-backed fake
AsyncAnthropic and drives a turn via the REAL `SlackApp._handle_app_mention`
entry point (D-02) -- dedup, per-event token resolve, Slack Connect gate,
tenant resolve, THEN `_orchestrate` -> `_run_thread_turn` -> `run_turn`.
`create_session` is boundary-stubbed (mirrors the Discord driver and the
existing `_make_orchestrate_app` test pattern); `resolve_agent`,
`resolve_environment`, `build_context_xml`, and `run_turn` itself all run
for real against the injected MARouter + an aioresponses-intercepted
AsyncWebClient (D-01) -- no `unittest.mock.patch` of `run_turn`.

The posted-control half crosses the same two processes production does. The
post runs the MCP server's own `_request_*_impl`, which builds the Block Kit
card and calls `chat.postMessage` through a real `AsyncWebClient`; the click
and the submit run the bot process's `handle_credential_request_click` and
`run_*_credential_submission`. Everything Slack is intercepted by
aioresponses, so the captured cards are the exact `blocks` Slack would have
been sent.
"""

from __future__ import annotations

import contextlib
import json
import re
import sys
import time
import uuid
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Literal, cast
from unittest.mock import MagicMock, patch

import httpx
from aioresponses import aioresponses as AioResponsesMock
from cryptography.fernet import Fernet
from daimon.adapters.mcp.auth.resolver import AuthIdentity
from daimon.adapters.mcp.runtime import McpRuntime
from daimon.adapters.mcp.tools.credential_requests import (
    _request_agent_key_impl,  # pyright: ignore[reportPrivateUsage]
    _request_mcp_token_impl,  # pyright: ignore[reportPrivateUsage]
)
from daimon.adapters.slack.agent_setup import write as slack_write
from daimon.adapters.slack.app import SlackApp
from daimon.adapters.slack.credential_requests import (
    CRED_CALLBACK_PREFIX,
    evaluate_credential_submission,
    handle_credential_request_click,
    run_env_credential_submission,
    run_env_file_credential_submission,
    run_mcp_credential_submission,
)
from daimon.adapters.slack.runtime import SlackRuntime, build_turn_deps
from daimon.core.config import (
    AnthropicSettings,
    DatabaseSettings,
    Settings,
    SlackSettings,
)
from daimon.core.credential_requests import SLACK_ACTION_ID
from daimon.core.defaults.ma_index import find_agents_by_daimon_tag
from daimon.core.defaults.provisioning import teardown_slack_install
from daimon.core.github_credentials import build_multifernet, encrypt_token
from daimon.core.ma_resolver import new_resolver_cache
from daimon.core.posted_controls import CardKind
from daimon.core.purge import AccountPurgeResult
from daimon.core.purge import purge_account as core_purge_account
from daimon.core.scope import DeploymentDefault
from daimon.core.stores.credential_requests import peek_credential_request
from daimon.core.stores.domain import Role
from daimon.core.stores.slack_bot_tokens import upsert_slack_bot_token
from daimon.core.stores.turn_origins import create_origin
from daimon.testing import ma_session
from daimon.testing.ma import MARouter, build_fake_anthropic
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .cards import CapturedCard, read_slack_card
from .protocol import parity_account_id

_SLACK_API_BASE = "https://slack.com/api"
_POST_JSON_METHODS: tuple[str, ...] = (
    "auth.test",
    "chat.postMessage",
    "chat.update",
    "chat.postEphemeral",
)
_USERS_INFO_PATTERN = re.compile(r"https://slack\.com/api/users\.info.*")
_REACTIONS_ADD_PATTERN = re.compile(r"https://slack\.com/api/reactions\.add.*")
_CONVERSATIONS_REPLIES_PATTERN = re.compile(r"https://slack\.com/api/conversations\.replies.*")

_BALANCE_BLOCKED_TEXT = (
    "This workspace's daimon credit is depleted. An admin can top up with `/billing`."
)
_CAP_BLOCKED_TEXT = (
    "Monthly usage cap reached for this workspace. "
    "An admin can adjust the cap with `/billing` (when available)."
)


def _register_slack_defaults(mock: AioResponsesMock) -> None:
    """Canned ok=True responses for the Slack Web API methods a turn touches.

    Mirrors `packages/adapters/slack/tests/conftest.py`'s
    `_register_slack_defaults` -- duplicated here rather than imported since
    that module lives under a package's `tests/` tree, not on the
    `daimon.testing` import path.
    """
    mock.get(  # pyright: ignore[reportUnknownMemberType]  # aioresponses has no type stubs
        _USERS_INFO_PATTERN,
        payload={
            "ok": True,
            "user": {"is_admin": False, "is_owner": False, "is_primary_owner": False},
        },
        repeat=True,
    )
    for method in _POST_JSON_METHODS:
        mock.post(  # pyright: ignore[reportUnknownMemberType]
            f"{_SLACK_API_BASE}/{method}",
            payload={"ok": True, "ts": "1000000000.000001", "channel": "C_PARITY"},
            repeat=True,
        )
    mock.post(_REACTIONS_ADD_PATTERN, payload={"ok": True}, repeat=True)  # pyright: ignore[reportUnknownMemberType]
    mock.get(  # pyright: ignore[reportUnknownMemberType]
        _CONVERSATIONS_REPLIES_PATTERN,
        payload={"ok": True, "messages": [], "has_more": False},
        repeat=True,
    )


def _extract_posted_texts(mock: AioResponsesMock) -> list[str]:
    """Every `text` body posted via chat.postMessage / chat.update, in
    request order (blocked-copy and the final agent reply both land here)."""
    posted: list[str] = []
    for (_method, url), reqs in mock.requests.items():
        if str(url) not in (
            f"{_SLACK_API_BASE}/chat.postMessage",
            f"{_SLACK_API_BASE}/chat.update",
        ):
            continue
        for req in reqs:
            # aioresponses ships no type stubs -- `req.kwargs` is Unknown;
            # narrow to Any explicitly rather than let it propagate.
            req_kwargs = cast(dict[str, Any], req.kwargs)  # pyright: ignore[reportUnknownMemberType]
            body = cast(dict[str, Any], req_kwargs.get("json") or {})
            text = cast(Any, body.get("text"))
            if isinstance(text, str):
                posted.append(text)
    return posted


_CONVERSATIONS_INFO_PATTERN = re.compile(r"https://slack\.com/api/conversations\.info.*")
_CHAT_POST_MESSAGE_URL = f"{_SLACK_API_BASE}/chat.postMessage"
_CHAT_UPDATE_URL = f"{_SLACK_API_BASE}/chat.update"
_CHAT_EPHEMERAL_URL = f"{_SLACK_API_BASE}/chat.postEphemeral"
_VIEWS_OPEN_URL = f"{_SLACK_API_BASE}/views.open"

#: The `ts` Slack hands back for the posted card. The row records it, and
#: every later click and edit is matched against it.
_POSTED_MESSAGE_TS = "1700000000.000100"
_TRIGGER_ID = "TRIGGER_PARITY"
_VALUE_BLOCK = "credential__value"
_FILE_BLOCK = "credential__file"
_UPLOADED_FILE_ID = "F_PARITY_ENV"
_DOWNLOAD_URL = "https://files.slack.com/parity/.env"

#: The daimon-mcp settings the MCP form demands before it will consume a
#: request. Fixed here: the scenarios are about the card, never about an
#: unconfigured deployment.
_MCP_PUBLIC_URL = "https://mcp.example.com/mcp"
_MCP_JWT_SECRET = "x" * 32


def _register_credential_defaults(mock: AioResponsesMock, *, channel_id: str) -> None:
    """Canned responses for every Slack call the card lifecycle makes.

    The requester is a full workspace admin and the channel is public, so
    `check_channel_access` clears on one `users.info` and the submit-time
    replacement re-check (`resolve_is_admin`) answers admin -- a replacement
    scenario has to reach its compare-and-set, not stop at the gate in front
    of it.
    """
    mock.get(  # pyright: ignore[reportUnknownMemberType]  # aioresponses has no type stubs
        _USERS_INFO_PATTERN,
        payload={
            "ok": True,
            "user": {
                "is_admin": True,
                "is_owner": False,
                "is_primary_owner": False,
                "is_restricted": False,
                "is_ultra_restricted": False,
            },
        },
        repeat=True,
    )
    mock.get(  # pyright: ignore[reportUnknownMemberType]
        _CONVERSATIONS_INFO_PATTERN,
        payload={
            "ok": True,
            "channel": {"id": channel_id, "is_private": False, "is_im": False, "is_mpim": False},
        },
        repeat=True,
    )
    for url in (_CHAT_POST_MESSAGE_URL, _CHAT_UPDATE_URL, _CHAT_EPHEMERAL_URL):
        mock.post(  # pyright: ignore[reportUnknownMemberType]
            url,
            payload={"ok": True, "ts": _POSTED_MESSAGE_TS, "channel": channel_id},
            repeat=True,
        )
    mock.post(  # pyright: ignore[reportUnknownMemberType]
        _VIEWS_OPEN_URL,
        payload={"ok": True, "view": {"id": "V_PARITY", "hash": "H_PARITY"}},
        repeat=True,
    )


def _bodies_for(mock: AioResponsesMock, url: str) -> list[dict[str, Any]]:
    """The JSON bodies posted to one Slack method, in request order."""
    bodies: list[dict[str, Any]] = []
    for (_method, request_url), reqs in mock.requests.items():
        if str(request_url) != url:
            continue
        for req in reqs:
            req_kwargs = cast(dict[str, Any], req.kwargs)  # pyright: ignore[reportUnknownMemberType]
            bodies.append(cast(dict[str, Any], req_kwargs.get("json") or {}))
    return bodies


@dataclass
class SlackDriver:
    """Drives turns through `SlackApp._handle_app_mention`, the real Slack entry point."""

    param_id: str = "slack"
    #: Every card this driver posted or edited, oldest first.
    _cards: list[CapturedCard] = field(default_factory=list[CapturedCard])
    #: One workspace key for the driver's whole life, so the bot token seeded
    #: by the first call is still decryptable by the last.
    _fernet_key: str = field(default_factory=lambda: Fernet.generate_key().decode())

    def _make_runtime(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        router: MARouter,
        *,
        fernet_key: str,
        billing_config: object | None = None,
    ) -> SlackRuntime:
        settings = MagicMock()
        settings.crypto.keys = (SecretStr(fernet_key),)
        settings.slack = SlackSettings(
            signing_secret=SecretStr("parity-signing-secret"),
            app_token=SecretStr("xapp-parity-test"),
            max_concurrent_turns_per_tenant=100,
        )
        settings.mcp.public_url = None
        settings.mcp.app_root_url = None
        settings.mcp.jwt_secret = None
        settings.defaults_root = MagicMock()
        settings.billing.markup = Decimal("1.0")
        anthropic = build_fake_anthropic(router.dispatch)
        deployment_default = DeploymentDefault(agent_name="test-agent", environment_name="test-env")
        resolver_cache = new_resolver_cache()
        turn_deps = build_turn_deps(
            settings,
            anthropic,
            sessionmaker,
            deployment_default=deployment_default,
            resolver_cache=resolver_cache,
            billing_config=billing_config,  # pyright: ignore[reportArgumentType]  # test-injected BillingConfig | None
        )
        return SlackRuntime(
            settings=settings,
            anthropic=anthropic,
            sessionmaker=sessionmaker,
            billing_config=billing_config,  # pyright: ignore[reportArgumentType]  # test-injected BillingConfig | None
            http_client=MagicMock(spec=httpx.AsyncClient),
            resolver_cache=resolver_cache,
            turn_deps=turn_deps,
            deployment_default=deployment_default,
        )

    async def dispatch_turn(
        self,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
        router: MARouter,
        tenant_id: uuid.UUID,
        workspace_id: str,
        channel_id: str,
        user_id: str,
        text: str,
        billing_config: object | None = None,
        thread_ts: str | None = None,
    ) -> list[str]:
        """`thread_ts`, when given, targets an existing thread (e.g. a
        pre-seeded live `thread_sessions` row) instead of starting a fresh
        one -- the event's own `ts`/`event_ts` stay unique per dispatch
        (mirrors a real follow-up mention), only `thread_ts` is pinned so
        `_orchestrate`'s `thread_id = event.get("thread_ts") or event.get("ts")`
        resolves to the caller-chosen id. Not part of the `PlatformDriver`
        Protocol (Discord's `channel_id` already doubles as the thread id) --
        this is Slack-only, keyword-only, and defaults to None so every
        existing call site is unaffected.
        """
        fernet_key = Fernet.generate_key().decode()
        fernet = build_multifernet((fernet_key,))
        async with sessionmaker() as s:
            await upsert_slack_bot_token(
                s, team_id=workspace_id, encrypted_token=encrypt_token(fernet, "xoxb-parity-test")
            )
            await s.commit()

        runtime = self._make_runtime(
            sessionmaker, router, fernet_key=fernet_key, billing_config=billing_config
        )
        app = SlackApp(runtime=runtime)

        event_ts = f"{time.time():.6f}"
        event: dict[str, Any] = {
            "type": "app_mention",
            "channel": channel_id,
            "event_ts": event_ts,
            "ts": event_ts,
            "user": user_id,
            "text": f"<@U_BOT> {text}",
        }
        if thread_ts is not None:
            event["thread_ts"] = thread_ts

        with (
            AioResponsesMock() as mock,
            patch("daimon.core.turn.prepare.create_session") as mock_create_session,
        ):
            _register_slack_defaults(mock)
            mock_create_session.return_value = ma_session(
                id="sess_parity_test",
                agent_id="ag_parity_test",
                model="claude-sonnet-4-6",
                environment_id="env_parity_test",
            )
            await app._handle_app_mention(event, team_id=workspace_id)  # pyright: ignore[reportPrivateUsage]
            posted = _extract_posted_texts(mock)
        return posted

    def expected_blocked_text(self, kind: Literal["balance", "cap"]) -> str:
        return _BALANCE_BLOCKED_TEXT if kind == "balance" else _CAP_BLOCKED_TEXT

    async def delete_agent(
        self,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
        router: MARouter,
        tenant_id: uuid.UUID,
        name: str,
    ) -> None:
        runtime = self._make_runtime(
            sessionmaker, router, fernet_key=Fernet.generate_key().decode()
        )
        await slack_write.delete_agent(runtime, tenant_id=tenant_id, name=name)

    async def fork_agent(
        self,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
        router: MARouter,
        tenant_id: uuid.UUID,
        source_name: str,
        new_name: str,
        account_id: uuid.UUID,
    ) -> None:
        runtime = self._make_runtime(
            sessionmaker, router, fernet_key=Fernet.generate_key().decode()
        )
        await slack_write.fork_agent(
            runtime,
            tenant_id=tenant_id,
            source_name=source_name,
            new_name=new_name,
            account_id=account_id,
        )

    async def purge_account(
        self,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
        router: MARouter,
        account_id: uuid.UUID,
    ) -> AccountPurgeResult:
        return await core_purge_account(
            sm=sessionmaker,
            account_id=account_id,
            anthropic=build_fake_anthropic(router.dispatch),
        )

    async def uninstall(
        self,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
        workspace_id: str,
    ) -> None:
        await teardown_slack_install(sessionmaker, team_id=workspace_id, now=datetime.now(UTC))

    # -- posted-control lifecycle -------------------------------------------

    async def _seed_bot_token(
        self, sessionmaker: async_sessionmaker[AsyncSession], team_id: str
    ) -> None:
        """Install this workspace's bot token, under the driver's own key."""
        fernet = build_multifernet((self._fernet_key,))
        async with sessionmaker() as session:
            await upsert_slack_bot_token(
                session,
                team_id=team_id,
                encrypted_token=encrypt_token(fernet, "xoxb-parity-test"),
            )
            await session.commit()

    def _credential_runtime(
        self, sessionmaker: async_sessionmaker[AsyncSession], router: MARouter
    ) -> SlackRuntime:
        """The bot-process runtime the click and the submit run against.

        Deliberately not `_make_runtime`: a credential submission needs the
        daimon-mcp settings the MCP form checks before it consumes anything,
        and a turn needs neither of them.
        """
        settings = MagicMock()
        settings.crypto.keys = (SecretStr(self._fernet_key),)
        settings.slack = SlackSettings(
            signing_secret=SecretStr("parity-signing-secret"),
            app_token=SecretStr("xapp-parity-test"),
            max_concurrent_turns_per_tenant=100,
        )
        settings.mcp.public_url = _MCP_PUBLIC_URL
        settings.mcp.jwt_secret = SecretStr(_MCP_JWT_SECRET)
        settings.mcp.app_root_url = None
        settings.defaults_root = MagicMock()
        settings.billing.markup = Decimal("1.0")
        settings.github.oauth_scopes = ()
        anthropic = build_fake_anthropic(router.dispatch)
        deployment_default = DeploymentDefault()
        resolver_cache = new_resolver_cache()
        return SlackRuntime(
            settings=settings,
            anthropic=anthropic,
            sessionmaker=sessionmaker,
            billing_config=None,
            http_client=MagicMock(spec=httpx.AsyncClient),
            resolver_cache=resolver_cache,
            turn_deps=build_turn_deps(
                settings,
                anthropic,
                sessionmaker,
                deployment_default=deployment_default,
                resolver_cache=resolver_cache,
                billing_config=None,
            ),
            deployment_default=deployment_default,
        )

    def _record_cards(self, mock: AioResponsesMock, url: str) -> None:
        for body in _bodies_for(mock, url):
            blocks = body.get("blocks")
            if blocks is not None:
                self._cards.append(read_slack_card(blocks))

    async def post_credential_card(
        self,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
        router: MARouter,
        tenant_id: uuid.UUID,
        workspace_id: str,
        channel_id: str,
        user_id: str,
        kind: CardKind,
        target: str,
        agent_name: str,
        mcp_server_url: str | None = None,
        branch: str | None = None,
        pending_task: str | None = None,
    ) -> str:
        await self._seed_bot_token(sessionmaker, workspace_id)
        anthropic = build_fake_anthropic(router.dispatch)
        runtime = McpRuntime(
            session_factory=sessionmaker,
            client=anthropic,
            settings=Settings(
                database=DatabaseSettings(url="postgresql+asyncpg://parity/parity"),  # pyright: ignore[reportArgumentType]  # pydantic coerces the DSN string
                anthropic=AnthropicSettings(api_key=SecretStr("parity")),
            ),
            deployment_default=DeploymentDefault(),
            fernet=build_multifernet((self._fernet_key,)),
        )
        auth = AuthIdentity(
            account_id=parity_account_id(tenant_id, user_id),
            tenant_id=tenant_id,
            role=Role.ADMIN,
            platform="slack",
            external_id=workspace_id,
            platform_user_id=user_id,
            is_admin=True,
        )
        # The tools refuse to mint for an agent they cannot pin by MA id, and
        # take that id off the origin's configuration target when the caller
        # passes none -- so the origin names the agent this card is for.
        agents = await find_agents_by_daimon_tag(anthropic, tenant_id=tenant_id, name=agent_name)
        if not agents:
            raise AssertionError(f"the router serves no agent named {agent_name!r}")
        now = datetime.now(UTC)
        async with sessionmaker.begin() as session:
            origin = await create_origin(
                session,
                tenant_id=tenant_id,
                account_id=auth.account_id,
                platform="slack",
                parent_channel_id=channel_id,
                thread_id=_POSTED_MESSAGE_TS,
                responder_ma_agent_id="ag_parity_responder",
                responder_name="Daimon",
                configuration_target_ma_agent_id=agents[0].id,
                configuration_target_name=agent_name,
                role=Role.ADMIN,
                expires_at=now + timedelta(minutes=30),
                now=now,
            )

        with AioResponsesMock() as mock:
            _register_credential_defaults(mock, channel_id=channel_id)
            if kind == "env":
                await _request_agent_key_impl(
                    runtime,
                    auth,
                    agent_name=agent_name,
                    key=target,
                    purpose="a parity scenario",
                    channel_id=channel_id,
                    pending_task=pending_task,
                    origin_context_id=str(origin.id),
                    expected_ma_agent_id=agents[0].id,
                )
            elif kind == "env_file":
                await _request_agent_key_impl(
                    runtime,
                    auth,
                    agent_name=agent_name,
                    key=None,
                    purpose="a parity scenario",
                    channel_id=channel_id,
                    pending_task=pending_task,
                    origin_context_id=str(origin.id),
                    expected_ma_agent_id=agents[0].id,
                )
            elif kind == "mcp":
                if mcp_server_url is None:
                    raise ValueError("kind='mcp' needs its server url")
                await _request_mcp_token_impl(
                    runtime,
                    auth,
                    agent_name=agent_name,
                    server_name=target,
                    url=mcp_server_url,
                    channel_id=channel_id,
                    pending_task=pending_task,
                    origin_context_id=str(origin.id),
                    expected_ma_agent_id=agents[0].id,
                )
            else:
                raise NotImplementedError(
                    f"kind={kind!r} is not wired into the parity drivers: the two repo "
                    "kinds submit against GitHub, which no scenario fakes yet"
                )
            posts = _bodies_for(mock, _CHAT_POST_MESSAGE_URL)

        if len(posts) != 1:
            raise AssertionError(f"expected exactly one posted card, got {len(posts)}")
        self._cards.append(read_slack_card(posts[0]["blocks"]))
        return _token_from_blocks(posts[0]["blocks"])

    async def click_private_input(
        self,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
        router: MARouter,
        tenant_id: uuid.UUID,
        workspace_id: str,
        channel_id: str,
        user_id: str,
        token: str,
    ) -> str | None:
        await self._seed_bot_token(sessionmaker, workspace_id)
        runtime = self._credential_runtime(sessionmaker, router)
        payload: dict[str, Any] = {
            "type": "block_actions",
            "team": {"id": workspace_id},
            "user": {"id": user_id},
            "channel": {"id": channel_id},
            "container": {"message_ts": _POSTED_MESSAGE_TS},
            "trigger_id": _TRIGGER_ID,
            "actions": [{"action_id": SLACK_ACTION_ID, "value": token}],
        }
        with AioResponsesMock() as mock:
            _register_credential_defaults(mock, channel_id=channel_id)
            await handle_credential_request_click(runtime, payload)
            self._record_cards(mock, _CHAT_UPDATE_URL)
            refusals = _bodies_for(mock, _CHAT_EPHEMERAL_URL)
            opened = _bodies_for(mock, _VIEWS_OPEN_URL)
        if opened:
            return None
        if not refusals:
            raise AssertionError("the click neither opened a form nor said why not")
        return str(refusals[0].get("text") or "")

    async def submit_private_input(
        self,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
        router: MARouter,
        tenant_id: uuid.UUID,
        workspace_id: str,
        channel_id: str,
        user_id: str,
        token: str,
        value: str = "",
        file_bytes: bytes | None = None,
    ) -> None:
        await self._seed_bot_token(sessionmaker, workspace_id)
        runtime = self._credential_runtime(sessionmaker, router)
        async with sessionmaker() as session:
            row = await peek_credential_request(session, token=token)
        if row is None:
            raise AssertionError(f"no credential request for token {token!r}")
        decision = evaluate_credential_submission(
            _submission_payload(
                kind=row.kind,
                token=token,
                channel_id=channel_id,
                team_id=workspace_id,
                user_id=user_id,
                value=value,
                file_bytes=file_bytes,
            )
        )
        if not decision.proceed:
            # The form stayed open with a field error; nothing was posted and
            # nothing was consumed, so there is no card edit to record.
            return
        with AioResponsesMock() as mock, _patched_file_download(file_bytes):
            _register_credential_defaults(mock, channel_id=channel_id)
            if decision.kind == "env":
                await run_env_credential_submission(
                    runtime,
                    team_id=workspace_id,
                    user_id=user_id,
                    channel_id=channel_id,
                    message_ts=_POSTED_MESSAGE_TS,
                    token=token,
                    value=decision.value,
                    dispatch_continuations=_no_dispatch,
                )
            elif decision.kind == "env_file":
                await run_env_file_credential_submission(
                    runtime,
                    team_id=workspace_id,
                    user_id=user_id,
                    channel_id=channel_id,
                    message_ts=_POSTED_MESSAGE_TS,
                    token=token,
                    file_id=decision.file_id or _UPLOADED_FILE_ID,
                    dispatch_continuations=_no_dispatch,
                )
            elif decision.kind == "mcp":
                await run_mcp_credential_submission(
                    runtime,
                    team_id=workspace_id,
                    user_id=user_id,
                    channel_id=channel_id,
                    message_ts=_POSTED_MESSAGE_TS,
                    token=token,
                    value=decision.value,
                    dispatch_continuations=_no_dispatch,
                )
            else:
                raise NotImplementedError(f"kind={decision.kind!r} has no parity submit path")
            self._record_cards(mock, _CHAT_UPDATE_URL)

    def captured_cards(self) -> list[CapturedCard]:
        return list(self._cards)

    def captured_card_states(self) -> list[str]:
        return [card.state for card in self._cards]


async def _no_dispatch() -> None:
    """The continuation trigger: a card scenario never runs the follow-up turn."""


def _token_from_blocks(blocks: object) -> str:
    """The request token the posted card's button carries in its `value`."""
    if isinstance(blocks, list):
        for block in cast(list[Any], blocks):
            if not isinstance(block, dict) or block.get("type") != "actions":
                continue
            elements: list[dict[str, Any]] = block.get("elements") or []
            for element in elements:
                if element.get("action_id") == SLACK_ACTION_ID:
                    return str(element.get("value") or "")
    raise AssertionError("the posted card carries no credential button")


def _submission_payload(
    *,
    kind: str,
    token: str,
    channel_id: str,
    team_id: str,
    user_id: str,
    value: str,
    file_bytes: bytes | None,
) -> dict[str, Any]:
    """The `view_submission` payload Slack sends when the form is saved.

    The one input is the kind's own: a `plain_text_input` for the three typed
    kinds, a `file_input` carrying the upload's handle and its client-reported
    size for `env_file`.
    """
    if kind == "env_file":
        size = len(file_bytes) if file_bytes is not None else 0
        values: dict[str, Any] = {
            _FILE_BLOCK: {
                _FILE_BLOCK: {
                    "type": "file_input",
                    "files": [{"id": _UPLOADED_FILE_ID, "name": ".env", "size": size}],
                }
            }
        }
    else:
        values = {_VALUE_BLOCK: {_VALUE_BLOCK: {"type": "plain_text_input", "value": value}}}
    return {
        "type": "view_submission",
        "team": {"id": team_id},
        "user": {"id": user_id},
        "view": {
            "callback_id": f"{CRED_CALLBACK_PREFIX}{kind}",
            "private_metadata": json.dumps(
                {"token": token, "channel_id": channel_id, "message_ts": _POSTED_MESSAGE_TS},
                separators=(",", ":"),
            ),
            "state": {"values": values},
        },
    }


def _patched_file_download(file_bytes: bytes | None) -> AbstractContextManager[object]:
    """Serve `files.info` plus the private download URL at the HTTP boundary.

    The runner builds its own `httpx.AsyncClient` for the CDN round trip, so
    the seam is that factory. It lives in whichever module the runner was
    defined in, which is why it is reached through the function's own module
    rather than a hard-coded import path -- the Slack credential surface is
    split across several modules and the runner may move between them.

    A submission with no upload needs no seam at all.
    """
    if file_bytes is None:
        return contextlib.nullcontext()

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/files.info":
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "file": {
                        "id": _UPLOADED_FILE_ID,
                        "name": ".env",
                        "mimetype": "text/plain",
                        "size": len(file_bytes),
                        "url_private_download": _DOWNLOAD_URL,
                    },
                },
            )
        return httpx.Response(200, content=file_bytes)

    transport = httpx.MockTransport(_handler)
    module = sys.modules[run_env_file_credential_submission.__module__]
    return patch.object(module, "_download_client", lambda: httpx.AsyncClient(transport=transport))
