"""Tests for `daimon.adapters.slack.continuation_dispatch.dispatch_pending_continuations`.

Real Postgres via `daimon.core.continuity.continuation` (the same store the
Slack turn-completion path calls) and a real `AsyncWebClient` intercepted by
`aioresponses` (`fake_slack_web_client`). `run_follow_up` is injected as a
plain async recorder -- the module's whole reason to accept it -- so these
tests assert claim/skip/dispatch behavior without running a second real turn.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import httpx
from daimon.adapters.slack.continuation_dispatch import dispatch_pending_continuations
from daimon.core.continuity.continuation import ContinuationRequest, record_continuation
from daimon.core.defaults.metadata import MA_METADATA_KEY_NAME, MA_METADATA_KEY_TENANT
from daimon.core.stores.domain import TaskContinuationRow
from daimon.core.stores.identity import get_or_create_platform_principal
from daimon.core.stores.task_continuations import get_continuation
from daimon.testing.factories import make_tenant
from daimon.testing.ma import (
    _agent_response as _agent_response,  # pyright: ignore[reportPrivateUsage]
)
from daimon.testing.ma import build_fake_anthropic
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

_TARGET_AGENT_ID = "agent_continuation_target"


def _fake_target_agent_handler(tenant_id_str: str) -> Any:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == f"/v1/agents/{_TARGET_AGENT_ID}":
            return httpx.Response(
                200,
                json=_agent_response(
                    agent_id=_TARGET_AGENT_ID,
                    metadata={
                        MA_METADATA_KEY_TENANT: tenant_id_str,
                        MA_METADATA_KEY_NAME: "receiving-agent",
                    },
                ),
            )
        raise AssertionError(f"unhandled {request.method} {request.url.path}")

    return handler


def _recorder() -> tuple[
    list[tuple[TaskContinuationRow, str]], Callable[[TaskContinuationRow, str], Awaitable[None]]
]:
    calls: list[tuple[TaskContinuationRow, str]] = []

    async def _run_follow_up(row: TaskContinuationRow, seed: str) -> None:
        calls.append((row, seed))

    return calls, _run_follow_up


async def test_dispatch_runs_the_follow_up_exactly_once_and_settles_delivered(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    tenant = await make_tenant(db_session, platform="slack", workspace_id="T_CONT_DISPATCH_DELIVER")
    requester = await get_or_create_platform_principal(
        db_session, tenant_id=tenant.id, platform="slack", external_id="U_REQUESTER"
    )
    await db_session.commit()

    thread_id = "9200000001.000001"
    request = ContinuationRequest(
        tenant_id=tenant.id,
        platform="slack",
        parent_channel_id="C_CONT_DISPATCH",
        thread_id=thread_id,
        requester_account_id=requester.account_id,
        requester_external_user_id="U_REQUESTER",
        target_ma_agent_id=_TARGET_AGENT_ID,
        target_name="receiving-agent",
        requested_work="please pick up the migration",
        reason="task_handoff",
        idempotency_key=uuid.uuid4(),
    )
    await record_continuation(db_session_factory, request)

    anthropic = build_fake_anthropic(_fake_target_agent_handler(str(tenant.id)))
    calls, run_follow_up = _recorder()

    await dispatch_pending_continuations(
        db_session_factory,
        anthropic,
        fake_slack_web_client.client,
        tenant_id=tenant.id,
        channel="C_CONT_DISPATCH",
        thread_id=thread_id,
        active_turn=False,
        run_follow_up=run_follow_up,
        now=lambda: datetime.now(UTC),
    )

    assert len(calls) == 1, "the follow-up must run exactly once for one pending continuation"
    dispatched_row, seed = calls[0]
    assert dispatched_row.idempotency_key == request.idempotency_key
    assert seed == "please pick up the migration"

    settled = await get_continuation(db_session, idempotency_key=request.idempotency_key)
    assert settled is not None
    assert settled.status == "delivered"
    assert settled.delivered_at is not None

    # A second dispatch call (e.g. the next turn's completion path, or a race)
    # must not run the follow-up again: the row is no longer pending.
    await dispatch_pending_continuations(
        db_session_factory,
        anthropic,
        fake_slack_web_client.client,
        tenant_id=tenant.id,
        channel="C_CONT_DISPATCH",
        thread_id=thread_id,
        active_turn=False,
        run_follow_up=run_follow_up,
        now=lambda: datetime.now(UTC),
    )
    assert len(calls) == 1, "a claimed/delivered continuation must never dispatch twice"


async def test_dispatch_skips_silently_when_no_work_was_requested(
    db_session: AsyncSession,
    db_session_factory: async_sessionmaker[AsyncSession],
    fake_slack_web_client: Any,
) -> None:
    """A handoff that only switched the responder (no `continuation` text)
    must settle `skipped`/`skip_save_only` without ever calling `run_follow_up`
    or posting anything -- nothing was ever promised."""
    tenant = await make_tenant(
        db_session, platform="slack", workspace_id="T_CONT_DISPATCH_SAVE_ONLY"
    )
    requester = await get_or_create_platform_principal(
        db_session, tenant_id=tenant.id, platform="slack", external_id="U_REQUESTER_2"
    )
    await db_session.commit()

    thread_id = "9200000002.000001"
    request = ContinuationRequest(
        tenant_id=tenant.id,
        platform="slack",
        parent_channel_id="C_CONT_DISPATCH_2",
        thread_id=thread_id,
        requester_account_id=requester.account_id,
        requester_external_user_id="U_REQUESTER_2",
        target_ma_agent_id=_TARGET_AGENT_ID,
        target_name="receiving-agent",
        requested_work=None,
        reason="task_handoff",
        idempotency_key=uuid.uuid4(),
    )
    await record_continuation(db_session_factory, request)

    anthropic = build_fake_anthropic(_fake_target_agent_handler(str(tenant.id)))
    calls, run_follow_up = _recorder()

    await dispatch_pending_continuations(
        db_session_factory,
        anthropic,
        fake_slack_web_client.client,
        tenant_id=tenant.id,
        channel="C_CONT_DISPATCH_2",
        thread_id=thread_id,
        active_turn=False,
        run_follow_up=run_follow_up,
        now=lambda: datetime.now(UTC),
    )

    assert calls == [], "no work was requested -- the follow-up must never run"
    settled = await get_continuation(db_session, idempotency_key=request.idempotency_key)
    assert settled is not None
    assert settled.status == "skipped"
    assert settled.skip_reason == "skip_save_only"

    posts = [
        req
        for (_method, url), reqs in fake_slack_web_client.mock.requests.items()
        if str(url) == "https://slack.com/api/chat.postMessage"
        for req in reqs
    ]
    assert posts == [], "a silent skip must not post anything into the thread"
