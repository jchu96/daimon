"""Installation events traverse real SDK ingress and committed tenant stores."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from daimon.adapters.teams.http_service import TeamsHttpService, create_teams_http_service
from daimon.adapters.teams.install import DECLINED_MESSAGE, WELCOME_MESSAGE
from daimon.core.config import load_settings
from daimon.core.defaults.report import Action, ApplyReport, ResourceOutcome
from daimon.core.ma_identity import derive_tenant_uuid
from daimon.core.stores.tenant_ledger import get_balance, list_for_tenant
from daimon.core.stores.tenants import get_tenant, list_tenants_by_platform
from daimon.testing.asgi import asgi_lifespan
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from structlog.testing import capture_logs

from .conftest import (
    ENTRA_TENANT_ID,
    TeamsApiFake,
    build_teams_client,
    build_teams_runtime,
    make_message_activity,
)

TENANT_ID = derive_tenant_uuid(platform="teams", workspace_id=ENTRA_TENANT_ID)
FOREIGN_TENANT = str(uuid.UUID(int=99))


def installation(action: str = "add", **kwargs: Any) -> dict[str, object]:
    activity = make_message_activity(**kwargs)
    activity.update(type="installationUpdate", action=action)
    activity.pop("text")
    return activity


@dataclass
class InstallHarness:
    service: TeamsHttpService
    client: httpx.AsyncClient
    fake: TeamsApiFake
    reconcile: AsyncMock

    async def post(self, action: str = "add", **kwargs: Any) -> httpx.Response:
        response = await self.client.post("/api/messages", json=installation(action, **kwargs))
        assert response.status_code == 200, response.text
        return response


@pytest.fixture
def install_db_factory(db_engine: AsyncEngine, db_clean: None) -> async_sessionmaker[AsyncSession]:
    # Each session must own a connection: lifecycle helpers commit independently,
    # and advisory-lock tests must exercise distinct PostgreSQL transactions.
    return async_sessionmaker(db_engine, expire_on_commit=False)


@pytest.fixture
async def installs(
    install_db_factory: async_sessionmaker[AsyncSession],
    entra_env: None,
    stub_bot_token: None,
) -> AsyncIterator[InstallHarness]:
    runtime = build_teams_runtime(install_db_factory)
    runtime.settings.billing.signup_credit = Decimal("7.50")
    fake = TeamsApiFake()
    service = create_teams_http_service(
        settings=runtime.settings.teams, runtime=runtime, client=build_teams_client(fake)
    )
    with patch(
        "daimon.adapters.teams.install.reconcile_tenant_defaults",
        new_callable=AsyncMock,
        return_value=ApplyReport(),
    ) as reconcile:
        async with asgi_lifespan(service.app):
            if service.boot_sweep_task is not None:
                await service.boot_sweep_task
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=service.app), base_url="http://test"
            ) as client:
                yield InstallHarness(service, client, fake, reconcile)


async def test_install_add_provisions_tenant(
    installs: InstallHarness, install_db_factory: async_sessionmaker[AsyncSession]
) -> None:
    async def reconcile_pending(*args: Any, **kwargs: Any) -> ApplyReport:
        async with install_db_factory() as session:
            tenant = await get_tenant(session, kwargs["tenant_id"])
        assert tenant is not None
        assert tenant.provision_status == "pending"
        assert args[0] is installs.service.runtime.anthropic
        assert args[1] is install_db_factory
        return ApplyReport()

    installs.reconcile.side_effect = reconcile_pending
    await installs.post()
    tenants = await list_tenants_by_platform(install_db_factory)
    assert len(tenants) == 1
    tenant = tenants[0]
    assert (tenant.id, tenant.platform, tenant.external_id) == (TENANT_ID, "teams", ENTRA_TENANT_ID)
    assert tenant.provision_status == "ready"
    assert tenant.archived_at is None
    installs.reconcile.assert_awaited_once()


async def test_install_add_twice_is_noop(
    installs: InstallHarness, install_db_factory: async_sessionmaker[AsyncSession]
) -> None:
    await installs.post()
    await installs.post()  # Retry of the identical event.
    await installs.post(activity_id="second-user-install", conversation_id="a:second-user")
    # A new HTTP service has no in-process dedup history.
    runtime = installs.service.runtime
    restarted = create_teams_http_service(
        settings=runtime.settings.teams, runtime=runtime, client=build_teams_client(installs.fake)
    )
    async with asgi_lifespan(restarted.app):
        if restarted.boot_sweep_task is not None:
            await restarted.boot_sweep_task
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=restarted.app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/api/messages", json=installation(activity_id="restarted")
            )
            assert response.status_code == 200
    assert len(await list_tenants_by_platform(install_db_factory)) == 1
    assert len(installs.fake.activity_requests) == 1
    installs.reconcile.assert_awaited_once()


async def test_install_add_declines_unlisted_tenant_when_closed(
    installs: InstallHarness, install_db_factory: async_sessionmaker[AsyncSession]
) -> None:
    installs.service.runtime.settings.teams.allowed_tenant_ids = frozenset({ENTRA_TENANT_ID})
    with capture_logs() as events:
        await installs.post(tenant_id=FOREIGN_TENANT, channel_tenant_id=FOREIGN_TENANT)
    assert await list_tenants_by_platform(install_db_factory) == []
    assert [request.body["text"] for request in installs.fake.activity_requests] == [
        DECLINED_MESSAGE
    ]
    installs.reconcile.assert_not_awaited()
    assert any(
        event["event"] == "teams.install.declined" and event["tenant_id"] == FOREIGN_TENANT
        for event in events
    )


async def test_install_remove_archives_tenant(
    installs: InstallHarness, install_db_factory: async_sessionmaker[AsyncSession]
) -> None:
    await installs.post()
    await installs.post("remove", activity_id="remove")
    tenants = await list_tenants_by_platform(install_db_factory)
    assert len(tenants) == 1
    assert tenants[0].id == TENANT_ID
    assert tenants[0].archived_at is not None
    assert len(installs.fake.activity_requests) == 1


async def test_install_add_posts_welcome_once(
    installs: InstallHarness,
    install_db_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    send = installs.fake.send
    committed_at_send: list[str] = []

    async def assert_committed(*args: Any, **kwargs: Any) -> httpx.Response:
        # A separate DB session must see ready before the outbound SDK call.
        async with install_db_factory() as session:
            tenant = await get_tenant(session, TENANT_ID)
        assert tenant is not None
        assert tenant.provision_status == "ready"
        committed_at_send.append(tenant.provision_status)
        return await send(*args, **kwargs)

    monkeypatch.setattr(installs.fake, "send", assert_committed)
    await installs.post()
    await installs.post(activity_id="retry")
    assert committed_at_send == ["ready"]
    request = installs.fake.activity_requests[0]
    assert request.method == "POST"
    assert "/conversations/a:conversation-1/activities" in request.url
    assert request.body["text"] == WELCOME_MESSAGE


async def test_install_never_dispatches_turn(installs: InstallHarness) -> None:
    await installs.post()
    await installs.post("remove", activity_id="remove")
    installs.service.runtime.dispatcher.dispatch.assert_not_awaited()  # type: ignore[attr-defined]
    installs.service.runtime.resolver.assert_not_awaited()  # type: ignore[attr-defined]


async def test_install_falls_back_to_conversation_tenant(
    installs: InstallHarness, install_db_factory: async_sessionmaker[AsyncSession]
) -> None:
    await installs.post(channel_tenant_id=None)
    tenants = await list_tenants_by_platform(install_db_factory)
    assert [tenant.id for tenant in tenants] == [TENANT_ID]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"channel_tenant_id": "malformed"},
        {"tenant_id": FOREIGN_TENANT},
        {"channel_tenant_id": None, "tenant_id": ""},
        {"channel_id": "emulator"},
        {"conversation_type": "groupChat"},
        {"is_group": True},
    ],
)
async def test_install_refuses_invalid_identity(
    installs: InstallHarness,
    install_db_factory: async_sessionmaker[AsyncSession],
    kwargs: dict[str, Any],
) -> None:
    await installs.post(**kwargs)
    assert await list_tenants_by_platform(install_db_factory) == []
    assert installs.fake.activity_requests == []


async def test_install_foreign_remove_cannot_archive(
    installs: InstallHarness, install_db_factory: async_sessionmaker[AsyncSession]
) -> None:
    await installs.post()
    await installs.post("remove", tenant_id=FOREIGN_TENANT, channel_tenant_id=FOREIGN_TENANT)
    tenants = await list_tenants_by_platform(install_db_factory)
    assert tenants[0].archived_at is None


async def test_install_remove_absent_tenant_is_noop(
    installs: InstallHarness, install_db_factory: async_sessionmaker[AsyncSession]
) -> None:
    await installs.post("remove")
    assert await list_tenants_by_platform(install_db_factory) == []
    assert installs.fake.activity_requests == []


async def test_install_readd_unarchives(
    installs: InstallHarness, install_db_factory: async_sessionmaker[AsyncSession]
) -> None:
    await installs.post()
    await installs.post("remove", activity_id="remove")
    await installs.post(activity_id="reinstall")
    tenants = await list_tenants_by_platform(install_db_factory)
    assert len(tenants) == 1
    assert tenants[0].archived_at is None
    assert tenants[0].provision_status == "ready"
    assert len(installs.fake.activity_requests) == 2


async def test_install_failed_defaults_remain_unready_and_retry(
    installs: InstallHarness, install_db_factory: async_sessionmaker[AsyncSession]
) -> None:
    installs.reconcile.return_value = ApplyReport(
        agents=[ResourceOutcome(kind="agent", name="default", action=Action.FAILED, error="failed")]
    )
    await installs.post()
    tenants = await list_tenants_by_platform(install_db_factory)
    assert tenants[0].provision_status == "failed"
    assert tenants[0].last_reconcile_error
    assert installs.fake.activity_requests == []
    installs.reconcile.return_value = ApplyReport()
    await installs.post()
    tenants = await list_tenants_by_platform(install_db_factory)
    assert tenants[0].provision_status == "ready"
    assert tenants[0].last_reconcile_error is None
    assert len(installs.fake.activity_requests) == 1


async def test_install_overlapping_adds_welcome_once(
    installs: InstallHarness, install_db_factory: async_sessionmaker[AsyncSession]
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed_reconcile(*args: Any, **kwargs: Any) -> ApplyReport:
        entered.set()
        await release.wait()
        return ApplyReport()

    installs.reconcile.side_effect = delayed_reconcile
    first = asyncio.create_task(installs.post())
    await asyncio.wait_for(entered.wait(), timeout=5)
    second = asyncio.create_task(installs.post(activity_id="overlapping"))
    release.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=5)
    assert len(await list_tenants_by_platform(install_db_factory)) == 1
    assert len(installs.fake.activity_requests) == 1
    installs.reconcile.assert_awaited_once()


async def test_install_reconcile_exception_keeps_tenant_unready(
    installs: InstallHarness, install_db_factory: async_sessionmaker[AsyncSession]
) -> None:
    installs.reconcile.side_effect = RuntimeError("provider unavailable")
    response = await installs.client.post("/api/messages", json=installation())
    assert response.status_code == 500
    tenants = await list_tenants_by_platform(install_db_factory)
    assert tenants[0].provision_status == "failed"
    assert (
        tenants[0].last_reconcile_error == "Installation defaults reconciliation raised an error."
    )
    assert installs.fake.activity_requests == []
    installs.reconcile.side_effect = None
    await installs.post()
    assert len(installs.fake.activity_requests) == 1


@pytest.mark.parametrize("enabled", [False, True])
async def test_install_obeys_ingress_gates(
    install_db_factory: async_sessionmaker[AsyncSession],
    stub_bot_token: None,
    enabled: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DANGEROUSLY_ALLOW_UNAUTHENTICATED_REQUESTS", raising=False)
    runtime = build_teams_runtime(install_db_factory)
    settings = runtime.settings.teams.model_copy(update={"enabled": enabled})
    service = create_teams_http_service(
        settings=settings, runtime=runtime, client=build_teams_client(TeamsApiFake())
    )
    async with asgi_lifespan(service.app):
        if service.boot_sweep_task is not None:
            await service.boot_sweep_task
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=service.app), base_url="http://test"
        ) as client:
            response = await client.post("/api/messages", json=installation())
    assert response.status_code == (401 if enabled else 503)
    assert await list_tenants_by_platform(install_db_factory) == []


async def test_install_add_provisions_any_tenant_when_open(
    installs: InstallHarness, install_db_factory: async_sessionmaker[AsyncSession]
) -> None:
    await installs.post(tenant_id=FOREIGN_TENANT, channel_tenant_id=FOREIGN_TENANT)
    await installs.post(
        tenant_id=FOREIGN_TENANT, channel_tenant_id=FOREIGN_TENANT, activity_id="retry"
    )
    tenants = await list_tenants_by_platform(install_db_factory)
    expected_id = derive_tenant_uuid(platform="teams", workspace_id=FOREIGN_TENANT)
    assert [tenant.id for tenant in tenants] == [expected_id]
    assert tenants[0].provision_status == "ready"
    async with install_db_factory() as session:
        assert await get_balance(session, tenant_id=expected_id) == Decimal("7.50")
        entries = await list_for_tenant(session, tenant_id=expected_id)
    assert len(entries) == 1
    assert entries[0].reason == "trial"
    assert len(installs.fake.activity_requests) == 1


@pytest.mark.parametrize("contact", [None, "Contact your administrator to request access."])
async def test_install_decline_replies(installs: InstallHarness, contact: str | None) -> None:
    settings = installs.service.runtime.settings.teams
    settings.allowed_tenant_ids = frozenset({ENTRA_TENANT_ID})
    settings.enrollment_contact = contact
    await installs.post(tenant_id=FOREIGN_TENANT, channel_tenant_id=FOREIGN_TENANT)
    expected = DECLINED_MESSAGE + ("\n" + contact if contact else "")
    assert [request.body["text"] for request in installs.fake.activity_requests] == [expected]


async def test_install_listed_tenant_enrolls_when_closed(
    installs: InstallHarness, install_db_factory: async_sessionmaker[AsyncSession]
) -> None:
    installs.service.runtime.settings.teams.allowed_tenant_ids = frozenset({FOREIGN_TENANT})
    await installs.post(tenant_id=FOREIGN_TENANT, channel_tenant_id=FOREIGN_TENANT)
    tenants = await list_tenants_by_platform(install_db_factory)
    assert [tenant.external_id for tenant in tenants] == [FOREIGN_TENANT]
    assert len(installs.fake.activity_requests) == 1


@pytest.mark.parametrize("allowed", [[], [str(uuid.UUID(int=0xABCDEF)).upper()]])
def test_enrollment_settings_load_from_environment(
    monkeypatch: pytest.MonkeyPatch, allowed: list[str]
) -> None:
    monkeypatch.setenv("DAIMON_DATABASE__URL", "postgresql+asyncpg://test:test@localhost/test")
    monkeypatch.setenv("DAIMON_ANTHROPIC__API_KEY", "test-key")
    monkeypatch.setenv("DAIMON_TEAMS__CLIENT_ID", "test-bot")
    monkeypatch.setenv("DAIMON_TEAMS__CLIENT_SECRET", "test-secret")
    monkeypatch.setenv("DAIMON_TEAMS__TENANT_ID", ENTRA_TENANT_ID)
    monkeypatch.setenv("DAIMON_TEAMS__ALLOWED_TENANT_IDS", json.dumps(allowed))
    settings = load_settings(_env_file=None)
    assert settings.teams is not None
    assert settings.teams.allowed_tenant_ids == frozenset(
        str(uuid.UUID(value)) for value in allowed
    )
    # The generated example must itself be a valid JSON env value, not a Python repr.
    example = Path(__file__).resolve().parents[4] / ".env.example"
    line = next(
        line
        for line in example.read_text().splitlines()
        if line.startswith("# DAIMON_TEAMS__ALLOWED_TENANT_IDS=")
    )
    assert json.loads(line.split("=", 1)[1]) == []
