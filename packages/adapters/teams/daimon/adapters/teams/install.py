"""Authenticated installation activities provision or archive the Teams tenant."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID
from weakref import WeakValueDictionary

import structlog
from daimon.adapters.teams.identity import (
    _canonical_uuid,  # pyright: ignore[reportPrivateUsage]
    _value,  # pyright: ignore[reportPrivateUsage]
)
from daimon.adapters.teams.runtime import TeamsRuntime
from daimon.core.config import TeamsSettings
from daimon.core.defaults.provisioning import (
    archive_tenant,
    provision_tenant,
    reconcile_tenant_defaults,
)
from daimon.core.defaults.report import compose_failure_reason
from daimon.core.ma_identity import derive_tenant_uuid
from daimon.core.stores.tenants import get_tenant, set_provision_status
from microsoft_teams.api import InstallUpdateActivity  # pyright: ignore[reportMissingTypeStubs]
from microsoft_teams.apps.routing import (  # pyright: ignore[reportMissingTypeStubs]
    ActivityContext,
)

log = structlog.get_logger(__name__)
WELCOME_MESSAGE = "Welcome! I can answer questions and help with tasks. Ask me here in chat."
DECLINED_MESSAGE = "Daimon isn't enabled for your organisation yet."


# Weak values keep idle tenants out of the lock registry. Active holders and
# waiters retain the lock; core upserts guarantee cross-process DB idempotency.
_installation_locks: WeakValueDictionary[UUID, asyncio.Lock] = WeakValueDictionary()


@dataclass(frozen=True)
class TeamsInstallHandler:
    """Handle installs independently of the message resolver and turn dispatcher.

    A per-tenant process lock serializes add/remove without holding connections.
    The tenant's committed ready state suppresses repeated personal installs,
    including retries with different activity ids and deliveries after restart.
    """

    settings: TeamsSettings
    runtime: TeamsRuntime

    async def __call__(self, ctx: ActivityContext[InstallUpdateActivity]) -> None:
        activity = ctx.activity
        action = _value(activity, "daimon_install_action")
        if action is None:
            action = activity.action
        channel_tenant = _value(_value(activity.channel_data, "tenant"), "id")
        conversation_tenant = activity.conversation.tenant_id
        tenant_id = _canonical_uuid(
            channel_tenant if channel_tenant is not None else conversation_tenant
        )
        if (
            tenant_id is None
            or (
                conversation_tenant is not None
                and _canonical_uuid(conversation_tenant) != tenant_id
            )
            or activity.channel_id != "msteams"
        ):
            log.warning("teams.install.refused")
            return
        if action not in ("add", "add-upgrade", "upgrade", "remove", "remove-upgrade"):
            log.info("teams.install.ignored", action=action)
            return
        removing = action in ("remove", "remove-upgrade")
        scope = activity.conversation.conversation_type
        if removing and scope == "personal":
            log.info(
                "teams.install.removed",
                action=action,
                tenant_id=tenant_id,
                user_id=activity.from_.aad_object_id,
            )
            return
        if (removing and scope not in ("channel", "groupChat")) or (
            not removing and (scope != "personal" or activity.conversation.is_group is True)
        ):
            log.warning("teams.install.refused")
            return
        if (
            not removing
            and self.settings.allowed_tenant_ids
            and tenant_id not in self.settings.allowed_tenant_ids
        ):
            reply = DECLINED_MESSAGE
            if self.settings.enrollment_contact:
                reply += "\n" + self.settings.enrollment_contact
            await ctx.send(reply)
            log.info("teams.install.declined", tenant_id=tenant_id)
            return

        derived_id = derive_tenant_uuid(platform="teams", workspace_id=tenant_id)
        sessionmaker = self.runtime.sessionmaker
        lock = _installation_locks.setdefault(derived_id, asyncio.Lock())
        async with lock:
            if removing:
                await archive_tenant(sessionmaker, tenant_id=derived_id, now=datetime.now(UTC))
                log.info(
                    "teams.install.removed",
                    action=action,
                    tenant_id=tenant_id,
                    user_id=activity.from_.aad_object_id,
                )
                return

            async with sessionmaker() as session:
                existing = await get_tenant(session, derived_id)
            if (
                existing is not None
                and existing.provision_status == "ready"
                and existing.archived_at is None
            ):
                return
            result = await provision_tenant(
                sessionmaker,
                platform="teams",
                workspace_id=tenant_id,
                signup_credit=self.runtime.settings.billing.signup_credit,
                provision_status="pending",
            )
            await set_provision_status(
                sessionmaker,
                tenant_id=result.tenant_id,
                status="pending",
                clear_archive=True,
            )
            try:
                public_url = self.runtime.settings.mcp.public_url
                report = await reconcile_tenant_defaults(
                    self.runtime.anthropic,
                    sessionmaker,
                    self.runtime.settings.defaults_root,
                    tenant_id=result.tenant_id,
                    public_url=str(public_url) if public_url is not None else None,
                )
            except Exception:
                # Provider exception strings can contain credentialed request
                # data. Persist a fixed reason and let the SDK report failure.
                await set_provision_status(
                    sessionmaker,
                    tenant_id=result.tenant_id,
                    status="failed",
                    reason="Installation defaults reconciliation raised an error.",
                )
                raise
            if report.is_failure():
                await set_provision_status(
                    sessionmaker,
                    tenant_id=result.tenant_id,
                    status="failed",
                    reason=compose_failure_reason(report),
                )
                log.warning("teams.install.defaults_failed", tenant_id=str(result.tenant_id))
                return
            await set_provision_status(
                sessionmaker, tenant_id=result.tenant_id, status="ready", clear_reason=True
            )
            await ctx.send(WELCOME_MESSAGE)
