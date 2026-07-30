"""Shared post-ack reachability refusal helper for Slack /agent-setup.

Field-follows-the-gate rule for the next contributor wiring up a new
mutating action: skills and MCP servers are part of the agent spec an
admin approved when the agent became reachable (a channel or workspace
default), so mutating those fields on a currently-reachable agent stays
admin-only. Repo bindings and env-variable credentials are per-agent
attachments that never enter the agent spec, so they stay open on every
agent regardless of reachability -- callers touching only those fields
must not route through this helper at all.

``refuse_if_reachable_and_not_admin`` resolves admin status live via
``resolve_is_admin`` (never trusts anything carried in the rendered view
or in ``private_metadata``) and re-reads reachability fresh from the DB on
every call -- no caching, matching the panel's existing re-resolve
discipline for admin status.
"""

from __future__ import annotations

import uuid

from daimon.adapters.slack.admin import resolve_is_admin
from daimon.adapters.slack.runtime import SlackRuntime
from daimon.core.stores.scoped_config_read import is_agent_reachable_in_tenant
from slack_sdk.web.async_client import AsyncWebClient

__all__ = ["refuse_if_reachable_and_not_admin"]


async def refuse_if_reachable_and_not_admin(
    runtime: SlackRuntime,
    web_client: AsyncWebClient,
    *,
    tenant_id: uuid.UUID,
    agent_name: str,
    channel_id: str,
    user_id: str,
    dev_allow_all: bool = False,
) -> bool:
    """Refuse a spec-touching action against a currently-reachable agent.

    Returns ``True`` when the caller must return early (refused); ``False``
    when the caller should proceed.

    An admin caller is never refused and never pays for a DB read --
    ``resolve_is_admin`` is checked first and this function returns
    immediately on ``True``. A non-admin caller is refused only when the
    target agent is currently reachable (scoped as a channel or workspace
    default); an unreachable agent has no live gate to defend, so any
    member may still configure it.

    Args:
        runtime:       Injected ``SlackRuntime`` (sessionmaker, deployment
                        default).
        web_client:     Per-event ``AsyncWebClient``.
        tenant_id:      Derived from the verified Socket Mode workspace id --
                        never accepted from the interactive payload.
        agent_name:     The target agent's name -- used only as a
                        tenant-scoped lookup key.
        channel_id:     Invoking channel, for the refusal ephemeral.
        user_id:        Invoking user, for the admin check and the ephemeral.
        dev_allow_all:  Testing-only admin-gate override, threaded through
                        unchanged from ``_dev_allow_all_admin(runtime)``.

    Returns:
        ``True`` if the caller must refuse and return early, ``False`` to
        proceed.
    """
    is_admin = await resolve_is_admin(web_client, user_id=user_id, dev_allow_all=dev_allow_all)
    if is_admin:
        return False

    async with runtime.sessionmaker() as session:
        reachable = await is_agent_reachable_in_tenant(
            session,
            tenant_id=tenant_id,
            agent_name=agent_name,
            default=runtime.deployment_default,
        )
    if not reachable:
        return False

    await web_client.chat_postEphemeral(  # pyright: ignore[reportUnknownMemberType]
        channel=channel_id or user_id,
        user=user_id,
        text=(
            ":lock: This agent is currently the default for this workspace or a "
            "channel, so changing its setup needs workspace-admin permission. "
            "Creating or forking your own agent is not restricted."
        ),
    )
    return True
