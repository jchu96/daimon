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

It refuses in two cases, checked in this order:

1. The target is a defaults-managed agent -- refused unconditionally, for
   admins too. Editing one from a panel never stamps the reconciler's spec
   hash, so the edit would survive every later reconcile and drift the agent
   from the shipped defaults with no way back. Forking is the editable path.
2. The caller is not a workspace admin and the target is currently
   reachable (a channel or workspace default). An unreachable agent has no
   live gate to defend, so any member may configure it.

Both checks live here rather than at the call sites so a newly-wired
mutating action inherits them by routing through this one helper.
"""

from __future__ import annotations

import uuid

from daimon.adapters.slack.admin import resolve_is_admin
from daimon.adapters.slack.runtime import SlackRuntime
from daimon.core.defaults.ma_index import find_agent_by_daimon_tag
from daimon.core.defaults.metadata import MA_METADATA_KEY_MANAGED
from daimon.core.stores.scoped_config_read import is_agent_reachable_in_tenant
from slack_sdk.web.async_client import AsyncWebClient

__all__ = ["refuse_if_reachable_and_not_admin"]

_SEEDED_AGENT_MESSAGE = (
    ":lock: This is the workspace's built-in agent, so its setup cannot be changed "
    "from here — not even by an admin. Fork it to get an editable copy you own."
)


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
    """Refuse a spec-touching action against a built-in or reachable agent.

    Returns ``True`` when the caller must return early (refused); ``False``
    when the caller should proceed.

    A defaults-managed agent is refused first and unconditionally, admins
    included, so that check precedes the admin short-circuit. Otherwise an
    admin caller is never refused and never pays for a DB read. A non-admin
    caller is refused only when the target agent is currently reachable
    (scoped as a channel or workspace default); an unreachable agent has no
    live gate to defend, so any member may still configure it.

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
    # Defaults-managed agents are off-limits to everyone, admins included, and
    # this check therefore runs BEFORE the admin short-circuit. A panel edit
    # never stamps the reconciler's spec hash, so an edited seeded agent would
    # be skipped by the hash short-circuit on every subsequent reconcile and
    # drift from the shipped defaults permanently. Forking is the editable path.
    # Read fresh from the roster rather than from anything the rendered view or
    # private_metadata carried, matching this module's re-resolve discipline.
    seeded = await find_agent_by_daimon_tag(runtime.anthropic, tenant_id=tenant_id, name=agent_name)
    if seeded is not None and seeded.metadata.get(MA_METADATA_KEY_MANAGED) == "true":
        await web_client.chat_postEphemeral(  # pyright: ignore[reportUnknownMemberType]
            channel=channel_id or user_id,
            user=user_id,
            text=_SEEDED_AGENT_MESSAGE,
        )
        return True

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
