"""Post-ack authorization gates for the legacy /agent-setup editors.

Field-follows-the-gate rule for the next contributor wiring up a mutating
action. Which of the two gates a write routes through follows from whether the
field it writes is part of the agent spec:

- Spec fields -- skills, MCP servers, the model, the system prompt -- route
  through ``refuse_if_reachable_and_not_admin`` (`agent_spec_edit`). That
  refuses a defaults-managed agent unconditionally, admins included, because a
  panel edit never stamps the reconciler's spec hash: the resulting drift would
  survive every later reconcile with no way back. Forking is the editable path.
  Below that, a non-admin is refused whenever the target is currently reachable
  (a channel or workspace default) -- an unreachable agent has no live gate to
  defend, so any member may configure it.

- Per-agent attachments -- the repo binding, the inline token, env-variable
  credentials -- route through ``refuse_if_shared_and_not_admin``
  (`key_replace`). They never enter the agent spec, so the defaults-managed
  absolutism above does not apply to them: an admin binding a repo to the
  workspace's built-in agent is a supported first-run step. They are NOT open
  to non-admins on a shared agent, though, because a repo re-point or a secret
  overwrite changes state every member of the workspace depends on.

A new mutating action routes through exactly one of the two. "No gate" is not
one of the options.

The two open-time behaviours differ on purpose. The edit-repo form is gated
when it is pushed, because that form prompts for a GitHub personal access
token and no such credential should transit for a write that is going to be
refused anyway. The paste-secrets form prompts for nothing on push, so it is
gated at submission only -- the boundary that actually writes.

Both functions are now thin wrappers over ``daimon.adapters.slack.agent_policy``,
which owns the shell half of the decision (live admin status, the target
fetch, the reachability read and the refusal copy) for the panel and the
chat-initiated credential path alike. Their names, signatures and behaviour are
unchanged for the legacy editors that still call them.
"""

from __future__ import annotations

import uuid

from daimon.adapters.slack.agent_policy import refuse_unless_allowed_for_agent_name
from daimon.adapters.slack.runtime import SlackRuntime
from slack_sdk.web.async_client import AsyncWebClient

__all__ = ["refuse_if_reachable_and_not_admin", "refuse_if_shared_and_not_admin"]


async def refuse_if_reachable_and_not_admin(
    runtime: SlackRuntime,
    web_client: AsyncWebClient,
    *,
    tenant_id: uuid.UUID,
    agent_name: str,
    channel_id: str,
    user_id: str,
) -> bool:
    """Refuse a spec-touching action against a built-in or reachable agent.

    Returns ``True`` when the caller must return early (refused); ``False``
    when the caller should proceed.

    Args:
        runtime:     Injected ``SlackRuntime`` (sessionmaker, deployment default).
        web_client:  Per-event ``AsyncWebClient``.
        tenant_id:   Derived from the verified Socket Mode workspace id --
                     never accepted from the interactive payload.
        agent_name:  The target agent's name -- used only as a tenant-scoped
                     lookup key.
        channel_id:  Invoking channel, for the refusal ephemeral.
        user_id:     Invoking user, for the admin check and the ephemeral.
    """
    return await refuse_unless_allowed_for_agent_name(
        runtime,
        web_client,
        operation="agent_spec_edit",
        tenant_id=tenant_id,
        agent_name=agent_name,
        channel_id=channel_id,
        user_id=user_id,
    )


async def refuse_if_shared_and_not_admin(
    runtime: SlackRuntime,
    web_client: AsyncWebClient,
    *,
    tenant_id: uuid.UUID,
    agent_name: str,
    channel_id: str,
    user_id: str,
) -> bool:
    """Refuse an attachment write against a shared agent by a non-admin.

    Returns ``True`` when the caller must return early (refused); ``False``
    when the caller should proceed. The admin short-circuit runs before the
    defaults-managed lookup, which is what keeps an admin able to bind a repo
    to the workspace's built-in agent -- the first-run onboarding step.

    Args:
        runtime:     Injected ``SlackRuntime`` (sessionmaker, deployment default).
        web_client:  Per-event ``AsyncWebClient``.
        tenant_id:   Derived from the verified Socket Mode workspace id --
                     never accepted from the interactive payload.
        agent_name:  The target agent's name -- used only as a tenant-scoped
                     lookup key.
        channel_id:  Invoking channel, for the refusal ephemeral.
        user_id:     Invoking user, for the admin check and the ephemeral.
    """
    return await refuse_unless_allowed_for_agent_name(
        runtime,
        web_client,
        operation="key_replace",
        tenant_id=tenant_id,
        agent_name=agent_name,
        channel_id=channel_id,
        user_id=user_id,
    )
