"""Click-time authorization for the chat-initiated repo bind.

This module holds the shell halves of the chat-initiated repo bind: the
click-time gate below, and — added by the plan that follows this one — the
clone-credential resolution the write itself needs.

`refuse_if_shared_and_not_admin_for_request` re-derives the panel gate's
(`refuse_if_shared_and_not_admin`) decision from primitives rather than
reusing it: a `credential_requests` row carries only a derived agent uuid,
not the roster-entry value the panel gate expects, so there is nothing to
hand it. The branch ORDER below is load-bearing and mirrors the panel
gate's order exactly — reordering it changes who can bind a repo to what.

The gate is called from two different response states and must work
correctly from both:

1. Before anything is acked, as a pre-filter ahead of opening the modal.
   The admin branch precedes every MA and database read, so an admin still
   opens the modal well inside the interaction budget. A refused member pays
   the MA and database reads before any response — which is the point of
   the pre-filter: a member who was always going to be refused is never
   asked to paste a token into a form that gets thrown away.
2. After the caller acks with a thinking response, immediately before the
   single-use consume — this is the actual authorization boundary. The
   pre-filter above does not replace it.

Because of (1), this gate must **never** ack the interaction on its own —
opening a modal is only possible from an un-acked interaction, so the
pre-filter has to finish its reads and either fall through or answer on
whichever half of `interaction.response.is_done()` is currently live,
without itself changing that state. That is why every ephemeral send below
routes through that same split rather than a fixed response call, and why
this module works unmodified from both call sites.

The pre-filter's caller bounds this gate with a timeout and opens the modal
anyway if it expires. That is sound only because call site (2) repeats every
check below before the write, so it is the real boundary — it is not a
"retry recovers it" situation: the underlying read paginates the operator's
entire MA org, so a slow read costs the same on every click, and a retry
fails identically. The timeout and its full rationale live with the
pre-filter itself.
"""

from __future__ import annotations

import uuid
from typing import Final

import structlog
from anthropic import AsyncAnthropic
from anthropic.types.beta import BetaManagedAgentsAgent
from daimon.adapters.discord.checks import is_guild_admin
from daimon.adapters.discord.runtime import DiscordRuntime
from daimon.core.defaults.ma_index import list_agents_by_tenant
from daimon.core.defaults.metadata import MA_METADATA_KEY_MANAGED
from daimon.core.ma_identity import derive_agent_uuid, derive_tenant_uuid
from daimon.core.stores.scoped_config_read import is_agent_reachable_in_tenant

import discord

log = structlog.get_logger()

# Kept character-identical to the panel gate's `_SHARED_AGENT_MESSAGE`
# (agent_setup's authz module) on purpose — a test asserts the two strings
# are equal, so a future edit to one without the other goes red rather than
# drifting silently.
_SHARED_AGENT_MESSAGE: Final[str] = (
    "This agent is shared — it either ships with the deployment or is the "
    "current default for this channel or the server — so changing its repo or "
    "environment variables needs Manage Server. Fork it to make an editable "
    "copy; the fork starts with no environment variables of its own."
)


_AGENT_GONE_MESSAGE: Final[str] = (
    "That agent no longer exists — ask again and a fresh request will be posted."
)

_WRONG_GUILD_MESSAGE: Final[str] = (
    "This request isn't for this server — ask again from the server it was posted in."
)


async def resolve_ma_agent_for_uuid(
    client: AsyncAnthropic, *, tenant_id: uuid.UUID, agent_id: uuid.UUID
) -> BetaManagedAgentsAgent | None:
    """Reverse-resolve a derived agent uuid back to the MA agent it came from.

    `derive_agent_uuid` is a one-way `uuid5` over `(tenant_id, ma_agent_id)`,
    so there is no direct lookup from the derived uuid a `credential_requests`
    row carries back to the MA agent id. The only available resolution is
    forward: list every non-archived MA agent tagged with this tenant and
    recompute the derived uuid for each until one matches. Returns `None`
    when nothing matches — the agent was archived or deleted since the row
    was minted.
    """
    agents = await list_agents_by_tenant(client, tenant_id=tenant_id)
    for agent in agents:
        if derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=str(agent.id)) == agent_id:
            return agent
    return None


async def _send_ephemeral(interaction: discord.Interaction, content: str) -> None:  # type: ignore[type-arg]  # discord.Interaction default generic arg; only response/followup are used
    if interaction.response.is_done():
        await interaction.followup.send(content, ephemeral=True)
    else:
        await interaction.response.send_message(content, ephemeral=True)


async def refuse_if_shared_and_not_admin_for_request(
    interaction: discord.Interaction,  # type: ignore[type-arg]  # discord.Interaction default generic arg; only response/followup/guild_id/user are used
    *,
    runtime: DiscordRuntime,
    tenant_id: uuid.UUID,
    agent_id: uuid.UUID,
) -> bool:
    """Click-time re-check for the chat-initiated repo-bind write.

    Returns True when the caller must return immediately (the write is
    refused); False when the write may proceed. Order, each short-circuiting:

    1. Tenant re-derivation. Re-derive the tenant from the live interaction's
       guild and compare it against the `tenant_id` argument the row was
       minted with. This is pure and costs no I/O, so it precedes even the
       admin check — without it, the tenant half of the decision would be
       taken on trust from a row written at some earlier time rather than
       re-derived from live state. Defense in depth: the posted button lives
       in the channel the mint named, so a cross-guild click is not
       reachable through normal use; this guard exists so it stays
       unreachable by construction rather than by luck.
    2. A live guild admin (`is_guild_admin`) -> allow, before any MA call and
       before any database read. This ordering is what keeps an admin able
       to bind a repo to the seeded agent — the first-run onboarding step
       this whole path exists to enable.
    3. Resolve the MA agent the row's derived uuid points at. No match ->
       refuse, fail closed (the agent may have been archived or deleted
       since the row was minted).
    4. The resolved agent is defaults-managed
       (`metadata.get(MA_METADATA_KEY_MANAGED) == "true"`) -> refuse; it
       belongs to the deployment and every member of the install shares it.
    5. Otherwise, read reachability fresh from the database and refuse when
       the agent currently resolves for some channel or the workspace.
    6. Otherwise allow.

    Steps 2, 4, and 5 are `refuse_if_shared_and_not_admin`'s steps 2, 3, and
    4 in the same order, driven off primitives instead of a roster entry.

    The two shared-agent refusals (4 and 5) carry the same message on
    purpose, exactly as in the panel gate: which limb fired is not
    actionable, and the fix is the same either way — ask an admin, or bind
    to a private agent instead.

    This function never acks the interaction — see the module docstring.
    """
    if interaction.guild_id is None or (
        derive_tenant_uuid(platform="discord", workspace_id=str(interaction.guild_id)) != tenant_id
    ):
        log.warning(
            "credential_repo_bind.wrong_guild",
            request_tenant_id=str(tenant_id),
            interaction_guild_id=interaction.guild_id,
        )
        await _send_ephemeral(interaction, _WRONG_GUILD_MESSAGE)
        return True
    if is_guild_admin(interaction):  # pyright: ignore[reportArgumentType]  # discord.Interaction vs Interaction[commands.Bot]; is_guild_admin only reads user/guild
        return False
    agent = await resolve_ma_agent_for_uuid(
        runtime.anthropic, tenant_id=tenant_id, agent_id=agent_id
    )
    if agent is None:
        log.warning(
            "credential_repo_bind.agent_gone",
            tenant_id=str(tenant_id),
            agent_id=str(agent_id),
        )
        await _send_ephemeral(interaction, _AGENT_GONE_MESSAGE)
        return True
    if agent.metadata.get(MA_METADATA_KEY_MANAGED) == "true":
        await _send_ephemeral(interaction, _SHARED_AGENT_MESSAGE)
        return True
    async with runtime.sessionmaker() as session:
        reachable = await is_agent_reachable_in_tenant(
            session,
            tenant_id=tenant_id,
            agent_name=agent.name,
            default=runtime.deployment_default,
        )
    if reachable:
        await _send_ephemeral(interaction, _SHARED_AGENT_MESSAGE)
        return True
    return False
