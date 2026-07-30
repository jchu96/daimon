"""Click-time authorization gate for the panel's spec-mutating writes.

`EditView`'s spec controls are disabled when the render-time snapshot says a
control shouldn't be usable — that snapshot is a rendering hint, not a
boundary. This module is the boundary: every callback that would write an
agent's prompt, model, skills, or MCP servers calls
`refuse_if_reachable_and_not_admin` first, which re-derives both admin status
and reachability from live state rather than trusting anything the view was
constructed with, so a stale open panel or a demoted admin cannot write a
change the rendered panel had already forbidden.
"""

from __future__ import annotations

import structlog
from daimon.adapters.discord.agent_setup.state import RosterEntry
from daimon.adapters.discord.agent_setup.tenant import resolve_tenant_for_panel
from daimon.adapters.discord.checks import is_guild_admin
from daimon.adapters.discord.runtime import DiscordRuntime
from daimon.core.stores.scoped_config_read import is_agent_reachable_in_tenant

import discord

log = structlog.get_logger()

_SYSTEM_AGENT_MESSAGE = (
    "This agent ships with the deployment and is managed from the repo defaults. "
    "Fork it to make an editable copy."
)
_REACHABLE_AGENT_MESSAGE = (
    "This agent is currently the default for this channel or the server, so "
    "changing its setup needs Manage Server. Fork it to make an editable copy."
)


async def _send_ephemeral(interaction: discord.Interaction, content: str) -> None:  # type: ignore[type-arg]  # discord.Interaction default generic arg; only response/followup are used
    if interaction.response.is_done():
        await interaction.followup.send(content, ephemeral=True)
    else:
        await interaction.response.send_message(content, ephemeral=True)


async def refuse_if_reachable_and_not_admin(
    interaction: discord.Interaction,  # type: ignore[type-arg]  # discord.Interaction default generic arg; only response/followup/guild_id are used
    *,
    runtime: DiscordRuntime,
    entry: RosterEntry | None,
) -> bool:
    """Click-time re-check shared by every panel callback that writes an agent spec.

    Returns True when the caller must return immediately (the write is
    refused); False when the write may proceed. Order, each short-circuiting:

    1. No target selected -> refuse silently (belt and braces; every call
       site already narrows `entry` past its own `selected is None` check).
    2. The target is a system agent -> refuse, unconditionally, even for an
       admin (a panel edit never stamps the seed's spec hash, so an admin
       bypass here would reintroduce silent drift against the defaults).
    3. A live guild admin -> allow, without reading the database.
    4. Otherwise, read reachability fresh from the database and refuse when
       the target currently resolves for some channel or the workspace.
    """
    if entry is None:
        log.debug("agent_setup.authz.no_target")
        return True
    if entry.is_system:
        await _send_ephemeral(interaction, _SYSTEM_AGENT_MESSAGE)
        return True
    if is_guild_admin(interaction):  # pyright: ignore[reportArgumentType]  # discord.Interaction vs Interaction[commands.Bot]; is_guild_admin only reads user/guild
        return False
    tenant_id = await resolve_tenant_for_panel(runtime, interaction)
    async with runtime.sessionmaker() as session:
        reachable = await is_agent_reachable_in_tenant(
            session,
            tenant_id=tenant_id,
            agent_name=entry.name,
            default=runtime.deployment_default,
        )
    if reachable:
        await _send_ephemeral(interaction, _REACHABLE_AGENT_MESSAGE)
        return True
    return False
