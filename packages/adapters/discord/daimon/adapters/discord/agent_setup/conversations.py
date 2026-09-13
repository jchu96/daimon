"""Open a shared public setup thread from a selected agent's panel."""

from __future__ import annotations

import contextlib

import anthropic
import structlog
from daimon.adapters.discord.agent_setup.state import PanelState
from daimon.adapters.discord.agent_setup.tenant import resolve_tenant_for_panel
from daimon.adapters.discord.checks import is_member_guild_admin
from daimon.adapters.discord.errors import generate_request_id, render_error
from daimon.adapters.discord.runtime import DiscordRuntime
from daimon.core.defaults.metadata import MA_METADATA_KEY_MANAGED
from daimon.core.errors import DaimonError
from daimon.core.ma_identity import derive_agent_uuid
from daimon.core.setup_conversations import build_setup_opener, resolve_setup_agents
from daimon.core.stores.agent_repo_binding import get_binding
from daimon.core.stores.thread_agent_bindings import create_binding, update_lifecycle
from sqlalchemy.exc import SQLAlchemyError

import discord

_log = structlog.get_logger()


async def open_setup_conversation(
    interaction: discord.Interaction, *, runtime: DiscordRuntime, state: PanelState
) -> None:
    """Validate identities, persist routing, then present a ready conversation.

    The caller defers before entering. Opening this thread never creates an MA
    session or runs a turn; the opener is deterministic platform content.
    """
    channel = interaction.channel
    if isinstance(channel, discord.Thread):
        channel = channel.parent
    if not isinstance(channel, discord.TextChannel):
        await interaction.followup.send(
            "Setup conversations need a server text channel. Open `/agent-setup` there.",
            ephemeral=True,
        )
        return
    if interaction.client.user is None:
        await interaction.followup.send("The bot is still connecting. Try again.", ephemeral=True)
        return
    thread: discord.Thread | None = None
    try:
        tenant_id = await resolve_tenant_for_panel(runtime, interaction)
        selected = state.selected
        if selected is not None and not selected.ma_agent_id:
            raise DaimonError("That agent is not ready. Reopen `/agent-setup` and try again.")
        responder, target = await resolve_setup_agents(
            runtime.anthropic,
            tenant_id=tenant_id,
            target_ma_agent_id=selected.ma_agent_id if selected else None,
        )
        has_repo = False
        if target is not None:
            async with runtime.sessionmaker() as session:
                has_repo = (
                    await get_binding(
                        session,
                        tenant_id=tenant_id,
                        agent_id=derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=str(target.id)),
                    )
                    is not None
                )
        target_name = selected.name if target is not None and selected else None
        opener = build_setup_opener(
            target_name=target_name,
            opener_mention=interaction.user.mention,
            bot_mention=interaction.client.user.mention,
            has_repo=has_repo,
            has_external_connection=bool(
                target
                and any(
                    server.url.rstrip("/") != (state.default_mcp_url or "").rstrip("/")
                    for server in target.mcp_servers
                )
            ),
            can_customize=bool(
                target
                and target.metadata.get(MA_METADATA_KEY_MANAGED) != "true"
                and (
                    (
                        isinstance(interaction.user, discord.Member)
                        and is_member_guild_admin(
                            interaction.user,
                            guild_owner_id=interaction.guild.owner_id
                            if interaction.guild
                            else None,
                        )
                    )
                    or not state.is_selected_reachable()
                )
            ),
        )
        thread = await channel.create_thread(
            name=f"Set up {target_name or 'an agent'} with Daimon"[:100],
            type=discord.ChannelType.public_thread,
            auto_archive_duration=10080,
            reason="Member opened an agent setup conversation",
        )
        try:
            async with runtime.sessionmaker() as session:
                await create_binding(
                    session,
                    tenant_id=tenant_id,
                    platform="discord",
                    parent_channel_id=str(channel.id),
                    thread_id=str(thread.id),
                    responder_ma_agent_id=str(responder.id),
                    responder_name="Daimon",
                    configuration_target_ma_agent_id=str(target.id) if target else None,
                    configuration_target_name=target_name,
                    creator_account_id=state.account_id,
                )
                await session.commit()
            await thread.send(opener, allowed_mentions=discord.AllowedMentions.none())
        except Exception:
            # This is the platform creation boundary. Never leave a failed
            # opener advertised as ready; retain the original exception.
            try:
                await thread.delete(reason="Setup conversation could not be initialized")
            except discord.HTTPException:
                with contextlib.suppress(discord.HTTPException):
                    await thread.edit(name="Setup failed — please open a new conversation")
            with contextlib.suppress(SQLAlchemyError):
                async with runtime.sessionmaker() as session:
                    await update_lifecycle(
                        session,
                        tenant_id=tenant_id,
                        platform="discord",
                        parent_channel_id=str(channel.id),
                        thread_id=str(thread.id),
                        deleted=True,
                    )
                    await session.commit()
            raise
        await interaction.followup.send(
            f"[Continue setup with Daimon]({thread.jump_url})",
            ephemeral=True,
            allowed_mentions=discord.AllowedMentions.none(),
        )
    except discord.Forbidden:
        _log.exception("setup_conversation.permission_denied")
        await interaction.followup.send(
            "I couldn't create or post in the setup thread here. Ask a server admin to give me "
            "Create Public Threads, Send Messages in Threads, and Manage Threads in this channel, "
            "then try again.",
            ephemeral=True,
        )
    except (DaimonError, anthropic.APIError, discord.HTTPException, SQLAlchemyError) as error:
        request_id = generate_request_id()
        _log.exception("setup_conversation.failed", request_id=request_id)
        await interaction.followup.send(render_error(error, request_id=request_id), ephemeral=True)
