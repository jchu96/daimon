"""Validate durable credential interaction destinations after a turn has ended."""

from __future__ import annotations

import anthropic
import structlog
from daimon.adapters.discord.runtime import DiscordRuntime
from daimon.core.defaults.ma_index import find_agent_by_derived_uuid
from daimon.core.ma_identity import derive_tenant_uuid
from daimon.core.stores.domain import CredentialRequestRow

import discord


def is_credential_interaction_valid(
    interaction: discord.Interaction, row: CredentialRequestRow
) -> bool:
    """Require the original requester, install, platform and posted card.

    Legacy cards have no origin fields. Their authenticated original Discord
    interaction remains usable until the existing expiry/single-use gate closes.
    """
    if str(interaction.user.id) != row.requester_platform_user_id:
        return False
    if (
        interaction.guild_id is None
        or derive_tenant_uuid(platform="discord", workspace_id=str(interaction.guild_id))
        != row.tenant_id
    ):
        return False
    if row.platform is not None and row.platform != "discord":
        return False
    if row.origin_thread_id is not None and str(interaction.channel_id) != row.origin_thread_id:
        return False
    if row.parent_channel_id is not None:
        channel = interaction.channel
        parent_id = (
            channel.parent_id if isinstance(channel, discord.Thread) else interaction.channel_id
        )
        if str(parent_id) != row.parent_channel_id:
            return False
    # Discord may omit message on modal-submit payloads. The bot-owned modal
    # already holds the row validated at the component click; recheck its
    # requester and location above, and reject a different message if supplied.
    if row.posted_message_id is None:
        return True
    if interaction.message is None:
        return interaction.type is discord.InteractionType.modal_submit
    return str(interaction.message.id) == row.posted_message_id


async def refuse_if_credential_target_unavailable(
    interaction: discord.Interaction, *, runtime: DiscordRuntime, row: CredentialRequestRow
) -> bool:
    """Verify the exact agent before consuming a private form or saving its value."""
    try:
        agent = await find_agent_by_derived_uuid(
            runtime.anthropic, tenant_id=row.tenant_id, agent_id=row.agent_id
        )
    except anthropic.APIError:
        structlog.get_logger().exception(
            "credential_target.lookup_failed", tenant_id=str(row.tenant_id)
        )
        await interaction.followup.send(
            "I couldn't verify this agent. Nothing was saved; please try submitting again.",
            ephemeral=True,
        )
        return True
    if agent is not None:
        return False
    await interaction.followup.send(
        "This request's agent no longer exists. Nothing was saved. "
        "Ask Daimon for a new request for the intended agent.",
        ephemeral=True,
    )
    return True
