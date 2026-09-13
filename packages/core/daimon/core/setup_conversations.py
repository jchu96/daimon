"""Resolve setup identities without changing routing or starting a session."""

from __future__ import annotations

import uuid

from anthropic import APIStatusError, AsyncAnthropic
from anthropic.types.beta import BetaManagedAgentsAgent
from daimon.core.defaults.ma_index import find_agents_by_daimon_tag
from daimon.core.defaults.metadata import (
    MA_METADATA_KEY_MANAGED,
    MA_METADATA_KEY_NAME,
    MA_METADATA_KEY_TENANT,
)
from daimon.core.errors import DaimonError


async def get_setup_agent(
    anthropic: AsyncAnthropic, *, tenant_id: uuid.UUID, ma_agent_id: str
) -> BetaManagedAgentsAgent:
    """Retrieve the exact live identity; never substitute an agent with the same name."""
    try:
        agent = await anthropic.beta.agents.retrieve(ma_agent_id)
    except APIStatusError as error:
        if error.status_code in (400, 404):
            raise DaimonError(
                "That agent no longer exists. Choose another agent for setup."
            ) from error
        raise
    if agent.archived_at is not None or agent.metadata.get(MA_METADATA_KEY_TENANT) != str(
        tenant_id
    ):
        raise DaimonError(
            "That agent is no longer available in this workspace. Choose another agent."
        )
    return agent


async def get_setup_responder(
    anthropic: AsyncAnthropic, *, tenant_id: uuid.UUID, ma_agent_id: str
) -> BetaManagedAgentsAgent:
    try:
        responder = await get_setup_agent(anthropic, tenant_id=tenant_id, ma_agent_id=ma_agent_id)
    except DaimonError as error:
        raise DaimonError(
            "This setup conversation's Daimon responder is missing. "
            "Ask the operator to restore it, then open a new setup conversation."
        ) from error
    if (
        responder.metadata.get(MA_METADATA_KEY_MANAGED) != "true"
        or responder.metadata.get(MA_METADATA_KEY_NAME) != "daimon"
    ):
        raise DaimonError(
            "This conversation's responder is no longer the built-in Daimon. "
            "Open a new setup conversation."
        )
    return responder


async def resolve_setup_agents(
    anthropic: AsyncAnthropic,
    *,
    tenant_id: uuid.UUID,
    target_ma_agent_id: str | None = None,
) -> tuple[BetaManagedAgentsAgent, BetaManagedAgentsAgent | None]:
    responders = [
        agent
        for agent in await find_agents_by_daimon_tag(anthropic, tenant_id=tenant_id, name="daimon")
        if agent.metadata.get(MA_METADATA_KEY_MANAGED) == "true"
    ]
    if len(responders) != 1:
        raise DaimonError(
            "The built-in Daimon is unavailable. Ask the operator to restore it, then retry setup."
        )
    target = None
    if target_ma_agent_id is not None:
        target = await get_setup_agent(
            anthropic, tenant_id=tenant_id, ma_agent_id=target_ma_agent_id
        )
    return responders[0], target


def build_setup_opener(
    *,
    target_name: str | None,
    opener_mention: str,
    bot_mention: str,
    has_repo: bool,
    has_external_connection: bool,
    can_customize: bool,
) -> str:
    if target_name is None:
        return (
            f"Setup opened by {opener_mention}. I'm Daimon. Which agent would you like to set up?\n"
            f"Mention {bot_mention} when replying here."
        )
    examples: list[str] = []
    if not has_repo:
        examples.append("set a working repo")
    if not has_external_connection:
        examples.append("connect an external service")
    if can_customize:
        examples.append("change its instructions or model")
    example_text = f" You can ask me to {', '.join(examples)}." if examples else ""
    return (
        f"Set up {target_name} · opened by {opener_mention}\n"
        f"I'm Daimon. What would you like to change about {target_name}?{example_text}\n"
        f"Mention {bot_mention} when replying here. Use private forms for keys and tokens."
    )
