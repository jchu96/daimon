"""Read-path helpers for the /agent-setup panel.

Shell module: performs real I/O (DB reads + MA agent list). Mirrors the
shape of routines_panel/read.py — async functions taking session + anthropic +
keyword tenant_id, returning view-model tuples. No try/except (propagate to
the actions.py/submit.py boundary).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable

from anthropic import AsyncAnthropic
from daimon.adapters.slack.agent_setup.state import RosterEntry
from daimon.adapters.slack.runtime import SlackRuntime
from daimon.core.agent_details import AgentDetails, GitHubDeploymentFacts, load_agent_details
from daimon.core.answering_map import AnsweringMap, load_answering_map
from daimon.core.defaults.ma_index import list_agents_by_tenant
from daimon.core.defaults.metadata import MA_METADATA_KEY_NAME
from daimon.core.defaults.provisioning import derive_guild_account_uuid
from daimon.core.ma_identity import derive_agent_uuid
from daimon.core.roster import Roster, load_roster
from daimon.core.scope import (
    ChannelConfigRow,
    ChannelScopeRef,
    DeploymentDefault,
    ScopeRef,
    TenantConfigRow,
    TenantScopeRef,
)
from daimon.core.stores.agent_files import list_agent_files
from daimon.core.stores.identity import get_slack_principal_for_account
from daimon.core.stores.scoped_config_read import get_scope, list_propagations_for_tenant
from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "coding_tools_available",
    "github_facts",
    "load_agent_channel_ids",
    "load_panel_answering_map",
    "load_panel_details",
    "load_panel_roster",
    "load_scope_hint",
    "load_section_data",
    "load_tenant_roster",
    "public_mcp_url",
    "resolve_attributions",
]

_ROSTER_CAP = 25


def github_facts(runtime: SlackRuntime) -> GitHubDeploymentFacts:
    """Read this deployment's two GitHub facts off its settings.

    The core assembler never touches `Settings` — it cannot resolve a token,
    by construction — so the adapter answers "is there an operator fallback
    token" and "is a GitHub App configured" and hands both over as plain
    booleans. The App counts as configured only with both halves of its
    identity present; an app id without a private key mints nothing.
    """
    github = runtime.settings.github
    return GitHubDeploymentFacts(
        has_fallback_pat=github.fallback_pat is not None,
        app_configured=github.app_id is not None and github.app_private_key is not None,
    )


def public_mcp_url(runtime: SlackRuntime) -> str | None:
    """This deployment's own MCP endpoint, or None when it runs without one."""
    url = runtime.settings.mcp.public_url
    return str(url) if url is not None else None


def coding_tools_available(runtime: SlackRuntime) -> bool:
    """Whether a coding-tool token could actually be minted and used.

    Both halves are needed: the URL the person's client connects to, and the
    secret the token is signed with. With either missing the panel says so
    instead of offering a button that cannot produce a working connection.
    """
    return public_mcp_url(runtime) is not None and runtime.settings.mcp.jwt_secret is not None


async def load_panel_roster(
    session: AsyncSession,
    anthropic: AsyncAnthropic,
    *,
    tenant_id: uuid.UUID,
    channel_id: str | None,
    thread_id: str | None,
    default: DeploymentDefault,
) -> Roster:
    """The tenant's agents, answering-here first, for the Agents view.

    `channel_id` is None only where the caller has no channel — a DM, or a
    payload that carried none. The roster still lists every agent; nothing is
    marked as answering here, because there is no here.
    """
    return await load_roster(
        session,
        anthropic,
        tenant_id=tenant_id,
        platform="slack",
        channel_id=channel_id,
        thread_id=thread_id,
        default=default,
    )


async def load_panel_details(
    session: AsyncSession,
    anthropic: AsyncAnthropic,
    runtime: SlackRuntime,
    *,
    tenant_id: uuid.UUID,
    roster: Roster,
    agent_name: str,
    channel_id: str,
    thread_id: str | None,
    is_admin: bool,
) -> AgentDetails | None:
    """Everything the Details view shows for `agent_name`, or None.

    The roster in hand is the name→MA id map, so a click carrying a name that
    is no longer in this tenant's roster resolves to None here rather than
    reaching MA with an id the caller supplied. `channel_label` is the Slack
    channel mention, so the routing sentence the core builds reads as a link
    in the modal instead of a raw id.
    """
    match = next((row for row in roster.rows if row.name == agent_name), None)
    if match is None:
        return None
    return await load_agent_details(
        session,
        anthropic,
        tenant_id=tenant_id,
        ma_agent_id=match.ma_agent_id,
        platform="slack",
        channel_id=channel_id,
        thread_id=thread_id,
        deployment_default=runtime.deployment_default,
        github=github_facts(runtime),
        public_mcp_url=public_mcp_url(runtime),
        is_admin=is_admin,
        channel_label=f"<#{channel_id}>",
    )


async def load_panel_answering_map(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    default: DeploymentDefault,
) -> AnsweringMap:
    """Every tier of this workspace's routing, for the Who-answers-where view."""
    return await load_answering_map(session, tenant_id=tenant_id, platform="slack", default=default)


async def resolve_attributions(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    account_ids: Iterable[uuid.UUID],
) -> dict[uuid.UUID, str]:
    """Map each account that has a Slack identity here to its `<@U…>` mention.

    An account with no Slack principal — a CLI-only actor, or someone who
    acted from Discord — is left out rather than rendered as a bare uuid, and
    the workspace's own stamp account is skipped outright: it is how a seeded
    agent is marked as belonging to the install, not a person who made it, so
    "made by" must never be attached to it.
    """
    workspace_stamp = derive_guild_account_uuid(tenant_id)
    resolved: dict[uuid.UUID, str] = {}
    for account_id in dict.fromkeys(account_ids):
        if account_id == workspace_stamp:
            continue
        external_id = await get_slack_principal_for_account(session, account_id=account_id)
        if external_id:
            resolved[account_id] = f"<@{external_id}>"
    return resolved


async def load_tenant_roster(
    session: AsyncSession,
    anthropic: AsyncAnthropic,
    *,
    tenant_id: uuid.UUID,
) -> tuple[list[RosterEntry], int]:
    """Fetch all non-archived MA agents in the tenant; sort, cap at 25.

    Returns ``(entries[:25], over_cap_count)``. The cap mirrors Slack
    static_select's 25-option limit (Structural Guarantee #4).

    Args:
        session:   Async DB session (unused — kept for signature parity with
                   other read.py functions so callers share the same DI bundle).
        anthropic: Injected ``AsyncAnthropic`` client.
        tenant_id: Slack workspace tenant UUID.

    Returns:
        Tuple of ``(entries[:25], over_cap_count)``.
    """
    agents = await list_agents_by_tenant(anthropic, tenant_id=tenant_id)
    entries: list[RosterEntry] = []
    for agent in agents:
        name: str | None = agent.metadata.get(MA_METADATA_KEY_NAME)  # type: ignore[assignment]
        if name is None:
            name = agent.id
        entries.append(
            RosterEntry(
                agent_name=name,
                model_id=agent.model.id,
                ma_agent_id=str(agent.id),
            )
        )
    entries.sort(key=lambda e: e.agent_name.lower())
    over_cap = max(0, len(entries) - _ROSTER_CAP)
    return entries[:_ROSTER_CAP], over_cap


async def load_section_data(
    session: AsyncSession,
    anthropic: AsyncAnthropic,
    *,
    tenant_id: uuid.UUID,
    agent_name: str,
    section: str,
) -> object:
    """Load per-section data for the L2 agent-detail view.

    Returns different types per ``section``:

    - ``"secrets"``: ``list[str]`` — key NAMES only (values NEVER returned;
      secret hygiene invariant enforced here at the boundary).
    - ``"agent"``: ``dict[str, str | None]`` with ``model_id`` and
      ``system_prompt`` keys (for preview / pre-fill in the edit modal).
    - ``"skills"`` / ``"mcps"``: ``list[str]`` of attached skill or MCP names.

    Args:
        session:    Async DB session.
        anthropic:  Injected ``AsyncAnthropic`` client.
        tenant_id:  Slack workspace tenant UUID.
        agent_name: Daimon-tag agent name (the MA metadata ``daimon_name``).
        section:    One of ``"secrets"``, ``"agent"``, ``"skills"``, ``"mcps"``.

    Returns:
        Section-specific view-model value (see above).
    """
    if section == "secrets":
        # Key NAMES only — values must never leave this read layer.
        # Look up all tenant agents to find the MA agent ID for this name.
        agents = await list_agents_by_tenant(anthropic, tenant_id=tenant_id)
        ma_agent_id: str | None = None
        for agent in agents:
            tag_name: str | None = agent.metadata.get(MA_METADATA_KEY_NAME)  # type: ignore[assignment]
            if tag_name == agent_name:
                ma_agent_id = agent.id
                break
        if ma_agent_id is None:
            return []
        agent_uuid = derive_agent_uuid(tenant_id=tenant_id, ma_agent_id=ma_agent_id)
        files = await list_agent_files(session, tenant_id=tenant_id, agent_id=agent_uuid)
        # Return KEY names only — content (the secret value) is deliberately dropped.
        return [f.key for f in files]

    if section == "agent":
        agents = await list_agents_by_tenant(anthropic, tenant_id=tenant_id)
        for agent in agents:
            tag_name = agent.metadata.get(MA_METADATA_KEY_NAME)  # type: ignore[assignment]
            if tag_name == agent_name:
                return {"model_id": agent.model.id, "system_prompt": agent.system}
        return {"model_id": None, "system_prompt": None}

    if section == "skills":
        agents = await list_agents_by_tenant(anthropic, tenant_id=tenant_id)
        for agent in agents:
            tag_name = agent.metadata.get(MA_METADATA_KEY_NAME)  # type: ignore[assignment]
            if tag_name == agent_name:
                return [sk.skill_id for sk in agent.skills]
        return []

    if section == "mcps":
        agents = await list_agents_by_tenant(anthropic, tenant_id=tenant_id)
        for agent in agents:
            tag_name = agent.metadata.get(MA_METADATA_KEY_NAME)  # type: ignore[assignment]
            if tag_name == agent_name:
                return [s.name for s in agent.mcp_servers]
        return []

    return {}


async def load_scope_hint(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    agent_name: str,
    channel_id: str,
) -> str:
    """Produce the scope-hint string per the UI-SPEC Copywriting Contract.

    Checks whether this agent has a propagated default at the channel scope
    first, then falls back to the workspace (tenant) scope.

    Returns one of:
    - ``:globe_with_meridians: Set for *Whole workspace*`` — tenant-scope hit
    - ``:hash: Set for *#{channel_id}*`` — channel-scope hit
    - ``_(no default set for this agent)_`` — no propagation seeded

    Args:
        session:    Async DB session.
        tenant_id:  Slack workspace tenant UUID.
        agent_name: Daimon-tag agent name to check against ``agent_name`` column.
        channel_id: Slack channel ID for the channel-scope check.
    """
    # Check channel scope first (more specific wins for the hint display).
    channel_scope: ScopeRef = ChannelScopeRef(tenant_id=tenant_id, channel_id=channel_id)
    channel_row = await get_scope(session, scope=channel_scope)
    if isinstance(channel_row, ChannelConfigRow) and channel_row.agent_name == agent_name:
        return f":hash: Set for *#{channel_id}*"

    # Fall back to workspace (tenant) scope.
    tenant_scope: ScopeRef = TenantScopeRef(tenant_id=tenant_id)
    tenant_row = await get_scope(session, scope=tenant_scope)
    if isinstance(tenant_row, TenantConfigRow) and tenant_row.agent_name == agent_name:
        return ":globe_with_meridians: Set for *Whole workspace*"

    return "_(no default set for this agent)_"


async def load_agent_channel_ids(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    agent_name: str,
) -> list[str]:
    """Return all channel IDs where this agent has a propagated channel-scope default.

    Delegates to ``list_propagations_for_tenant`` (the core store) and filters by
    ``agent_name``.  Returns an empty list when the agent has no channel defaults.

    Args:
        session:    Async DB session.
        tenant_id:  Slack workspace tenant UUID.
        agent_name: Daimon-tag agent name to match against ``channel_config.agent_name``.

    Returns:
        List of Slack channel IDs that carry a default for this agent.
    """
    _, channel_rows = await list_propagations_for_tenant(session, tenant_id=tenant_id)
    return [row.channel_id for row in channel_rows if row.agent_name == agent_name]
