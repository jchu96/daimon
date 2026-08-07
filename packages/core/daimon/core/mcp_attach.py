"""Attach an external MCP server to an MA agent's spec.

Two adapters need this and neither may import the other: the MCP tool
``attach_mcp_server`` and the Discord credential modal, which must attach the
server it has just written a vault credential for. Before this module existed
only the MCP tool could do it, so the credential flow stored a token for a
server the agent was never told about and reported success anyway.

MA rejects an agent whose ``mcp_servers`` entries are not each referenced by a
matching ``mcp_toolset`` in ``tools``, so the two lists must move together —
which is the whole reason this is one function rather than a caller-assembled
pair of updates.
"""

from __future__ import annotations

from typing import Any, Final, cast

from anthropic import AsyncAnthropic
from anthropic.types.beta import BetaManagedAgentsAgent
from anthropic.types.beta.agent_create_params import Tool
from anthropic.types.beta.beta_managed_agents_url_mcp_server_params import (
    BetaManagedAgentsURLMCPServerParams,
)
from daimon.core.ma import update_agent_with_version_retry

DEFAULT_MCP_TOOLSET_CONFIG: Final[dict[str, Any]] = {
    "permission_policy": {"type": "always_allow"},
}


def _ma_tool_to_param(tool: Any) -> Tool:  # pyright: ignore[reportExplicitAny]
    """Dump an MA response Tool to a Params dict suitable for the SDK update body."""
    return cast(Tool, tool.model_dump(mode="json", exclude_none=True))


def build_attached_spec(
    agent: BetaManagedAgentsAgent, *, server_name: str, url: str
) -> tuple[list[BetaManagedAgentsURLMCPServerParams], list[Tool]]:
    """Return ``(mcp_servers, tools)`` with ``server_name`` attached at ``url``.

    Pure. An entry with the same ``server_name`` is replaced (last-write-wins)
    and every other server, toolset, and tool is preserved. The matching
    ``mcp_toolset`` is appended only when absent, so re-attaching an existing
    server does not accumulate duplicate toolsets.

    Takes the agent it should compute against as an argument rather than
    reading one, so a version-retry can recompute from the freshly-retrieved
    agent instead of a stale read.
    """
    servers: list[BetaManagedAgentsURLMCPServerParams] = [
        {"name": s.name, "type": "url", "url": s.url}
        for s in (agent.mcp_servers or [])
        if s.name != server_name
    ]
    servers.append({"name": server_name, "type": "url", "url": url})

    tools: list[Tool] = [_ma_tool_to_param(t) for t in agent.tools]
    if not any(
        t.get("type") == "mcp_toolset" and t.get("mcp_server_name") == server_name for t in tools
    ):
        tools.append(
            cast(
                Tool,
                {
                    "type": "mcp_toolset",
                    "mcp_server_name": server_name,
                    "default_config": DEFAULT_MCP_TOOLSET_CONFIG,
                },
            )
        )
    return servers, tools


async def attach_mcp_server_to_agent(
    client: AsyncAnthropic, agent_id: str, *, server_name: str, url: str
) -> BetaManagedAgentsAgent:
    """Attach ``server_name`` at ``url`` to ``agent_id``, preserving the rest.

    Retrieves the agent, recomputes both lists from that fresh read, and
    updates. ``anthropic.ConflictError`` propagates after the single retry
    ``update_agent_with_version_retry`` performs — callers at an adapter
    boundary decide how to surface it.

    Callers are responsible for any policy gate (reserved-server rejection,
    admin checks). This function performs the write and nothing else.
    """

    async def _apply(fresh: BetaManagedAgentsAgent) -> BetaManagedAgentsAgent:
        servers, tools = build_attached_spec(fresh, server_name=server_name, url=url)
        return await client.beta.agents.update(
            fresh.id, version=fresh.version, mcp_servers=servers, tools=tools
        )

    return await update_agent_with_version_retry(client, agent_id, _apply)
