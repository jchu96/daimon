"""Tests for the MCP-server attach spec computation.

The defect these exist for (#49) was a missing step, not a wrong one: the
credential modal stored an auth token for a server it never added to the
agent, so the agent held a credential for something it could not reach. The
assertions below are therefore about what the attach must PRESERVE and what it
must ALWAYS produce — a returned server list is worthless if its matching
``mcp_toolset`` is absent, because MA rejects that spec outright.
"""

from __future__ import annotations

from datetime import UTC, datetime

from anthropic.types.beta import BetaManagedAgentsAgent
from anthropic.types.beta.beta_managed_agents_agent_toolset20260401 import (
    BetaManagedAgentsAgentToolset20260401,
)
from anthropic.types.beta.beta_managed_agents_agent_toolset_default_config import (
    BetaManagedAgentsAgentToolsetDefaultConfig,
)
from anthropic.types.beta.beta_managed_agents_always_allow_policy import (
    BetaManagedAgentsAlwaysAllowPolicy,
)
from anthropic.types.beta.beta_managed_agents_custom_tool import BetaManagedAgentsCustomTool
from anthropic.types.beta.beta_managed_agents_mcp_toolset import BetaManagedAgentsMCPToolset
from anthropic.types.beta.beta_managed_agents_mcp_toolset_default_config import (
    BetaManagedAgentsMCPToolsetDefaultConfig,
)
from anthropic.types.beta.beta_managed_agents_model_config import BetaManagedAgentsModelConfig
from daimon.core.mcp_attach import build_attached_spec


def _agent_toolset() -> BetaManagedAgentsAgentToolset20260401:
    return BetaManagedAgentsAgentToolset20260401(
        type="agent_toolset_20260401",
        configs=[],
        default_config=BetaManagedAgentsAgentToolsetDefaultConfig(
            enabled=True,
            permission_policy=BetaManagedAgentsAlwaysAllowPolicy(type="always_allow"),
        ),
    )


def _mcp_toolset(server_name: str) -> BetaManagedAgentsMCPToolset:
    return BetaManagedAgentsMCPToolset(
        type="mcp_toolset",
        mcp_server_name=server_name,
        configs=[],
        default_config=BetaManagedAgentsMCPToolsetDefaultConfig(
            enabled=True,
            permission_policy=BetaManagedAgentsAlwaysAllowPolicy(type="always_allow"),
        ),
    )


def _agent(
    *,
    mcp_servers: list[dict[str, str]],
    tools: list[object],
) -> BetaManagedAgentsAgent:
    now = datetime(2026, 8, 7, tzinfo=UTC)
    return BetaManagedAgentsAgent(
        id="agent_01Test",
        archived_at=None,
        created_at=now,
        description=None,
        mcp_servers=[{"name": s["name"], "type": "url", "url": s["url"]} for s in mcp_servers],  # pyright: ignore[reportArgumentType]
        metadata={},
        model=BetaManagedAgentsModelConfig(id="claude-sonnet-5"),
        name="test-agent",
        skills=[],
        system=None,
        tools=tools,  # pyright: ignore[reportArgumentType]
        type="agent",
        updated_at=now,
        version=1,
    )


def test_attach_adds_the_server_and_its_matching_toolset() -> None:
    agent = _agent(mcp_servers=[], tools=[_agent_toolset()])

    servers, tools = build_attached_spec(agent, server_name="acme", url="https://acme.test/mcp")

    assert {"name": "acme", "type": "url", "url": "https://acme.test/mcp"} in servers, (
        "the server must be added to mcp_servers"
    )
    assert any(
        t.get("type") == "mcp_toolset" and t.get("mcp_server_name") == "acme" for t in tools
    ), "MA rejects an mcp_servers entry with no matching mcp_toolset, so both must be written"


def test_attach_preserves_the_builtin_daimon_mcp_server_and_toolset() -> None:
    """The reserved daimon-mcp entry must survive — losing it cuts the agent off."""
    agent = _agent(
        mcp_servers=[{"name": "daimon-mcp", "url": "https://daimon.test/mcp"}],
        tools=[_agent_toolset(), _mcp_toolset("daimon-mcp")],
    )

    servers, tools = build_attached_spec(agent, server_name="acme", url="https://acme.test/mcp")

    assert [s["name"] for s in servers] == ["daimon-mcp", "acme"], (
        "the existing server must be preserved alongside the new one"
    )
    assert any(t.get("type") == "agent_toolset_20260401" for t in tools), (
        "the builtin agent toolset must survive"
    )
    assert sum(1 for t in tools if t.get("type") == "mcp_toolset") == 2, (
        "each server keeps its own toolset"
    )


def test_attach_replaces_same_name_server_without_duplicating_its_toolset() -> None:
    """Re-attaching under one name is a URL change, not a second entry."""
    agent = _agent(
        mcp_servers=[{"name": "acme", "url": "https://old.test/mcp"}],
        tools=[_mcp_toolset("acme")],
    )

    servers, tools = build_attached_spec(agent, server_name="acme", url="https://new.test/mcp")

    assert servers == [{"name": "acme", "type": "url", "url": "https://new.test/mcp"}], (
        "same-name attach is last-write-wins on the URL, not an append"
    )
    assert sum(1 for t in tools if t.get("mcp_server_name") == "acme") == 1, (
        "the existing toolset is reused, never duplicated"
    )


def test_attach_preserves_unrelated_custom_tools() -> None:
    agent = _agent(
        mcp_servers=[],
        tools=[
            BetaManagedAgentsCustomTool(
                type="custom",
                name="my_tool",
                description="a tool the agent already had",
                input_schema={"type": "object"},
            ),
            _agent_toolset(),
        ],
    )

    _servers, tools = build_attached_spec(agent, server_name="acme", url="https://acme.test/mcp")

    assert any(t.get("type") == "custom" and t.get("name") == "my_tool" for t in tools), (
        "attaching a server must not drop the agent's other tools"
    )
