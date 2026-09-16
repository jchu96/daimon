"""Unit tests for the personal-MCP-server classification and list surgery.

Pure module, so no DB and no fake MA: grants in, server names out, and the
two agent arrays cut in step. The DB read that feeds it lives in
`agent_mcp_credentials` and is covered by `test_agent_mcp_credentials.py`.
"""

from __future__ import annotations

import uuid

from daimon.core.mcp_personal_servers import (
    hidden_mcp_server_names,
    visible_mcp_servers,
    visible_tools,
)
from daimon.core.stores.domain import McpOAuthGrantRow
from daimon.testing.ma_models import ma_agent

_CONNECTED = uuid.UUID("00000000-0000-0000-0000-0000000000aa")
_OTHER = uuid.UUID("00000000-0000-0000-0000-0000000000bb")


def _toolset(server_name: str) -> dict[str, object]:
    """One `mcp_toolset` entry in the shape MA returns it on an agent."""
    return {
        "type": "mcp_toolset",
        "mcp_server_name": server_name,
        "configs": [],
        "default_config": {
            "enabled": True,
            "permission_policy": {"type": "always_allow"},
        },
    }


def test_a_server_only_someone_else_connected_is_hidden() -> None:
    grants = (
        McpOAuthGrantRow(
            account_id=_CONNECTED, server_name="notion", mcp_server_url="https://notion/mcp"
        ),
    )
    assert hidden_mcp_server_names(grants, account_id=_OTHER, shared_server_urls=()) == frozenset(
        {"notion"}
    ), "a caller with no grant of their own cannot authenticate the server"
    assert (
        hidden_mcp_server_names(grants, account_id=_CONNECTED, shared_server_urls=()) == frozenset()
    ), "the person who signed in keeps the server they connected"


def test_nothing_is_hidden_when_nobody_has_signed_in() -> None:
    assert hidden_mcp_server_names((), account_id=_OTHER, shared_server_urls=()) == frozenset(), (
        "an agent with no OAuth grants hides nothing"
    )


def test_a_server_with_an_agent_wide_credential_is_never_hidden() -> None:
    grants = (
        McpOAuthGrantRow(
            account_id=_CONNECTED, server_name="notion", mcp_server_url="https://notion/mcp/"
        ),
    )
    assert (
        hidden_mcp_server_names(
            grants, account_id=_OTHER, shared_server_urls=("https://notion/mcp",)
        )
        == frozenset()
    ), "a token stored on the agent is mirrored into every caller's vault, trailing slash or not"


def test_visible_lists_drop_a_hidden_server_and_its_toolset_together() -> None:
    agent = ma_agent(
        mcp_servers=[
            {"name": "notion", "type": "url", "url": "https://notion/mcp"},
            {"name": "daimon-mcp", "type": "url", "url": "https://daimon/mcp"},
        ],
        tools=[
            _toolset("notion"),
            _toolset("daimon-mcp"),
        ],
    )

    servers = visible_mcp_servers(agent, frozenset({"notion"}))
    tools = visible_tools(agent, frozenset({"notion"}))

    assert [server.name for server in servers] == ["daimon-mcp"], "the hidden server is gone"
    assert len(tools) == 1, "its toolset goes with it — MA rejects one without the other"
    assert not any(getattr(tool, "mcp_server_name", None) == "notion" for tool in tools), (
        "no toolset may reference a server this session does not mount"
    )


def test_visible_lists_are_the_agent_lists_when_nothing_is_hidden() -> None:
    agent = ma_agent(
        mcp_servers=[{"name": "daimon-mcp", "type": "url", "url": "https://daimon/mcp"}],
        tools=[_toolset("daimon-mcp")],
    )

    assert list(visible_mcp_servers(agent, frozenset())) == list(agent.mcp_servers), (
        "an empty hidden set must not reshape the agent's servers"
    )
    assert list(visible_tools(agent, frozenset())) == list(agent.tools), (
        "an empty hidden set must not reshape the agent's tools"
    )
