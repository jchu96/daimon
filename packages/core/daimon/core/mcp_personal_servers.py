"""Which of an agent's MCP servers a given caller can actually authenticate.

An OAuth sign-in is personal: the grant lands in the connecting person's own
(account, agent) vault, but `attach_mcp_server_to_agent` puts the server on
the agent everyone shares. MA opens every attached server on every turn and
resolves its credential from the vault mounted on that session — the
caller's — so one person connecting Notion left every other person's turn
opening a server they have no token for, failing it with
`mcp_authentication_failed_error`, and carrying the degraded-turn notice
("⚠️ `notion` was unavailable this turn…") under every reply whether or not
the turn ever wanted Notion.

`hidden_mcp_server_names` names the servers to leave out of one caller's
session; `visible_mcp_servers` and `visible_tools` cut them out of the two
arrays MA insists move together — an `mcp_toolset` whose server is gone is
rejected, as is a server no toolset references.

Pure, and deliberately import-free at runtime: `session_snapshot` hashes what
these return, and `stores.domain` imports `session_snapshot`, so a store
import here would close a cycle. The grant rows are read by
`agent_mcp_credentials.resolve_hidden_mcp_server_names` and passed in.

A server whose credential is agent-wide (`agent_mcp_credentials`, mirrored
into every caller's vault at session create) is never personal, even when
somebody also signed in to it personally: everyone can authenticate it.
Server *name* is the join key, because that is what the agent spec and MA's
failure events carry.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from anthropic.types.beta import BetaManagedAgentsAgent
from anthropic.types.beta.beta_managed_agents_agent import Tool as MATool
from anthropic.types.beta.beta_managed_agents_mcp_server_url_definition import (
    BetaManagedAgentsMCPServerURLDefinition,
)
from anthropic.types.beta.beta_managed_agents_mcp_toolset import BetaManagedAgentsMCPToolset

if TYPE_CHECKING:
    from daimon.core.stores.domain import McpOAuthGrantRow

__all__ = ["hidden_mcp_server_names", "visible_mcp_servers", "visible_tools"]


def hidden_mcp_server_names(
    grants: Iterable[McpOAuthGrantRow],
    *,
    account_id: uuid.UUID,
    shared_server_urls: Iterable[str],
) -> frozenset[str]:
    """Servers somebody connected personally that `account_id` has not.

    Pure. Trailing slashes are stripped on both sides of the shared-credential
    comparison, the same normalisation `put_mcp_oauth_credential` uses to
    match a vault credential to a server URL.
    """
    shared = {url.rstrip("/") for url in shared_server_urls}
    personal = {
        grant.server_name for grant in grants if grant.mcp_server_url.rstrip("/") not in shared
    }
    connected = {grant.server_name for grant in grants if grant.account_id == account_id}
    return frozenset(personal - connected)


def visible_mcp_servers(
    agent: BetaManagedAgentsAgent, hidden: frozenset[str]
) -> Sequence[BetaManagedAgentsMCPServerURLDefinition]:
    """The agent's MCP servers without the ones `hidden` names."""
    return [server for server in agent.mcp_servers if server.name not in hidden]


def visible_tools(agent: BetaManagedAgentsAgent, hidden: frozenset[str]) -> Sequence[MATool]:
    """The agent's tools without the toolsets of the servers `hidden` names."""
    return [
        tool
        for tool in agent.tools
        if not (isinstance(tool, BetaManagedAgentsMCPToolset) and tool.mcp_server_name in hidden)
    ]
