"""Which of an agent's MCP servers a given caller can actually authenticate.

An OAuth sign-in is personal: the grant lands in the connecting person's own
(account, agent) vault, but `attach_mcp_server_to_agent` puts the server on
the agent everyone shares. MA opens every attached server on every turn and
resolves its credential from the vault mounted on that session — the
caller's — so one person signing in left every other person's turn opening a
server they have no token for, failing it with
`mcp_authentication_failed_error`, and carrying the degraded-turn notice
under every reply whether or not the turn had anything to do with that
server.

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
Grants are matched to the agent's servers by URL and count only for the
agent they were made on: a fork copies its source's servers but inherits no
sign-ins, and a name re-pointed at a different server does not let a grant
for the old one pass for a connection to the new.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping, Sequence
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
    agent_id: uuid.UUID,
    account_id: uuid.UUID,
    shared_server_urls: Iterable[str],
    server_urls: Mapping[str, str],
) -> frozenset[str]:
    """Of `server_urls`, the ones people sign in to that this caller has not
    signed in to on this agent.

    Pure. `server_urls` is the agent's own `{name: url}`, so every decision is
    made about the server a session would actually mount. `grants` is the
    whole tenant's: a URL anybody signed in to, on any agent, is a sign-in
    server wherever it appears, because MA authenticates it from the vault
    of (caller, this agent) and only a sign-in on this agent fills that
    vault. So a fork carrying its source's server hides it from everyone —
    the person who signed in on the source included — until someone signs
    in on the fork. A URL nobody signed in to anywhere is not personal at
    all and is left alone.

    Matching is by URL: it is what MA authenticates against, and the same
    server may be attached under different names on different agents.
    Trailing slashes are stripped on both sides of every comparison, the
    same normalisation `put_mcp_oauth_credential` uses to match a vault
    credential to a server URL.
    """
    shared = {url.rstrip("/") for url in shared_server_urls}
    hidden: set[str] = set()
    for name, url in server_urls.items():
        target = url.rstrip("/")
        if target in shared:
            continue
        matching = [grant for grant in grants if grant.mcp_server_url.rstrip("/") == target]
        if not matching:
            continue
        if any(g.agent_id == agent_id and g.account_id == account_id for g in matching):
            continue
        hidden.add(name)
    return frozenset(hidden)


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
