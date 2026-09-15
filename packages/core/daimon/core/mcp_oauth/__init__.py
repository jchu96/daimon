"""Authorization-code flow for MCP servers that only accept OAuth (Notion, Slack, …).

Managed Agents holds the tokens (an `mcp_oauth` vault credential, refreshed by
Anthropic) and connects to the server at session time; daimon's only job is
the browser half — discovery, dynamic client registration, PKCE, the code
exchange — and writing the result into the caller's vault. Grants are per
person: each member connects their own account, and the credential lands in
that member's per-agent vault only.

`models` and `flow.build_authorization_url` are pure; `discovery`, `flow`'s
HTTP calls and `vault` are the shell, with the `httpx.AsyncClient` and the
Anthropic client injected.
"""

from daimon.core.mcp_oauth.discovery import (
    McpProbe,
    OAuthDiscovery,
    discover_authorization_server,
    probe_mcp_server,
)
from daimon.core.mcp_oauth.flow import (
    Pkce,
    build_authorization_url,
    exchange_authorization_code,
    generate_pkce,
    register_client,
)
from daimon.core.mcp_oauth.models import (
    AuthorizationServerMetadata,
    ClientRegistration,
    ProtectedResourceMetadata,
    TokenResponse,
)
from daimon.core.mcp_oauth.vault import put_mcp_oauth_credential

__all__ = [
    "AuthorizationServerMetadata",
    "ClientRegistration",
    "McpProbe",
    "OAuthDiscovery",
    "Pkce",
    "ProtectedResourceMetadata",
    "TokenResponse",
    "build_authorization_url",
    "discover_authorization_server",
    "exchange_authorization_code",
    "generate_pkce",
    "probe_mcp_server",
    "put_mcp_oauth_credential",
    "register_client",
]
