"""Find the authorization server behind an MCP URL (MCP authorization spec).

Order, per the spec: the `resource_metadata` URL a 401 advertises in
`WWW-Authenticate`, then the path-based and root `.well-known` protected
resource documents, then the authorization server's own metadata under
`/.well-known/oauth-authorization-server` (path-aware first) with the OIDC
document as the fallback. Every step is one GET with an injected client;
nothing here retries or swallows — a server that cannot be discovered raises
`McpOAuthDiscoveryError` naming the last URL tried.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from daimon.core.errors import DaimonError
from daimon.core.mcp_oauth.models import AuthorizationServerMetadata, ProtectedResourceMetadata

_INITIALIZE_BODY = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-06-18",
        "capabilities": {},
        "clientInfo": {"name": "daimon", "version": "0"},
    },
}
_RESOURCE_METADATA_RE = re.compile(r'resource_metadata="([^"]+)"')


class McpOAuthDiscoveryError(DaimonError):
    """No authorization server could be found for the MCP URL."""


@dataclass(frozen=True, slots=True)
class McpProbe:
    """One unauthenticated (or bearer) `initialize` against the server."""

    status_code: int
    resource_metadata_url: str | None

    @property
    def accepts_bearer(self) -> bool:
        return self.status_code < 400

    @property
    def rejects_credentials(self) -> bool:
        return self.status_code in (401, 403)


@dataclass(frozen=True, slots=True)
class OAuthDiscovery:
    resource: ProtectedResourceMetadata | None
    authorization_server: AuthorizationServerMetadata


async def probe_mcp_server(
    http: httpx.AsyncClient, *, mcp_server_url: str, bearer_token: str | None = None
) -> McpProbe:
    """POST an MCP `initialize` and report how the server answered.

    Used at the token form to refuse a rejected token before it is stored,
    and before an OAuth flow to learn where the server's metadata lives.
    """
    headers = {"Accept": "application/json, text/event-stream"}
    if bearer_token is not None:
        headers["Authorization"] = f"Bearer {bearer_token}"
    response = await http.post(mcp_server_url, json=_INITIALIZE_BODY, headers=headers)
    match = _RESOURCE_METADATA_RE.search(response.headers.get("www-authenticate", ""))
    return McpProbe(
        status_code=response.status_code,
        resource_metadata_url=match.group(1) if match else None,
    )


def protected_resource_urls(mcp_server_url: str, advertised: str | None) -> list[str]:
    parsed = urlparse(mcp_server_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    urls = [advertised] if advertised else []
    if parsed.path and parsed.path != "/":
        urls.append(urljoin(base, f"/.well-known/oauth-protected-resource{parsed.path}"))
    urls.append(urljoin(base, "/.well-known/oauth-protected-resource"))
    return urls


def authorization_server_urls(issuer: str) -> list[str]:
    parsed = urlparse(issuer)
    base = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path.rstrip("/")
    if path:
        return [
            urljoin(base, f"/.well-known/oauth-authorization-server{path}"),
            urljoin(base, f"/.well-known/openid-configuration{path}"),
            urljoin(base, f"{path}/.well-known/openid-configuration"),
        ]
    return [
        urljoin(base, "/.well-known/oauth-authorization-server"),
        urljoin(base, "/.well-known/openid-configuration"),
    ]


async def discover_authorization_server(
    http: httpx.AsyncClient, *, mcp_server_url: str, resource_metadata_url: str | None = None
) -> OAuthDiscovery:
    """Resolve the MCP URL to its authorization server's endpoints."""
    resource: ProtectedResourceMetadata | None = None
    for url in protected_resource_urls(mcp_server_url, resource_metadata_url):
        response = await http.get(url, headers={"Accept": "application/json"})
        if response.status_code == 200:
            resource = ProtectedResourceMetadata.model_validate(response.json())
            break
    # Without a resource document the 2025-03-26 spec's fallback applies: the
    # authorization server is assumed to live at the MCP server's origin.
    origin = urlparse(mcp_server_url)
    issuer_candidates = (
        resource.authorization_servers if resource else [f"{origin.scheme}://{origin.netloc}"]
    )
    tried: list[str] = []
    for issuer in issuer_candidates:
        for url in authorization_server_urls(issuer):
            tried.append(url)
            response = await http.get(url, headers={"Accept": "application/json"})
            if response.status_code == 200:
                metadata = AuthorizationServerMetadata.model_validate(response.json())
                return OAuthDiscovery(resource=resource, authorization_server=metadata)
    raise McpOAuthDiscoveryError(
        f"no OAuth authorization server metadata for {mcp_server_url} (tried {', '.join(tried)})"
    )
