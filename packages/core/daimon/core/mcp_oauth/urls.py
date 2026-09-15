"""Which URLs daimon will contact on a model's or a remote server's say-so.

Before this flow nothing in daimon fetched an MCP URL itself; Managed Agents
did, from its own network. The probe and discovery now run inside the
deployment, so a URL a prompt-injected agent or a hostile server names must
be a public https endpoint. Names are not resolved here: a hostname that
points at a private range is the egress policy's problem, an address that
already is one is refused up front.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

from daimon.core.errors import DaimonError

_LOCAL_SUFFIXES = (".localhost", ".local", ".internal")


class McpUrlError(DaimonError):
    """The URL is not a public endpoint daimon will contact."""


def assert_public_host(url: str, *, what: str = "url") -> str:
    """Refuse a URL with no host, embedded credentials, a local name or a non-public address."""
    parsed = urlparse(url)
    host = parsed.hostname
    if not host:
        raise McpUrlError(f"{what} must name a host: {url}")
    if parsed.username is not None or parsed.password is not None:
        raise McpUrlError(f"{what} must not carry credentials: {url}")
    if host == "localhost" or host.endswith(_LOCAL_SUFFIXES):
        raise McpUrlError(f"{what} points at a local name: {url}")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return url
    if not address.is_global:
        raise McpUrlError(f"{what} points at a non-public address: {url}")
    return url


def assert_public_https_url(url: str, *, what: str = "url") -> str:
    """`assert_public_host` plus the https scheme; every URL daimon fetches passes here."""
    if urlparse(url).scheme != "https":
        raise McpUrlError(f"{what} must be https: {url}")
    return assert_public_host(url, what=what)
