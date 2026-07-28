"""Wire contract for the chat-initiated credential-request button.

This module lives in `daimon.core`, not in either adapter, because the two
processes that need it cannot import each other: the MCP server (Cloud Run)
mints the button and posts it to a channel, while the Discord bot (worker VM)
later receives the click and must decode the same `custom_id`. Import-linter's
independence contract forbids `daimon.adapters.mcp` from importing
`daimon.adapters.discord` (or vice versa), so the shared encode/decode logic
has to sit one level up, in core.

Discord caps `custom_id` at 100 characters, which rules out encoding the
target fields (kind, agent, key/server name, requester, expiry) directly in
the id — an MCP server name alone can blow that budget. Instead the
`custom_id` carries only an opaque, single-use token; the actual target
fields live in the `credential_requests` DB row keyed by that token (see
`daimon.core.stores.credential_requests`). The button's *label* is what
names the human-readable target, via `build_button_label`.

`CUSTOM_ID_PATTERN` MUST fullmatch every minted custom_id: discord.py's
`DynamicItem` dispatch matches incoming custom_ids with `pattern.fullmatch`,
not a prefix search, so a template that merely starts with the right prefix
would never dispatch (or could over-match unrelated ids).
"""

from __future__ import annotations

import datetime as dt
import re
import secrets
from typing import Final, Literal

CUSTOM_ID_PREFIX: Final[str] = "ztc:"

# secrets.token_urlsafe(16) yields a 22-character URL-safe string (~128 bits
# of entropy) — the resulting custom_id ("ztc:" + 22 chars = 26 chars) sits
# comfortably under Discord's 100-character custom_id cap.
TOKEN_BYTES: Final[int] = 16

CUSTOM_ID_TEMPLATE: Final[str] = r"ztc:(?P<token>[A-Za-z0-9_-]{16,64})"
CUSTOM_ID_PATTERN: Final[re.Pattern[str]] = re.compile(CUSTOM_ID_TEMPLATE)

CredentialRequestKind = Literal["env", "mcp"]

# TTL bounds the replay window for a never-clicked button sitting in channel
# scrollback; combined with the store's single-use consume, this is the full
# replay defense.
DEFAULT_TTL: Final[dt.timedelta] = dt.timedelta(minutes=30)

# Discord's documented Button.label limit.
MAX_BUTTON_LABEL_CHARS: Final[int] = 80

_KIND_DISPLAY: Final[dict[CredentialRequestKind, str]] = {"env": "env", "mcp": "MCP"}


def mint_request_token() -> str:
    """Return a fresh, URL-safe, single-use token. Two calls never collide."""
    return secrets.token_urlsafe(TOKEN_BYTES)


def build_custom_id(token: str) -> str:
    """Return the wire `custom_id` for a minted token."""
    return f"{CUSTOM_ID_PREFIX}{token}"


def build_button_label(kind: CredentialRequestKind, target: str) -> str:
    """Return the human-readable button label naming the exact target.

    Truncates `target` (with a trailing "…") when it would overflow Discord's
    80-character button label limit — the label, not the custom_id, is what
    names the exact target, so it must fit on its own.
    """
    prefix = f"Add {_KIND_DISPLAY[kind]} credential: "
    available = MAX_BUTTON_LABEL_CHARS - len(prefix)
    if len(target) <= available:
        return f"{prefix}{target}"
    truncated = target[: available - 1] + "…"
    return f"{prefix}{truncated}"
