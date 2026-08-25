"""Pure escaping helpers for text sent in Slack's native Markdown blocks.

Slack uses three HTML entities to prevent literal text from being interpreted
as links or mentions. The escape order is load-bearing:

1. ``&`` → ``&amp;``
2. ``<`` → ``&lt;``
3. ``>`` → ``&gt;``

This module is stdlib-only so Slack delivery and eval can share the exact
pre-delivery transform without creating an adapter-to-adapter dependency.
"""

from __future__ import annotations

import re


def escape_mrkdwn(text: str) -> str:
    """Escape Slack control characters in *text*."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


_ESCAPED_MENTION = re.compile(r"&lt;([@#][A-Z0-9]+(?:\|[^&<>]*)?)&gt;")


def escape_mrkdwn_preserving_mentions(text: str) -> str:
    """Escape Slack controls but preserve user and channel mention tokens."""
    escaped = escape_mrkdwn(text)
    return _ESCAPED_MENTION.sub(lambda match: f"<{match.group(1)}>", escaped)
