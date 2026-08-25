"""Slack mrkdwn syntax checks for Slack-rendered eval answers."""

from __future__ import annotations

import re

from daimon.adapters.cli.eval.models import Grade

_FORBIDDEN: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("double-asterisk bold", re.compile(r"\*\*")),
    ("Markdown link", re.compile(r"\[[^\]\n]+\]\(https?://[^)\n]+\)")),
    ("Markdown heading", re.compile(r"(?m)^\s{0,3}#{1,6}\s+")),
)
_PIPE_TABLE_LINE = re.compile(r"^\|.*\|$")
_FENCED_CODE = re.compile(r"```.*?```", re.DOTALL)


def grade(answer: str) -> Grade:
    """Reject Markdown constructs unsupported by Slack's mrkdwn surface."""
    prose = _FENCED_CODE.sub("", answer)
    found = [label for label, pattern in _FORBIDDEN if pattern.search(prose)]
    if sum(bool(_PIPE_TABLE_LINE.fullmatch(line)) for line in prose.splitlines()) >= 2:
        found.append("Markdown pipe table")
    return {
        "name": "mrkdwn_compliance",
        "passed": not found,
        "gating": True,
        "detail": "compatible Slack mrkdwn" if not found else f"found: {', '.join(found)}",
    }
