"""Compatibility exports for the shared Slack text escaper."""

from daimon.core.slack_mrkdwn import (
    escape_mrkdwn,
    escape_mrkdwn_preserving_mentions,
)

__all__ = [
    "escape_mrkdwn",
    "escape_mrkdwn_preserving_mentions",
]
