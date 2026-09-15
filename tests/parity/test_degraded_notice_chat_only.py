"""Executable record: the #79 degraded-turn notice is rendered by the two chat
adapters only.

The headless (routine) and CLI lifecycles get the same `TurnState`, with the
dropped server on `mcp_failures`, and report the turn as the driver decided:
success when the model answered, failure when it produced nothing. They have
no reader to warn mid-conversation; the CLI consumer reads the serialized
state. If either grows a rendering, replace this record rather than deleting it.
"""

from __future__ import annotations

import inspect
from types import ModuleType

import daimon.adapters.cli.run.lifecycle as cli_lifecycle
import daimon.adapters.discord.lifecycle as discord_lifecycle
import daimon.adapters.slack.lifecycle as slack_lifecycle
import daimon.core.headless_runner as headless_runner


def _mentions_notice(module: ModuleType) -> bool:
    return "render_degraded_notice" in inspect.getsource(module)


def test_only_discord_and_slack_render_the_degraded_notice() -> None:
    assert _mentions_notice(discord_lifecycle) and _mentions_notice(slack_lifecycle), (
        "both chat adapters name the dropped server under the reply"
    )
    assert not _mentions_notice(headless_runner) and not _mentions_notice(cli_lifecycle), (
        "headless and CLI lifecycles carry mcp_failures on the state without rendering"
    )
