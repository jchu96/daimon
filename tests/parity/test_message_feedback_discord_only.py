"""Executable record of message feedback's deliberate Discord-only scope.

An absent capability file in this suite is mechanically indistinguishable
from an oversight: nothing else in the test suite would fail if Slack's bot
scopes silently grew the two permissions a Slack feedback path would need,
without anyone building that path or updating the platform-neutral core's
documented scope. Asserting the exemption here, rather than merely
documenting it in a comment, makes that drift fail loudly instead of
silently.

No platform parametrization, no database -- this is a scope check, not a
scenario test.
"""

from __future__ import annotations

import daimon.core.message_feedback
from daimon.core.slack_oauth import SLACK_BOT_SCOPES


def test_slack_bot_scopes_still_lack_the_reaction_feedback_scopes() -> None:
    assert "reactions:read" not in SLACK_BOT_SCOPES and "im:write" not in SLACK_BOT_SCOPES, (
        "message feedback is Discord-only because Slack cannot observe reactions or open a "
        "direct message without the reactions:read and im:write bot scopes; if this test "
        "fails, either build the Slack feedback path or update this exemption record -- do "
        "not simply delete the assertion"
    )


def test_message_feedback_core_documents_the_discord_only_scope() -> None:
    doc = daimon.core.message_feedback.__doc__

    assert doc is not None, "daimon.core.message_feedback must carry a module docstring"
    assert "reactions:read" in doc, (
        "the module docstring must name the reactions:read scope Slack lacks"
    )
    assert "im:write" in doc, "the module docstring must name the im:write scope Slack lacks"
