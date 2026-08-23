"""Tests for the experimental composed Slack answer card."""

from __future__ import annotations

from typing import Any

import pytest
import yarl
from daimon.adapters.slack.answer_card import (
    AnswerCardValidationError,
    compose_answer_card,
    deliver_answer_card,
    validate_answer_card,
)

_POST_URL = yarl.URL("https://slack.com/api/chat.postMessage")
_UPDATE_URL = yarl.URL("https://slack.com/api/chat.update")


def _header() -> dict[str, Any]:
    return {
        "type": "header",
        "text": {"type": "plain_text", "text": "Answer · revision 1"},
    }


def _section(text: str = "Answer") -> dict[str, Any]:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def _actions() -> dict[str, Any]:
    return {
        "type": "actions",
        "elements": [
            {
                "type": "button",
                "action_id": "helpful",
                "text": {"type": "plain_text", "text": "Helpful"},
                "value": "yes",
            }
        ],
    }


def test_validator_accepts_supported_card() -> None:
    validate_answer_card([_header(), _section(), _actions()])


@pytest.mark.parametrize(
    "blocks",
    [
        [_header(), *[_section(str(index)) for index in range(50)]],
        [_header(), _section("x" * 3001)],
        [_header(), {"type": "divider"}, _section()],
        [_header(), _actions(), _section()],
    ],
    ids=["too-many-blocks", "section-too-long", "type-not-allowed", "actions-not-trailing"],
)
def test_validator_rejects_invalid_cards(blocks: list[dict[str, Any]]) -> None:
    with pytest.raises(AnswerCardValidationError):
        validate_answer_card(blocks)


def test_compose_is_deterministic_and_splits_sections_at_3000_characters() -> None:
    answer = "x" * 6001
    actions = _actions()

    first = compose_answer_card(answer, 3, actions)
    second = compose_answer_card(answer, 3, actions)

    assert first == second, "the same inputs must produce an identical complete card"
    blocks, fallback = first
    section_lengths = [len(block["text"]["text"]) for block in blocks if block["type"] == "section"]
    assert section_lengths == [3000, 3000, 1]
    assert blocks[-1] == actions, "actions must be included as the trailing block"
    assert blocks[-1] is not actions, "composition must not retain the caller's mutable block"
    assert fallback == "x" * 3000


async def test_delivery_posts_then_updates_with_actions_on_every_revision(
    fake_slack_web_client: Any,
) -> None:
    actions = _actions()
    message_ts = await deliver_answer_card(
        fake_slack_web_client.client,
        channel="C_TEST",
        thread_ts="1700000000.000000",
        answer_text="Draft answer",
        revision=1,
        actions_block=actions,
    )
    await deliver_answer_card(
        fake_slack_web_client.client,
        channel="C_TEST",
        thread_ts="1700000000.000000",
        answer_text="Revised answer",
        revision=2,
        message_ts=message_ts,
        actions_block=actions,
    )

    posts = fake_slack_web_client.mock.requests.get(("POST", _POST_URL), [])
    updates = fake_slack_web_client.mock.requests.get(("POST", _UPDATE_URL), [])
    assert len(posts) == 1, "the first card revision must use chat.postMessage"
    assert len(updates) == 1, "later revisions must use chat.update"
    assert posts[0].kwargs["json"]["blocks"][-1] == actions
    update_blocks = updates[0].kwargs["json"]["blocks"]
    assert update_blocks[-1] == actions, "the full update must re-include the actions block"
    assert update_blocks[0]["text"]["text"] == "Answer · revision 2"
    assert update_blocks[1]["text"]["text"] == "Revised answer"


async def test_delivery_rejects_oversized_card_before_calling_slack(
    fake_slack_web_client: Any,
) -> None:
    with pytest.raises(AnswerCardValidationError, match="exceeds 50 blocks"):
        await deliver_answer_card(
            fake_slack_web_client.client,
            channel="C_TEST",
            thread_ts="1700000000.000000",
            answer_text="x" * 147_001,
            revision=1,
        )

    assert not fake_slack_web_client.mock.requests.get(("POST", _POST_URL), []), (
        "invalid cards must fail locally before Slack receives a request"
    )
