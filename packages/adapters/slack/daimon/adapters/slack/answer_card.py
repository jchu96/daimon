"""Experimental composed answer-card rendering and delivery for Slack."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

from daimon.adapters.slack.mrkdwn import escape_mrkdwn_preserving_mentions
from slack_sdk.web.async_client import AsyncWebClient

__all__ = [
    "AnswerCardValidationError",
    "compose_answer_card",
    "deliver_answer_card",
    "validate_answer_card",
]

_MAX_BLOCKS = 50
_MAX_SECTION_CHARS = 3000
_MAX_HEADER_CHARS = 150
_MAX_ACTION_ELEMENTS = 25
_ALLOWED_BLOCK_TYPES = frozenset({"header", "section", "actions"})


class AnswerCardValidationError(ValueError):
    """Raised when a composed card would be rejected by Slack."""


def _text_value(block: dict[str, Any], *, expected_type: str | None = None) -> str:
    raw_text = block.get("text")
    if not isinstance(raw_text, dict):
        raise AnswerCardValidationError("block text must be a text object")
    text = cast(dict[str, object], raw_text)
    text_type = text.get("type")
    if expected_type is not None and text_type != expected_type:
        raise AnswerCardValidationError(f"block text type must be {expected_type}")
    if expected_type is None and text_type not in {"mrkdwn", "plain_text"}:
        raise AnswerCardValidationError("section text type must be mrkdwn or plain_text")
    value = text.get("text")
    if not isinstance(value, str) or not value:
        raise AnswerCardValidationError("block text must be a non-empty string")
    return value


def validate_answer_card(blocks: list[dict[str, Any]]) -> None:
    """Reject cards outside the deliberately small supported Block Kit subset."""
    if not blocks:
        raise AnswerCardValidationError("answer card must contain at least one block")
    if len(blocks) > _MAX_BLOCKS:
        raise AnswerCardValidationError(f"answer card exceeds {_MAX_BLOCKS} blocks")

    actions_seen = False
    for index, block in enumerate(blocks):
        block_type = block.get("type")
        if block_type not in _ALLOWED_BLOCK_TYPES:
            raise AnswerCardValidationError(f"unsupported block type: {block_type!r}")

        if block_type == "header":
            if index != 0:
                raise AnswerCardValidationError("header must be the first block")
            value = _text_value(block, expected_type="plain_text")
            if len(value) > _MAX_HEADER_CHARS:
                raise AnswerCardValidationError(
                    f"header text exceeds {_MAX_HEADER_CHARS} characters"
                )
        elif block_type == "section":
            if actions_seen:
                raise AnswerCardValidationError("sections cannot follow an actions block")
            value = _text_value(block)
            if len(value) > _MAX_SECTION_CHARS:
                raise AnswerCardValidationError(
                    f"section text exceeds {_MAX_SECTION_CHARS} characters"
                )
        else:
            if actions_seen or index != len(blocks) - 1:
                raise AnswerCardValidationError("actions must be the single trailing block")
            raw_elements = block.get("elements")
            if not isinstance(raw_elements, list) or not raw_elements:
                raise AnswerCardValidationError("actions must contain at least one element")
            elements = cast(list[object], raw_elements)
            if len(elements) > _MAX_ACTION_ELEMENTS:
                raise AnswerCardValidationError(f"actions exceeds {_MAX_ACTION_ELEMENTS} elements")
            if any(not isinstance(element, dict) for element in elements):
                raise AnswerCardValidationError("action elements must be objects")
            actions_seen = True

    if blocks[0].get("type") != "header":
        raise AnswerCardValidationError("answer card must start with a header")
    if not any(block.get("type") == "section" for block in blocks):
        raise AnswerCardValidationError("answer card must contain an answer section")


def compose_answer_card(
    answer_text: str,
    revision: int,
    actions_block: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Compose and validate one complete answer card.

    Each call rebuilds the whole card, including ``actions_block``. This is
    required because Slack's ``chat.update`` replaces the complete block list;
    blocks omitted from a revision disappear from the message.
    """
    if isinstance(revision, bool) or revision < 1:
        raise AnswerCardValidationError("revision must be a positive integer")
    if not answer_text:
        raise AnswerCardValidationError("answer text must not be empty")

    escaped = escape_mrkdwn_preserving_mentions(answer_text)
    sections = [
        escaped[start : start + _MAX_SECTION_CHARS]
        for start in range(0, len(escaped), _MAX_SECTION_CHARS)
    ]
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"Answer · revision {revision}"},
        },
        *({"type": "section", "text": {"type": "mrkdwn", "text": section}} for section in sections),
    ]
    if actions_block is not None:
        blocks.append(deepcopy(actions_block))

    validate_answer_card(blocks)
    return blocks, escaped[:_MAX_SECTION_CHARS]


async def deliver_answer_card(
    client: AsyncWebClient,
    *,
    channel: str,
    thread_ts: str,
    answer_text: str,
    revision: int,
    message_ts: str | None = None,
    actions_block: dict[str, Any] | None = None,
) -> str:
    """Post the first card or replace an existing card with a full revision."""
    blocks, fallback_text = compose_answer_card(answer_text, revision, actions_block)
    if message_ts is None:
        response = await client.chat_postMessage(  # pyright: ignore[reportUnknownMemberType]
            channel=channel,
            thread_ts=thread_ts,
            blocks=blocks,
            text=fallback_text,
        )
        return cast(str, response["ts"])  # pyright: ignore[reportUnknownVariableType]

    await client.chat_update(  # pyright: ignore[reportUnknownMemberType]
        channel=channel,
        ts=message_ts,
        blocks=blocks,
        text=fallback_text,
    )
    return message_ts
