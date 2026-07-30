"""Unit tests for wizard answer serialization (agent block vs display summary)."""

from __future__ import annotations

import json

from daimon.core.wizard.answers import (
    MAX_SUMMARY_CHARS,
    escape_discord_markup,
    format_answer_block,
    summarize_answers,
)
from daimon.core.wizard.spec import Option, Step, StepKind, WizardSpec
from daimon.core.wizard.state import WizardState

_SPEC = WizardSpec(
    prompt="Order form",
    steps=[
        Step(
            key="color",
            question="Pick a color",
            kind=StepKind.CHOICE,
            options=[Option(label="Red", value="red"), Option(label="Blue", value="blue")],
        ),
        Step(
            key="toppings",
            question="Pick toppings",
            kind=StepKind.MULTI,
            options=[
                Option(label="Cheese", value="cheese"),
                Option(label="Olives", value="olives"),
            ],
        ),
        Step(key="notes", question="Any notes?", kind=StepKind.TEXT, options=[]),
    ],
)


def _state(answers: dict[str, list[str]]) -> WizardState:
    return WizardState(short_id="abcd1234", answers=answers)


def test_format_answer_block_first_line_names_the_prompt() -> None:
    block = format_answer_block(_SPEC, _state({}))

    assert block.startswith("Wizard answers (Order form):"), (
        "the block's header line must name the form's prompt"
    )


def test_format_answer_block_one_line_per_step_in_spec_order() -> None:
    block = format_answer_block(
        _SPEC, _state({"color": ["red"], "toppings": ["cheese", "olives"], "notes": ["hi"]})
    )
    lines = block.splitlines()

    key_lines = [
        line
        for line in lines
        if ":" in line and line.split(":")[0] in {"color", "toppings", "notes"}
    ]
    assert [line.split(":")[0] for line in key_lines] == ["color", "toppings", "notes"], (
        "answer block must emit one line per step, in spec order"
    )


def test_format_answer_block_unanswered_step_emits_empty_array() -> None:
    block = format_answer_block(_SPEC, _state({}))

    assert "notes: []" in block, "an unanswered step must emit an empty JSON array"


def test_format_answer_block_adversarial_value_survives_json_round_trip() -> None:
    adversarial = 'colon: value, "quoted", newline\nhere, and a ` backtick'
    block = format_answer_block(_SPEC, _state({"notes": [adversarial]}))

    notes_line = next(line for line in block.splitlines() if line.startswith("notes:"))
    recovered = json.loads(notes_line.removeprefix("notes: "))

    assert recovered == [adversarial], (
        "adversarial text with a colon, quotes, a newline and a backtick must round-trip via JSON"
    )


def test_format_answer_block_uses_raw_values_not_labels() -> None:
    block = format_answer_block(_SPEC, _state({"color": ["red"]}))

    assert "red" in block
    assert "Red" not in block.split("Wizard answers")[1], (
        "the answer block must carry raw option values, never labels"
    )


def test_summarize_answers_substitutes_labels_for_values() -> None:
    summary = summarize_answers(_SPEC, _state({"color": ["red"]}))

    assert "Red" in summary, "summarize_answers must substitute the option label for its value"
    assert "red" not in summary.replace("Red", ""), (
        "the raw value should not appear once substituted with its label"
    )


def test_summarize_answers_unanswered_step_renders_em_dash() -> None:
    summary = summarize_answers(_SPEC, _state({}))

    assert "—" in summary, "an unanswered step must render an em dash"


def test_summarize_answers_escapes_markdown_and_mass_mention() -> None:
    summary = summarize_answers(_SPEC, _state({"notes": ["*bold* @everyone"]}))

    assert "\\*bold\\*" in summary, "markdown control characters must be escaped in the summary"
    assert "@everyone" not in summary, "a raw @everyone must never survive into the summary"
    assert "@​everyone" in summary, "mass mention must be neutralized with a zero-width space"


def test_answer_block_and_summary_disagree_on_escaping_by_design() -> None:
    text = "*bold* @everyone"
    block = format_answer_block(_SPEC, _state({"notes": [text]}))
    summary = summarize_answers(_SPEC, _state({"notes": [text]}))

    assert text in block, "the agent-facing block must carry the raw, unescaped value"
    assert text not in summary, "the display summary must escape the same value"


def test_escape_discord_markup_escapes_control_characters() -> None:
    escaped = escape_discord_markup("*_~`|>#-[]()")

    for char in "*_~`|>#-[]()":
        assert f"\\{char}" in escaped, f"{char!r} must be backslash-escaped"


def test_escape_discord_markup_neutralizes_here_and_role_mention() -> None:
    escaped = escape_discord_markup("@here <@&12345>")

    assert "@here" not in escaped
    assert "<@&12345>" not in escaped
    assert "@​here" in escaped
    assert "<@​&12345\\>" in escaped, (
        "the mass-mention sequence is neutralized; > is still escaped like any other control char"
    )


def test_summary_is_truncated_to_the_text_display_budget() -> None:
    """An unbounded summary makes the review screen -- and the post-submit
    collapse re-render, on an already-claimed submission -- fail to render."""
    long_spec = WizardSpec(
        prompt="Order form",
        steps=[
            Step(key=f"k{i}", question=f"Question {i}", kind=StepKind.TEXT, options=[])
            for i in range(20)
        ],
    )
    state = WizardState(short_id="abcd1234", answers={f"k{i}": ["x" * 200] for i in range(20)})

    summary = summarize_answers(long_spec, state)

    assert len(summary) <= MAX_SUMMARY_CHARS, (
        "the summary must stay inside the budget a TextDisplay can carry"
    )
    assert summary.endswith("(answers truncated for display)"), (
        "a truncated summary must say so rather than silently losing answers"
    )
    assert not summary.endswith("\\\n… (answers truncated for display)"), (
        "the cut must not end on a lone backslash that would escape the marker's newline"
    )
