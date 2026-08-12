"""Unit tests for the agent-authored wizard form schema (no I/O, no DB)."""

from __future__ import annotations

import pytest
from daimon.core.wizard.spec import (
    HEAD_TEXT_OVERHEAD_CHARS,
    MAX_SELECT_OPTION_DESCRIPTION_CHARS,
    MAX_SELECT_OPTION_VALUE_CHARS,
    MAX_SELECT_OPTIONS,
    MAX_SELECT_PLACEHOLDER_CHARS,
    MAX_STEPS,
    MAX_TEXT_DISPLAY_CHARS,
    Option,
    Step,
    StepKind,
    WizardSpec,
)
from pydantic import ValidationError


def test_minimal_choice_step_is_valid() -> None:
    spec = WizardSpec(
        prompt="pick one",
        steps=[
            Step(
                key="k", question="q", kind=StepKind.CHOICE, options=[Option(label="a", value="a")]
            )
        ],
    )

    assert len(spec.steps) == 1, "a single valid choice step must be accepted"


def test_minimal_multi_step_is_valid() -> None:
    spec = WizardSpec(
        prompt="pick some",
        steps=[
            Step(
                key="k",
                question="q",
                kind=StepKind.MULTI,
                options=[Option(label="a", value="a"), Option(label="b", value="b")],
                min=0,
                max=2,
            )
        ],
    )

    assert len(spec.steps) == 1, "a single valid multi step must be accepted"


def test_minimal_text_step_is_valid() -> None:
    spec = WizardSpec(prompt="tell me", steps=[Step(key="k", question="q", kind=StepKind.TEXT)])

    assert len(spec.steps) == 1, "a single valid text step must be accepted"


def test_zero_steps_rejected() -> None:
    with pytest.raises(ValidationError, match="zero steps"):
        WizardSpec(prompt="p", steps=[])


def test_more_than_max_steps_rejected() -> None:
    steps = [Step(key=f"k{i}", question="q", kind=StepKind.TEXT) for i in range(MAX_STEPS + 1)]

    with pytest.raises(ValidationError, match="exceeding the"):
        WizardSpec(prompt="p", steps=steps)


def test_duplicate_keys_rejected() -> None:
    steps = [
        Step(key="dup", question="q1", kind=StepKind.TEXT),
        Step(key="dup", question="q2", kind=StepKind.TEXT),
    ]

    with pytest.raises(ValidationError, match="dup") as exc_info:
        WizardSpec(prompt="p", steps=steps)
    assert "dup" in str(exc_info.value), "the offending step's key must appear in the message"


def test_choice_step_with_no_options_rejected() -> None:
    with pytest.raises(ValidationError, match="no options") as exc_info:
        WizardSpec(prompt="p", steps=[Step(key="k", question="q", kind=StepKind.CHOICE)])
    assert "k" in str(exc_info.value), "the offending step's key must appear in the message"


def test_too_many_options_rejected() -> None:
    options = [Option(label=str(i), value=str(i)) for i in range(MAX_SELECT_OPTIONS + 1)]

    with pytest.raises(ValidationError, match="exceeding the") as exc_info:
        WizardSpec(
            prompt="p", steps=[Step(key="k", question="q", kind=StepKind.CHOICE, options=options)]
        )
    assert "k" in str(exc_info.value), "the offending step's key must appear in the message"


def test_oversized_option_label_rejected() -> None:
    option = Option(label="x" * 81, value="v")

    with pytest.raises(ValidationError, match="button label limit") as exc_info:
        WizardSpec(
            prompt="p", steps=[Step(key="k", question="q", kind=StepKind.CHOICE, options=[option])]
        )
    assert "k" in str(exc_info.value), "the offending step's key must appear in the message"


def test_multi_min_greater_than_max_rejected() -> None:
    options = [Option(label="a", value="a"), Option(label="b", value="b")]

    with pytest.raises(ValidationError, match="must not exceed") as exc_info:
        WizardSpec(
            prompt="p",
            steps=[
                Step(
                    key="k",
                    question="q",
                    kind=StepKind.MULTI,
                    options=options,
                    min=2,
                    max=1,
                )
            ],
        )
    assert "k" in str(exc_info.value), "the offending step's key must appear in the message"


def test_multi_max_above_option_count_rejected() -> None:
    options = [Option(label="a", value="a")]

    with pytest.raises(ValidationError, match="available options") as exc_info:
        WizardSpec(
            prompt="p",
            steps=[Step(key="k", question="q", kind=StepKind.MULTI, options=options, min=0, max=5)],
        )
    assert "k" in str(exc_info.value), "the offending step's key must appear in the message"


def test_multi_min_above_option_count_rejected_when_max_is_unset() -> None:
    # With `max` unset the renderer derives max_values from the option count,
    # so min=3 against 2 options would emit min_values > max_values.
    options = [Option(label="a", value="a"), Option(label="b", value="b")]

    with pytest.raises(ValidationError, match="available options") as exc_info:
        WizardSpec(
            prompt="p",
            steps=[Step(key="k", question="q", kind=StepKind.MULTI, options=options, min=3)],
        )
    assert "k" in str(exc_info.value), "the offending step's key must appear in the message"


def test_oversized_option_value_rejected() -> None:
    option = Option(label="a", value="v" * (MAX_SELECT_OPTION_VALUE_CHARS + 1))

    with pytest.raises(ValidationError, match="option value") as exc_info:
        WizardSpec(
            prompt="p", steps=[Step(key="k", question="q", kind=StepKind.CHOICE, options=[option])]
        )
    assert "k" in str(exc_info.value), "the offending step's key must appear in the message"


def test_oversized_option_description_rejected() -> None:
    option = Option(
        label="a", value="a", description="d" * (MAX_SELECT_OPTION_DESCRIPTION_CHARS + 1)
    )

    with pytest.raises(ValidationError, match="option description") as exc_info:
        WizardSpec(
            prompt="p", steps=[Step(key="k", question="q", kind=StepKind.CHOICE, options=[option])]
        )
    assert "k" in str(exc_info.value), "the offending step's key must appear in the message"


def test_oversized_head_text_rejected() -> None:
    question = "q" * (MAX_TEXT_DISPLAY_CHARS - HEAD_TEXT_OVERHEAD_CHARS)

    with pytest.raises(ValidationError, match="head text") as exc_info:
        WizardSpec(prompt="p", steps=[Step(key="k", question=question, kind=StepKind.TEXT)])
    assert "k" in str(exc_info.value), "the offending step's key must appear in the message"


def test_multi_question_above_the_placeholder_limit_rejected() -> None:
    options = [Option(label="a", value="a")]
    question = "q" * (MAX_SELECT_PLACEHOLDER_CHARS + 1)

    with pytest.raises(ValidationError, match="placeholder") as exc_info:
        WizardSpec(
            prompt="p",
            steps=[Step(key="k", question=question, kind=StepKind.MULTI, options=options)],
        )
    assert "k" in str(exc_info.value), "the offending step's key must appear in the message"


def test_a_long_question_is_accepted_on_a_non_multi_step() -> None:
    """Only a multi step's question becomes a select placeholder; the same
    length on a choice step renders as ordinary head text."""
    spec = WizardSpec(
        prompt="p",
        steps=[
            Step(
                key="k",
                question="q" * (MAX_SELECT_PLACEHOLDER_CHARS + 1),
                kind=StepKind.CHOICE,
                options=[Option(label="a", value="a")],
            )
        ],
    )

    assert len(spec.steps) == 1


def test_too_many_image_bearing_steps_rejected() -> None:
    steps = [
        Step(key=f"k{i}", question="q", kind=StepKind.TEXT, image_handle=f"handle-{i}")
        for i in range(11)
    ]

    with pytest.raises(ValidationError, match="image ceiling"):
        WizardSpec(prompt="p", steps=steps)


def test_duplicate_image_handle_rejected() -> None:
    steps = [
        Step(key="k0", question="q0", kind=StepKind.TEXT, image_handle="same-handle"),
        Step(key="k1", question="q1", kind=StepKind.TEXT, image_handle="same-handle"),
    ]

    with pytest.raises(ValidationError, match="must be unique") as exc_info:
        WizardSpec(prompt="p", steps=steps)
    assert "same-handle" in str(exc_info.value), "the shared handle must appear in the message"


def test_choice_step_exceeding_component_ceiling_rejected() -> None:
    # 15 described options is well within MAX_SELECT_OPTIONS (25), but each
    # described option renders as a heavier section (button + text), not a
    # bare button -- pushing the estimated component count past the ceiling
    # even though the option-count ceiling alone would allow it.
    options = [Option(label=str(i), value=str(i), description="d") for i in range(15)]

    with pytest.raises(ValidationError, match="component") as exc_info:
        WizardSpec(
            prompt="p",
            steps=[Step(key="k", question="q", kind=StepKind.CHOICE, options=options)],
        )
    assert "k" in str(exc_info.value), "the offending step's key must appear in the message"


def test_step_accepts_label_as_the_question_field() -> None:
    """`label` must alias `question`, like `title` already does.

    A production session sent every step keyed on `label` with `type` +
    `single_choice`/`free_text`/`multi_select`. The kind aliases absorbed their
    half; `question` did not, so all four steps were rejected and the form was
    abandoned. `label` is the predictable miss because `Option.label` is this
    schema's own word for user-facing text.
    """
    step = Step.model_validate(
        {
            "type": "single_choice",
            "label": "Which format do you want?",
            "options": ["pdf", "docx"],
        }
    )
    assert step.question == "Which format do you want?", (
        "`label` must populate `question`; it is a documented alias"
    )
    assert step.kind is StepKind.CHOICE, "`type`/`single_choice` aliases must still apply"


def test_step_prefers_question_over_label_when_both_are_present() -> None:
    """AliasChoices resolves left-to-right: canonical `question` wins."""
    step = Step.model_validate({"kind": "text", "question": "canonical", "label": "aliased"})
    assert step.question == "canonical", (
        "the canonical field name must win over an alias when both are supplied"
    )
