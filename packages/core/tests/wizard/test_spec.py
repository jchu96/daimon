"""Unit tests for the agent-authored wizard form schema (no I/O, no DB)."""

from __future__ import annotations

import pytest
from daimon.core.wizard.spec import (
    MAX_SELECT_OPTIONS,
    MAX_STEPS,
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
