"""The one pure transition function every wizard tap runs through.

`apply` is the single place that decides what a tap or a typed submission
does to a `WizardState` — navigation, answer recording, and the review-to-
submitted flip. No adapter branches on `ActionKind` itself; every process
that dispatches a tap decodes it into a `WizardAction` (via `build_action`)
and hands both action and state to `apply`, so the behaviour a user sees is
identical no matter which platform rendered the screen that produced the
tap.

`build_action` is the companion translator: it turns an already-parsed
custom_id action string (see `daimon.core.wizard.state`'s three grammars)
into a `WizardAction`. It is deliberately narrow — `s{n}_enter` (the button
that opens a modal) has no `WizardAction` because opening a modal is not a
state transition; the adapter that receives that tap opens its platform's
text-input UI directly and only calls back into `build_action` once the user
has actually typed something, at which point the action string is
`s{n}_custom`.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence

from daimon.core.wizard.spec import StepKind, WizardSpec
from daimon.core.wizard.state import ActionKind, WizardAction, WizardState, WizardStatus

_NAV_ACTION_RE = re.compile(r"^s(?P<step>\d{1,2})_(?P<suffix>c(?P<option>\d{1,2})|enter|custom)$")
_SELECT_ACTION_RE = re.compile(r"^s(?P<step>\d{1,2})_sel$")


def _clamp_step(step_index: int, spec: WizardSpec) -> int:
    return max(0, min(step_index, len(spec.steps)))


def _require_step_index(action: WizardAction, spec: WizardSpec) -> int:
    step_index = action.step_index
    if step_index is None or not (0 <= step_index < len(spec.steps)):
        raise ValueError(
            f"action {action.kind.value!r} carries step_index {step_index!r}, which is "
            f"out of range for a {len(spec.steps)}-step spec"
        )
    return step_index


def _with_answer(state: WizardState, key: str, values: list[str]) -> WizardState:
    answers = dict(state.answers)
    answers[key] = values
    return dataclasses.replace(state, answers=answers)


def _append_answer(state: WizardState, key: str, value: str) -> WizardState:
    answers = dict(state.answers)
    answers[key] = [*answers.get(key, []), value]
    return dataclasses.replace(state, answers=answers)


def apply(state: WizardState, spec: WizardSpec, action: WizardAction) -> WizardState:
    """Return the new `WizardState` after `action` is applied to `state`.

    Never mutates `state.answers` — every branch that records a value builds
    a fresh dict via `_with_answer`/`_append_answer` and returns a new
    `WizardState` via `dataclasses.replace`.
    """
    match action.kind:
        case ActionKind.SELECT_CHOICE:
            step_index = _require_step_index(action, spec)
            if not action.values:
                raise ValueError("select_choice action carries no values")
            step = spec.steps[step_index]
            new_state = _with_answer(state, step.key, [action.values[0]])
            return dataclasses.replace(new_state, current_step=_clamp_step(step_index + 1, spec))

        case ActionKind.SET_MULTI:
            step_index = _require_step_index(action, spec)
            step = spec.steps[step_index]
            return _with_answer(state, step.key, list(action.values))

        case ActionKind.ADD_CUSTOM:
            step_index = _require_step_index(action, spec)
            text = (action.text or "").strip()
            if not text:
                raise ValueError("add_custom action carries empty or whitespace-only text")
            step = spec.steps[step_index]
            if step.kind is StepKind.MULTI:
                return _append_answer(state, step.key, text)
            new_state = _with_answer(state, step.key, [text])
            return dataclasses.replace(new_state, current_step=_clamp_step(step_index + 1, spec))

        case ActionKind.NEXT:
            if state.current_step < len(spec.steps):
                step = spec.steps[state.current_step]
                if step.kind is StepKind.MULTI and step.min is not None:
                    recorded = len(state.answers.get(step.key, []))
                    if recorded < step.min:
                        return state
            next_step = _clamp_step(state.current_step + 1, spec)
            return dataclasses.replace(state, current_step=next_step)

        case ActionKind.BACK:
            prev_step = _clamp_step(state.current_step - 1, spec)
            return dataclasses.replace(state, current_step=prev_step)

        case ActionKind.SUBMIT:
            if state.current_step != len(spec.steps):
                return state
            return dataclasses.replace(state, status=WizardStatus.SUBMITTED)


def build_action(
    action: str, *, spec: WizardSpec, values: Sequence[str], text: str | None
) -> WizardAction:
    """Translate one already-parsed custom_id action string into a `WizardAction`.

    Raises `ValueError` for `s{n}_enter` (a modal opener, never a
    transition), any string neither grammar recognizes, or a step/option
    index out of range for `spec`.
    """
    if action == "next":
        return WizardAction(kind=ActionKind.NEXT)
    if action == "back":
        return WizardAction(kind=ActionKind.BACK)
    if action == "submit":
        return WizardAction(kind=ActionKind.SUBMIT)

    select_match = _SELECT_ACTION_RE.match(action)
    if select_match is not None:
        step_index = int(select_match.group("step"))
        if not (0 <= step_index < len(spec.steps)):
            raise ValueError(f"action {action!r} names step {step_index}, out of range for spec")
        return WizardAction(kind=ActionKind.SET_MULTI, step_index=step_index, values=list(values))

    nav_match = _NAV_ACTION_RE.match(action)
    if nav_match is not None:
        step_index = int(nav_match.group("step"))
        if not (0 <= step_index < len(spec.steps)):
            raise ValueError(f"action {action!r} names step {step_index}, out of range for spec")
        suffix = nav_match.group("suffix")
        if suffix == "enter":
            raise ValueError(
                f"action {action!r} opens a modal and is not a state transition; "
                f"build_action must not be called for it"
            )
        if suffix == "custom":
            return WizardAction(kind=ActionKind.ADD_CUSTOM, step_index=step_index, text=text)
        option_index = int(nav_match.group("option"))
        step = spec.steps[step_index]
        if not (0 <= option_index < len(step.options)):
            raise ValueError(
                f"action {action!r} names option {option_index}, out of range for "
                f"step {step_index}'s {len(step.options)} options"
            )
        return WizardAction(
            kind=ActionKind.SELECT_CHOICE,
            step_index=step_index,
            values=[step.options[option_index].value],
        )

    raise ValueError(f"action {action!r} does not match any known wizard action grammar")
