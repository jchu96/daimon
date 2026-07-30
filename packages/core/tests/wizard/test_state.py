"""Unit tests for the wizard's domain state and custom_id grammar (no I/O, no DB)."""

from __future__ import annotations

import dataclasses
import importlib.util

import pytest
from daimon.core.credential_requests import CUSTOM_ID_PATTERN as CREDENTIAL_CUSTOM_ID_PATTERN
from daimon.core.credential_requests import build_custom_id as build_credential_custom_id
from daimon.core.credential_requests import mint_request_token
from daimon.core.wizard.state import (
    NAV_CUSTOM_ID_PATTERN,
    SELECT_CUSTOM_ID_PATTERN,
    SUBMIT_CUSTOM_ID_PATTERN,
    ActionKind,
    WizardAction,
    WizardState,
    build_custom_id,
    new_short_id,
)

# Every action string the renderer can emit, spanning the low, middle and
# high end of the two-digit step-index range MAX_STEPS (20) allows.
_NAV_ACTIONS = [
    "next",
    "back",
    "s0_c0",
    "s0_c24",
    "s5_enter",
    "s12_custom",
    "s19_c9",
]
_SELECT_ACTIONS = ["s0_sel", "s5_sel", "s19_sel"]
_SUBMIT_ACTIONS = ["submit"]
_ALL_ACTIONS = _NAV_ACTIONS + _SELECT_ACTIONS + _SUBMIT_ACTIONS

_ALL_PATTERNS = (NAV_CUSTOM_ID_PATTERN, SELECT_CUSTOM_ID_PATTERN, SUBMIT_CUSTOM_ID_PATTERN)


def test_new_short_id_returns_eight_base62_characters() -> None:
    short_id = new_short_id()

    assert len(short_id) == 8, "short id must be exactly SHORT_ID_LEN characters"
    assert short_id.isalnum(), "short id must be drawn from the base62 alphabet"


def test_new_short_id_two_calls_differ() -> None:
    first = new_short_id()
    second = new_short_id()

    assert first != second, "two short-id mints must never collide in practice"


@pytest.mark.parametrize("action", _NAV_ACTIONS)
def test_nav_action_round_trips_through_build_custom_id(action: str) -> None:
    short_id = new_short_id()

    custom_id = build_custom_id(short_id, action)
    match = NAV_CUSTOM_ID_PATTERN.fullmatch(custom_id)

    assert match is not None, "nav pattern must fullmatch a custom_id it minted for this action"
    assert match["short_id"] == short_id, "the short_id group must round-trip the input short_id"
    assert match["action"] == action, "the action group must round-trip the input action"


@pytest.mark.parametrize("action", _SELECT_ACTIONS)
def test_select_action_round_trips_through_build_custom_id(action: str) -> None:
    short_id = new_short_id()

    custom_id = build_custom_id(short_id, action)
    match = SELECT_CUSTOM_ID_PATTERN.fullmatch(custom_id)

    assert match is not None, "select pattern must fullmatch a custom_id it minted for this action"
    assert match["short_id"] == short_id, "the short_id group must round-trip the input short_id"
    assert match["action"] == action, "the action group must round-trip the input action"


def test_submit_action_round_trips_through_build_custom_id() -> None:
    short_id = new_short_id()

    custom_id = build_custom_id(short_id, "submit")
    match = SUBMIT_CUSTOM_ID_PATTERN.fullmatch(custom_id)

    assert match is not None, "submit pattern must fullmatch a custom_id it minted for submit"
    assert match["short_id"] == short_id, "the short_id group must round-trip the input short_id"


def test_build_custom_id_raises_for_unroutable_action() -> None:
    with pytest.raises(ValueError, match="does not fullmatch"):
        build_custom_id(new_short_id(), "totally_unroutable_action")


def test_build_custom_id_raises_when_short_id_blows_the_char_budget() -> None:
    oversized_short_id = "x" * 200

    with pytest.raises(ValueError, match="exceeding"):
        build_custom_id(oversized_short_id, "next")


@pytest.mark.parametrize("action", _ALL_ACTIONS)
def test_action_fullmatches_exactly_one_of_the_three_grammars(action: str) -> None:
    short_id = new_short_id()
    custom_id = build_custom_id(short_id, action)

    match_count = sum(1 for pattern in _ALL_PATTERNS if pattern.fullmatch(custom_id))

    assert match_count == 1, (
        f"custom_id {custom_id!r} for action {action!r} must fullmatch exactly one "
        f"grammar -- discord.py fires every registered dynamic-item template whose "
        f"pattern fullmatches an incoming custom_id with no early break, so an "
        f"overlap double-dispatches one tap"
    )


def test_wizard_grammars_disjoint_from_credential_request_grammar() -> None:
    token = mint_request_token()
    credential_custom_id = build_credential_custom_id(token)

    for pattern in _ALL_PATTERNS:
        assert pattern.fullmatch(credential_custom_id) is None, (
            "a credential-request custom_id must never fullmatch a wizard grammar, "
            "or discord.py would double-dispatch a credential-button tap into the "
            "wizard's turn-spawning path"
        )


@pytest.mark.parametrize("action", _ALL_ACTIONS)
def test_credential_request_grammar_disjoint_from_wizard_custom_id(action: str) -> None:
    wizard_custom_id = build_custom_id(new_short_id(), action)

    assert CREDENTIAL_CUSTOM_ID_PATTERN.fullmatch(wizard_custom_id) is None, (
        "a wizard custom_id must never fullmatch the credential-request grammar, "
        "or discord.py would double-dispatch a wizard tap into the credential path"
    )


def test_wizard_grammars_disjoint_from_message_feedback_grammar_when_present() -> None:
    if importlib.util.find_spec("daimon.core.message_feedback") is None:
        pytest.skip("message_feedback module has not landed in this tree yet")

    from daimon.core.message_feedback import CUSTOM_ID_PATTERN as FEEDBACK_CUSTOM_ID_PATTERN

    for action in _ALL_ACTIONS:
        wizard_custom_id = build_custom_id(new_short_id(), action)

        assert FEEDBACK_CUSTOM_ID_PATTERN.fullmatch(wizard_custom_id) is None, (
            "a wizard custom_id must never fullmatch the message-feedback grammar, "
            "or discord.py would double-dispatch a wizard tap into the feedback path"
        )


def test_wizard_state_is_frozen_and_replace_produces_a_new_instance() -> None:
    state = WizardState(short_id="abcd1234")

    updated = dataclasses.replace(state, current_step=1)

    assert updated.current_step == 1, "dataclasses.replace must apply the field override"
    assert state.current_step == 0, "the original instance must be untouched"
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.current_step = 2  # type: ignore[misc]


def test_wizard_action_is_frozen_and_replace_produces_a_new_instance() -> None:
    action = WizardAction(kind=ActionKind.NEXT)

    updated = dataclasses.replace(action, kind=ActionKind.BACK)

    assert updated.kind == ActionKind.BACK, "dataclasses.replace must apply the field override"
    assert action.kind == ActionKind.NEXT, "the original instance must be untouched"
    with pytest.raises(dataclasses.FrozenInstanceError):
        action.kind = ActionKind.SUBMIT  # type: ignore[misc]
