"""Executable guard that the MCP-side and Discord-side wizard renderers agree.

The two thin Screen-to-LayoutView translations
(`daimon.adapters.mcp.tools.discord._wizard.build_wizard_view` and
`daimon.adapters.discord.wizard_render.build_wizard_view`) are duplicated on
purpose -- the MCP adapter and the Discord adapter cannot import each other
(import-linter's independence contract) -- but duplication without a guard
drifts silently: a button built in a different order, or a different accent
colour, on either side would desync what a user sees after a tap from what
they saw on the original post, with nothing else in the test suite catching
it. This module imports both renderers directly (legitimate in a test,
which is outside the packages the import-linter contracts govern) and
asserts they produce byte-identical serialized components for the same
`Screen`. This test is that guard.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from daimon.adapters.discord.wizard_render import (
    build_wizard_view as discord_build_wizard_view,
)
from daimon.adapters.mcp.tools.discord._wizard import (
    build_wizard_view as mcp_build_wizard_view,
)
from daimon.core.wizard.render import Screen, to_screen
from daimon.core.wizard.spec import Option, Step, StepKind, WizardSpec
from daimon.core.wizard.state import WizardState, WizardStatus

_SHORT_ID = "abcd1234"


def _choice_spec() -> WizardSpec:
    return WizardSpec(
        prompt="Configure the pipeline",
        steps=[
            Step(
                key="stage",
                question="Pick a stage",
                kind=StepKind.CHOICE,
                options=[Option(label="dev", value="dev"), Option(label="prod", value="prod")],
            )
        ],
    )


def _choice_screen() -> Screen:
    return to_screen(_choice_spec(), WizardState(short_id=_SHORT_ID))


def _multi_spec() -> WizardSpec:
    return WizardSpec(
        prompt="Configure the pipeline",
        steps=[
            Step(
                key="regions",
                question="Pick regions",
                kind=StepKind.MULTI,
                options=[
                    Option(label="us", value="us"),
                    Option(label="eu", value="eu"),
                    Option(label="apac", value="apac"),
                ],
                min=1,
                max=2,
            )
        ],
    )


def _multi_screen_with_selection() -> Screen:
    return to_screen(
        _multi_spec(), WizardState(short_id=_SHORT_ID, answers={"regions": ["us", "eu"]})
    )


def _multi_screen_no_selection() -> Screen:
    return to_screen(_multi_spec(), WizardState(short_id=_SHORT_ID))


def _text_spec() -> WizardSpec:
    return WizardSpec(
        prompt="Configure the pipeline",
        steps=[Step(key="notes", question="Anything else?", kind=StepKind.TEXT)],
    )


def _text_screen() -> Screen:
    return to_screen(_text_spec(), WizardState(short_id=_SHORT_ID))


def _image_spec() -> WizardSpec:
    return WizardSpec(
        prompt="Configure the pipeline",
        steps=[
            Step(
                key="stage",
                question="Pick a stage",
                kind=StepKind.CHOICE,
                options=[Option(label="dev", value="dev")],
                image_handle="img-1.png",
                image_url="https://cdn.discordapp.com/attachments/1/2/img-1.png?ex=abc123",
            )
        ],
    )


def _image_screen() -> Screen:
    return to_screen(_image_spec(), WizardState(short_id=_SHORT_ID))


def _review_spec() -> WizardSpec:
    return WizardSpec(
        prompt="Configure the pipeline",
        steps=[
            Step(
                key="stage",
                question="Pick a stage",
                kind=StepKind.CHOICE,
                options=[Option(label="dev", value="dev")],
            )
        ],
    )


def _review_screen() -> Screen:
    return to_screen(
        _review_spec(),
        WizardState(short_id=_SHORT_ID, current_step=1, answers={"stage": ["dev"]}),
    )


def _submitted_screen() -> Screen:
    return to_screen(
        _review_spec(),
        WizardState(
            short_id=_SHORT_ID,
            current_step=1,
            answers={"stage": ["dev"]},
            status=WizardStatus.SUBMITTED,
        ),
    )


_SCREEN_CASES: list[tuple[str, Callable[[], Screen]]] = [
    ("choice step", _choice_screen),
    ("multi step with an existing selection", _multi_screen_with_selection),
    ("multi step with no selection", _multi_screen_no_selection),
    ("text step", _text_screen),
    ("step carrying an image with a persisted url", _image_screen),
    ("review screen", _review_screen),
    ("submitted screen", _submitted_screen),
]


@pytest.mark.parametrize(
    ("name", "build_screen"), _SCREEN_CASES, ids=[case[0] for case in _SCREEN_CASES]
)
def test_renderers_agree_component_for_component(
    name: str, build_screen: Callable[[], Screen]
) -> None:
    screen = build_screen()
    # No `files` on the posting side: both renderers must resolve the image
    # (when present) by its persisted url, matching how a re-render works.
    mcp_components = mcp_build_wizard_view(screen, files=None).to_components()
    discord_components = discord_build_wizard_view(screen).to_components()
    assert mcp_components == discord_components, (
        f"the two renderers diverged on the {name!r} screen -- their serialized "
        f"components must be identical:\nmcp={mcp_components!r}\n"
        f"discord={discord_components!r}"
    )


def test_both_renderers_report_components_v2() -> None:
    screen = _choice_screen()
    assert mcp_build_wizard_view(screen, files=None).has_components_v2(), (
        "the mcp-side view must report components-v2"
    )
    assert discord_build_wizard_view(screen).has_components_v2(), (
        "the discord-side view must report components-v2"
    )
