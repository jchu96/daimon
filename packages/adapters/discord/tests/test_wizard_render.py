"""Tests for wizard_render.build_wizard_view -- Screen to LayoutView translation."""

from __future__ import annotations

from collections.abc import Callable

import discord
import pytest
from daimon.adapters.discord.wizard_render import build_wizard_view
from daimon.core.wizard.apply import apply
from daimon.core.wizard.render import to_screen
from daimon.core.wizard.spec import Option, Step, StepKind, WizardSpec
from daimon.core.wizard.state import (
    NAV_CUSTOM_ID_PATTERN,
    SELECT_CUSTOM_ID_PATTERN,
    SUBMIT_CUSTOM_ID_PATTERN,
    ActionKind,
    WizardAction,
    WizardState,
)

_SHORT_ID = "abcd1234"


def _choice_spec(*, image_handle: str | None = None, image_url: str | None = None) -> WizardSpec:
    return WizardSpec(
        prompt="Pick a favourite",
        steps=[
            Step(
                key="colour",
                question="Which colour?",
                kind=StepKind.CHOICE,
                options=[Option(label="Red", value="red"), Option(label="Blue", value="blue")],
                image_handle=image_handle,
                image_url=image_url,
            )
        ],
    )


def _multi_spec() -> WizardSpec:
    return WizardSpec(
        prompt="Pick some toppings",
        steps=[
            Step(
                key="toppings",
                question="Which toppings?",
                kind=StepKind.MULTI,
                options=[
                    Option(label="Cheese", value="cheese"),
                    Option(label="Olives", value="olives"),
                ],
            )
        ],
    )


def _container(view: discord.ui.LayoutView) -> discord.ui.Container[discord.ui.LayoutView]:
    children = view.children
    assert len(children) == 1, "a wizard view must hold exactly one top-level item"
    container = children[0]
    assert isinstance(container, discord.ui.Container), "the one top-level item must be a Container"
    return container


def test_choice_screen_renders_one_container_with_expected_child_ordering() -> None:
    spec = _choice_spec()
    screen = to_screen(spec, WizardState(short_id=_SHORT_ID))

    view = build_wizard_view(screen)

    container = _container(view)
    kinds = [type(item) for item in container.children]
    assert kinds[0] is discord.ui.TextDisplay, "head text must render first"
    assert kinds[1] is discord.ui.Separator, "a small separator must follow the head text"
    # Choice step: no image, no select, no body text -> next is the pre-nav
    # gap separator, then one ActionRow per button row (options + nav).
    assert kinds[2] is discord.ui.Separator, (
        "a large invisible separator must precede the button rows"
    )
    assert all(kind is discord.ui.ActionRow for kind in kinds[3:]), (
        "every remaining child must be an ActionRow of buttons"
    )


def test_select_screen_renders_exactly_one_action_row_before_nav_row() -> None:
    spec = _multi_spec()
    screen = to_screen(spec, WizardState(short_id=_SHORT_ID))
    assert screen.select is not None, "a multi step's screen must carry a select"

    view = build_wizard_view(screen)

    container = _container(view)
    kinds = [type(item) for item in container.children]
    select_row_indices = [
        index
        for index, item in enumerate(container.children)
        if isinstance(item, discord.ui.ActionRow)
        and any(isinstance(c, discord.ui.Select) for c in item.children)
    ]
    assert len(select_row_indices) == 1, "exactly one action row must hold the select"
    action_row_indices = [index for index, kind in enumerate(kinds) if kind is discord.ui.ActionRow]
    assert action_row_indices[0] == select_row_indices[0], (
        "the select's action row must come before the nav row's action row"
    )
    assert len(action_row_indices) == 2, "a multi step renders the select row plus one nav row"


def test_image_with_url_renders_a_media_gallery() -> None:
    spec = _choice_spec(image_url="https://cdn.example.com/step0.png")
    screen = to_screen(spec, WizardState(short_id=_SHORT_ID))
    assert screen.image is not None and screen.image.url is not None

    view = build_wizard_view(screen)

    container = _container(view)
    galleries = [item for item in container.children if isinstance(item, discord.ui.MediaGallery)]
    assert len(galleries) == 1, "a screen image with a url must render exactly one media gallery"


def test_image_with_only_a_handle_and_no_url_renders_no_gallery() -> None:
    spec = _choice_spec(image_handle="handle-only-no-url")
    screen = to_screen(spec, WizardState(short_id=_SHORT_ID))
    assert screen.image is not None and screen.image.handle is not None
    assert screen.image.url is None, "this test's premise is a handle with no persisted url"

    view = build_wizard_view(screen)

    container = _container(view)
    galleries = [item for item in container.children if isinstance(item, discord.ui.MediaGallery)]
    assert galleries == [], (
        "this process cannot resolve a bare handle to media -- it never touches "
        "the posting process's file store, so the image must be omitted"
    )


def test_collapsed_screen_renders_no_action_rows_at_all() -> None:
    spec = _choice_spec()
    state = WizardState(short_id=_SHORT_ID, current_step=1, answers={"colour": ["red"]})

    submitted = apply(state, spec, WizardAction(kind=ActionKind.SUBMIT))
    screen = to_screen(spec, submitted)
    assert screen.button_rows == (), "a submitted run's screen must carry no button rows"

    view = build_wizard_view(screen)

    container = _container(view)
    action_rows = [item for item in container.children if isinstance(item, discord.ui.ActionRow)]
    assert action_rows == [], "a collapsed (submitted) screen must render no action rows"


@pytest.mark.parametrize(
    "spec_factory",
    [_choice_spec, _multi_spec],
)
def test_has_components_v2_is_true_for_every_produced_view(
    spec_factory: Callable[[], WizardSpec],
) -> None:
    spec = spec_factory()
    screen = to_screen(spec, WizardState(short_id=_SHORT_ID))

    view = build_wizard_view(screen)

    assert view.has_components_v2(), "every wizard view must carry the components-v2 flag"


def test_custom_id_on_every_button_and_select_fullmatches_a_core_pattern() -> None:
    spec = _multi_spec()
    screen = to_screen(spec, WizardState(short_id=_SHORT_ID))

    view = build_wizard_view(screen)

    container = _container(view)
    for item in container.children:
        if isinstance(item, discord.ui.ActionRow):
            for child in item.children:
                custom_id = getattr(child, "custom_id", None)
                assert custom_id is not None, "every dispatchable item must carry a custom_id"
                matched = (
                    NAV_CUSTOM_ID_PATTERN.fullmatch(custom_id)
                    or SELECT_CUSTOM_ID_PATTERN.fullmatch(custom_id)
                    or SUBMIT_CUSTOM_ID_PATTERN.fullmatch(custom_id)
                )
                assert matched is not None, (
                    f"custom_id {custom_id!r} must fullmatch one of the three core "
                    "wizard dispatch grammars"
                )
