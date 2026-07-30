"""Screen-to-`discord.ui` renderer, and the REST post that opens a wizard run.

Posted from the MCP process (Cloud Run); every tap on the posted message is
dispatched later by the Discord bot process (worker VM). The two processes
cannot import each other (import-linter's independence contract), and the
layer they share -- `daimon.core.wizard` -- deliberately carries no Discord
SDK type, so each side owns a thin translation of the platform-neutral
`Screen` (`daimon.core.wizard.render`) into its own component tree. This
module is that translation for the posting side; a divergent copy here (a
button built in a different order, a different style mapping) would silently
desync the two renderers without either side raising, so the two are held to
producing byte-identical `to_components()` output for the same `Screen`.

A plain `discord.ui.View` cannot hold a `Container`, `TextDisplay`,
`Separator`, or `MediaGallery` -- those are components-v2 items, and
`View.add_item` raises `ValueError` for any of them unless the view is a
`discord.ui.LayoutView`. `LayoutView` is therefore the only view type built
here. The components-v2 message flag is never set by this module: the HTTP
layer sets it automatically from the view's own contents. A components-v2
message also cannot carry `content`, `embed`, `embeds`, `stickers`, or
`poll` -- which is why the screen's head text is rendered as a `TextDisplay`
component instead of message content.
"""

from __future__ import annotations

from collections.abc import Mapping

import discord
from daimon.core.wizard.render import Screen, ScreenButton
from daimon.core.wizard.state import build_custom_id

_ACCENT_COLOURS: dict[str, int] = {
    "blurple": 0x5865F2,
    "green": 0x57F287,
}

_BUTTON_STYLES: dict[str, discord.ButtonStyle] = {
    "primary": discord.ButtonStyle.primary,
    "secondary": discord.ButtonStyle.secondary,
    "success": discord.ButtonStyle.success,
}


def _build_button(short_id: str, button: ScreenButton) -> discord.ui.Button[discord.ui.LayoutView]:
    return discord.ui.Button(
        style=_BUTTON_STYLES[button.style],
        label=button.label,
        custom_id=build_custom_id(short_id, button.action),
        disabled=button.disabled,
    )


def build_wizard_view(
    screen: Screen, *, files: Mapping[str, discord.File] | None = None
) -> discord.ui.LayoutView:
    """Translate a `Screen` into a real `LayoutView`.

    Component order is a contract: head text, a small separator, an optional
    media gallery, optional body text, an optional select row, then (after a
    large invisible separator gap) one action row per button row. A screen's
    image is resolved by preferring the matching `discord.File` in `files`
    (keyed by handle) over its persisted `url`, and is omitted entirely when
    neither resolves -- this is how a later step's already-uploaded image
    rides the first post's attachment set without being displayed on it.
    """
    container: discord.ui.Container[discord.ui.LayoutView] = discord.ui.Container(
        accent_colour=_ACCENT_COLOURS[screen.accent]
    )
    container.add_item(discord.ui.TextDisplay(screen.head_text))
    container.add_item(discord.ui.Separator(visible=True, spacing=discord.SeparatorSpacing.small))

    if screen.image is not None:
        media: str | discord.File | None = None
        if files is not None and screen.image.handle is not None and screen.image.handle in files:
            media = files[screen.image.handle]
        elif screen.image.url is not None:
            media = screen.image.url
        if media is not None:
            gallery: discord.ui.MediaGallery[discord.ui.LayoutView] = discord.ui.MediaGallery()
            gallery.add_item(media=media)
            container.add_item(gallery)

    if screen.body_text is not None:
        container.add_item(discord.ui.TextDisplay(screen.body_text))

    if screen.select is not None:
        select: discord.ui.Select[discord.ui.LayoutView] = discord.ui.Select(
            custom_id=build_custom_id(screen.short_id, screen.select.action),
            placeholder=screen.select.placeholder,
            min_values=screen.select.min_values,
            max_values=screen.select.max_values,
            options=[
                discord.SelectOption(
                    label=option.label,
                    value=option.value,
                    description=option.description,
                    default=option.default,
                )
                for option in screen.select.options
            ],
        )
        select_row: discord.ui.ActionRow[discord.ui.LayoutView] = discord.ui.ActionRow()
        select_row.add_item(select)
        container.add_item(select_row)

    if screen.button_rows:
        container.add_item(
            discord.ui.Separator(visible=False, spacing=discord.SeparatorSpacing.large)
        )
        for row in screen.button_rows:
            action_row: discord.ui.ActionRow[discord.ui.LayoutView] = discord.ui.ActionRow()
            for button in row:
                action_row.add_item(_build_button(screen.short_id, button))
            container.add_item(action_row)

    view = discord.ui.LayoutView(timeout=None)
    view.add_item(container)
    return view
