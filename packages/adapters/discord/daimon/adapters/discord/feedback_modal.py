"""FeedbackModal -- the single-field free-text form the feedback button opens.

Opened from `FeedbackButton.callback` after a click on a persistent button
delivered into a direct message. The form collects exactly ONE field: the
free-text criticism itself. There is no category picker and no second input
-- the row being annotated is already fixed by the `feedback_id` carried in
the button's `custom_id`, so the user never retypes anything else.

Write path: `on_submit` defers ephemerally, rejects whitespace-only input
server-side (the client already enforces `required=True`, but that is not
trusted alone), then calls `daimon.core.stores.message_feedback.
attach_feedback_text`, which gates the write on the clicking user's own
`platform_user_id`. A `None` return covers both "the row was purged" and
"this row belongs to someone else" -- the reply text deliberately does not
distinguish the two, so a click carrying someone else's row id learns
nothing about whether that row exists.

Logging discipline: the submitted text is somebody's unsolicited criticism,
not a secret, but it belongs in exactly one place -- the database row. It
never enters a log record, a `custom_id`, an embed, or any non-ephemeral
message. Every log line below carries the feedback row id only. This mirrors
the hygiene guarantees `credential_modals.py` already documents for a
different reason (there it is secrecy; here it is that this is unsolicited,
personal criticism that should not multiply across the observability
pipeline).
"""

from __future__ import annotations

import uuid

import structlog
from daimon.adapters.discord.runtime import DiscordRuntime
from daimon.core.stores.message_feedback import attach_feedback_text

import discord

_log = structlog.get_logger()

_EMPTY_TEXT = "Feedback cannot be empty -- try again."
_NO_LONGER_AVAILABLE = "This feedback request is no longer available."
_SUBMIT_FAILED = "Something went wrong submitting your feedback -- please try again."
_THANKS = "Thanks -- your feedback has been recorded."


class FeedbackModal(discord.ui.Modal, title="What went wrong?"):
    """Single-field free-text form, written through the ownership-predicated store call."""

    def __init__(self, *, runtime: DiscordRuntime, feedback_id: uuid.UUID) -> None:
        super().__init__()
        self._runtime = runtime
        self._feedback_id = feedback_id
        self.text_input: discord.ui.TextInput[FeedbackModal] = discord.ui.TextInput(
            label="What went wrong?",
            style=discord.TextStyle.paragraph,
            required=True,
            max_length=4000,
        )
        self.add_item(self.text_input)

    async def on_submit(self, interaction: discord.Interaction) -> None:
        await interaction.response.defer(ephemeral=True, thinking=True)
        raw_text = str(self.text_input.value or "")

        if not raw_text.strip():
            await interaction.followup.send(_EMPTY_TEXT, ephemeral=True)
            return

        async with self._runtime.sessionmaker() as session, session.begin():
            updated_row = await attach_feedback_text(
                session,
                feedback_id=self._feedback_id,
                platform_user_id=str(interaction.user.id),
                feedback_text=raw_text,
            )

        if updated_row is None:
            _log.info("feedback_modal.no_longer_available", feedback_id=str(self._feedback_id))
            await interaction.followup.send(_NO_LONGER_AVAILABLE, ephemeral=True)
            return

        _log.info("feedback_modal.submit", feedback_id=str(self._feedback_id))
        await interaction.followup.send(_THANKS, ephemeral=True)

    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        """Adapter-boundary catch-all: a modal submission has no other error surface.

        Unlike `from_custom_id`/`interaction_check`/`callback` on a
        `DynamicItem` (see `feedback_button.py`), discord.py does NOT swallow
        exceptions raised from `on_submit` -- it routes them here instead. But
        leaving the default `on_error` behavior (log only) would still leave
        the person staring at a form that silently failed, so this overrides
        it to also send an apology.
        """
        _log.exception("feedback_modal.on_error", err_type=type(error).__name__)
        if interaction.response.is_done():
            await interaction.followup.send(_SUBMIT_FAILED, ephemeral=True)
        else:
            await interaction.response.send_message(_SUBMIT_FAILED, ephemeral=True)
