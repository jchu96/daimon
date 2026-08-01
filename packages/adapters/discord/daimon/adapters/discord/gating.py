"""Discord-specific message gating -- pure pre-DB checks."""

from __future__ import annotations


def should_process_message(
    *,
    author_is_bot: bool,
    author_id: str,
    bot_mentioned: bool,
    guild_id: str | None,
    self_user_id: str | None = None,
    qa_bot_user_id: str | None = None,
) -> bool:
    """Pre-DB gate checks for on_message. Returns True if message passes all non-DB filters."""
    if author_is_bot and not _is_allowed_bot_author(
        author_id=author_id,
        self_user_id=self_user_id,
        qa_bot_user_id=qa_bot_user_id,
    ):
        return False
    if not bot_mentioned:
        return False
    return guild_id is not None


def _is_allowed_bot_author(
    *,
    author_id: str,
    self_user_id: str | None,
    qa_bot_user_id: str | None,
) -> bool:
    """Whether a bot-authored mention may start a turn.

    Bots are rejected by default. A deployment may allow-list exactly one bot
    -- an automated QA driver -- so a harness can exercise the mention path
    end-to-end without a human. Discord forbids bots from invoking application
    commands or component interactions, so a mention is the only turn trigger
    reachable by automation at all.

    Allow-listing the bot's own id is refused rather than honored: daimon's
    answers mention the caller, so a self-allow-list would have every turn
    trigger the next one without bound.
    """
    if qa_bot_user_id is None:
        return False
    if self_user_id is not None and qa_bot_user_id == self_user_id:
        return False
    return author_id == qa_bot_user_id
