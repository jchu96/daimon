"""Pre-DB message gate: who may start a turn by mention."""

from __future__ import annotations

from daimon.adapters.discord.gating import should_process_message

DAIMON_ID = "111"
HUMAN_ID = "222"
QA_BOT_ID = "333"
OTHER_BOT_ID = "444"


def test_gate_admits_mentioning_human_in_guild() -> None:
    assert should_process_message(
        author_is_bot=False,
        author_id=HUMAN_ID,
        bot_mentioned=True,
        guild_id="g1",
        self_user_id=DAIMON_ID,
    ), "a human's explicit mention in a guild is the ordinary turn trigger"


def test_gate_rejects_human_when_not_mentioned() -> None:
    assert not should_process_message(
        author_is_bot=False,
        author_id=HUMAN_ID,
        bot_mentioned=False,
        guild_id="g1",
        self_user_id=DAIMON_ID,
    ), "an unmentioned message must not start a turn"


def test_gate_rejects_human_mention_in_dm() -> None:
    assert not should_process_message(
        author_is_bot=False,
        author_id=HUMAN_ID,
        bot_mentioned=True,
        guild_id=None,
        self_user_id=DAIMON_ID,
    ), "DMs have no tenant, so they must not start a turn"


def test_gate_rejects_bot_author_when_no_qa_bot_configured() -> None:
    assert not should_process_message(
        author_is_bot=True,
        author_id=OTHER_BOT_ID,
        bot_mentioned=True,
        guild_id="g1",
        self_user_id=DAIMON_ID,
        qa_bot_user_id=None,
    ), "with no allow-list, every bot-authored mention is rejected"


def test_gate_admits_allow_listed_qa_bot() -> None:
    assert should_process_message(
        author_is_bot=True,
        author_id=QA_BOT_ID,
        bot_mentioned=True,
        guild_id="g1",
        self_user_id=DAIMON_ID,
        qa_bot_user_id=QA_BOT_ID,
    ), "the allow-listed QA bot's mention must start a turn like a human's"


def test_gate_rejects_unlisted_bot_when_qa_bot_configured() -> None:
    assert not should_process_message(
        author_is_bot=True,
        author_id=OTHER_BOT_ID,
        bot_mentioned=True,
        guild_id="g1",
        self_user_id=DAIMON_ID,
        qa_bot_user_id=QA_BOT_ID,
    ), "the allow-list admits exactly one bot, not bots in general"


def test_gate_rejects_self_authored_mention_when_allow_listed_to_own_id() -> None:
    assert not should_process_message(
        author_is_bot=True,
        author_id=DAIMON_ID,
        bot_mentioned=True,
        guild_id="g1",
        self_user_id=DAIMON_ID,
        qa_bot_user_id=DAIMON_ID,
    ), "allow-listing daimon's own id must not arm an unbounded self-trigger loop"


def test_gate_rejects_qa_bot_when_not_mentioned() -> None:
    assert not should_process_message(
        author_is_bot=True,
        author_id=QA_BOT_ID,
        bot_mentioned=False,
        guild_id="g1",
        self_user_id=DAIMON_ID,
        qa_bot_user_id=QA_BOT_ID,
    ), "the allow-list relaxes the bot-author check only, not the mention requirement"


def test_gate_rejects_qa_bot_in_dm() -> None:
    assert not should_process_message(
        author_is_bot=True,
        author_id=QA_BOT_ID,
        bot_mentioned=True,
        guild_id=None,
        self_user_id=DAIMON_ID,
        qa_bot_user_id=QA_BOT_ID,
    ), "the allow-list relaxes the bot-author check only, not the guild requirement"
