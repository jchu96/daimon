from __future__ import annotations

from daimon.core.slack_mrkdwn import escape_mrkdwn_preserving_mentions


def test_delivery_transform_preserves_balanced_parenthesis_markdown_link() -> None:
    source = "See [Foo](https://en.wikipedia.org/wiki/Foo_(bar))."

    assert escape_mrkdwn_preserving_mentions(source) == source


def test_delivery_transform_does_not_leave_javascript_link_parser_residue() -> None:
    source = "[click](javascript:alert(1))"

    assert escape_mrkdwn_preserving_mentions(source) == source


def test_delivery_transform_leaves_an_already_fenced_table_unchanged() -> None:
    source = "```\n| # | Site |\n|---|---|\n| 1 | Alpha |\n```"

    assert escape_mrkdwn_preserving_mentions(source) == source


def test_delivery_transform_escapes_controls_and_preserves_mentions() -> None:
    source = "5 < 6 & ping <@U123> in <#C123|general>; not <!channel>"

    assert escape_mrkdwn_preserving_mentions(source) == (
        "5 &lt; 6 &amp; ping <@U123> in <#C123|general>; not &lt;!channel&gt;"
    )
