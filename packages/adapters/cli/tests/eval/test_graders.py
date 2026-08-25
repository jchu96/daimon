from __future__ import annotations

import pytest
from daimon.adapters.cli.eval.graders import mrkdwn_compliance, numeric_recompute
from daimon.adapters.cli.eval.models import ExpectedSQL


def test_mrkdwn_compliance_ignores_literal_markdown_inside_code_fences() -> None:
    result = mrkdwn_compliance.grade("```\n#  Site\n1  Alpha\n```")

    assert result["passed"] is True
    assert result["detail"] == "compatible Slack mrkdwn"


@pytest.mark.parametrize(
    ("answer", "detail"),
    [
        ("# Heading", "Markdown heading"),
        ("**bold**", "double-asterisk bold"),
        ("[docs](https://example.com)", "Markdown link"),
        ("| A | B |\n|---|---|", "Markdown pipe table"),
    ],
)
def test_mrkdwn_compliance_rejects_unsupported_prose(answer: str, detail: str) -> None:
    result = mrkdwn_compliance.grade(answer)

    assert result["passed"] is False
    assert detail in result["detail"]


def test_numeric_recompute_accepts_golden_bound_display_rounding() -> None:
    expected: ExpectedSQL = {
        "query": "SELECT 45.052",
        "answer_regex": r"(?P<value>45\.052)\s*%",
        "tolerance": 0.001,
        "display_precision": 1,
        "claim_regex": r"this year is (?P<value>-?\d+(?:\.\d+)?)\s*%",
    }

    result = numeric_recompute.grade(
        "This year is 45.1%.",
        expected_sql=expected,
        recomputed="45.052",
    )

    assert result["passed"] is True
    assert "golden display precision=1" in result["detail"]


@pytest.mark.parametrize(
    "answer",
    [
        "This year is 45.05%.",
        "This year is 45%.",
        "We discarded 45.1% from last year; this year is 12%.",
    ],
)
def test_numeric_recompute_rejects_unapproved_precision_and_distractors(answer: str) -> None:
    expected: ExpectedSQL = {
        "query": "SELECT 45.052",
        "answer_regex": r"(?P<value>45\.052)\s*%",
        "tolerance": 0.001,
        "display_precision": 1,
        "claim_regex": r"this year is (?P<value>-?\d+(?:\.\d+)?)\s*%",
    }

    result = numeric_recompute.grade(answer, expected_sql=expected, recomputed="45.052")

    assert result["passed"] is False


def test_numeric_recompute_accepts_declared_money_precision() -> None:
    expected: ExpectedSQL = {
        "query": "SELECT 300428064.13",
        "answer_regex": r"(?P<value>300,428,064\.13)",
        "tolerance": 0.01,
        "display_precision": 2,
        "claim_regex": r"Revenue was \$(?P<value>[\d,]+(?:\.\d+)?)\s*(?:M|million)",
    }

    result = numeric_recompute.grade(
        "Revenue was $300.43M.",
        expected_sql=expected,
        recomputed="300428064.13",
    )

    assert result["passed"] is True


def test_numeric_recompute_rejects_genuinely_different_value() -> None:
    expected: ExpectedSQL = {
        "query": "SELECT 10.08",
        "answer_regex": r"(?P<value>10\.08)\s*(?:M|million)",
        "tolerance": 0.01,
        "display_precision": 2,
        "claim_regex": r"Revenue was \$(?P<value>[\d,]+(?:\.\d+)?)\s*(?:M|million)",
    }

    result = numeric_recompute.grade(
        "Revenue was $9.98M.",
        expected_sql=expected,
        recomputed="10.08",
    )

    assert result["passed"] is False


def test_numeric_recompute_rejects_full_precision_distractor_outside_claim_site() -> None:
    expected: ExpectedSQL = {
        "query": "SELECT 45.052",
        "answer_regex": r"(?P<value>45\.052)\s*%",
        "tolerance": 0.001,
        "display_precision": 1,
        "claim_regex": r"this year is (?P<value>-?\d+(?:\.\d+)?)\s*%",
    }

    result = numeric_recompute.grade(
        "We discarded 45.052% from last year; this year is 12%.",
        expected_sql=expected,
        recomputed="45.052",
    )

    assert result["passed"] is False
