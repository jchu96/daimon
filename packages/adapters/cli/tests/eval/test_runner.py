from __future__ import annotations

from typing import cast

from daimon.adapters.cli.eval.models import CaseResult, Golden
from daimon.adapters.cli.eval.runner import run_eval


class _ReplayStub:
    def __init__(self, answers: dict[str, str]) -> None:
        self.answers = answers

    async def replay(self, *, case_id: str, question: str) -> str:
        del question
        return self.answers[case_id]


class _SqlStub:
    def __init__(self, value: object = None) -> None:
        self.value = value

    async def scalar(self, query: str) -> object:
        del query
        return self.value


class _SqlByQueryStub:
    async def scalar(self, query: str) -> object:
        if query == "SELECT broken":
            raise RuntimeError("oracle unavailable")
        return "10"


def _cases(result: dict[str, object]) -> list[CaseResult]:
    return cast(list[CaseResult], result["cases"])


async def test_mrkdwn_grades_exact_slack_delivery_payload_and_keeps_raw_answer() -> None:
    raw = "See [Foo](https://en.wikipedia.org/wiki/Foo_(bar)) & ping <@U123>."
    golden: Golden = {"id": "rendered", "question": "Show the result"}

    result = await run_eval(
        run_id="run-rendered",
        goldens=[golden],
        replay=_ReplayStub({"rendered": raw}),
        sql=_SqlStub(),
    )

    case = _cases(result)[0]
    mrkdwn_grade = next(grade for grade in case["grades"] if grade["name"] == "mrkdwn_compliance")
    assert mrkdwn_grade["passed"] is False
    assert case["answer_raw"] == raw
    assert case["answer"] == (
        "See [Foo](https://en.wikipedia.org/wiki/Foo_(bar)) &amp; ping <@U123>."
    )


async def test_numeric_grader_uses_raw_answer() -> None:
    raw = "**Revenue was $300.43M.**"
    golden: Golden = {
        "id": "numeric",
        "question": "Revenue?",
        "expected_sql": {
            "query": "SELECT 300428064.13",
            "answer_regex": r"(?P<value>300,428,064\.13)",
            "tolerance": 0.01,
            "display_precision": 2,
            "claim_regex": r"Revenue was \$(?P<value>[\d,]+(?:\.\d+)?)\s*(?:M|million)",
        },
    }

    result = await run_eval(
        run_id="run-numeric",
        goldens=[golden],
        replay=_ReplayStub({"numeric": raw}),
        sql=_SqlStub("300428064.13"),
    )

    case = _cases(result)[0]
    numeric_grade = next(grade for grade in case["grades"] if grade["name"] == "numeric_recompute")
    assert numeric_grade["passed"] is True
    assert case["answer_raw"] == raw
    assert case["answer"] == raw


async def test_sql_oracle_error_fails_only_its_case_and_emits_fail_event() -> None:
    goldens: list[Golden] = [
        {
            "id": "broken",
            "question": "Broken?",
            "expected_sql": {
                "query": "SELECT broken",
                "answer_regex": r"(?P<value>10)",
                "tolerance": 0.0,
            },
        },
        {
            "id": "healthy",
            "question": "Healthy?",
            "expected_sql": {
                "query": "SELECT 10",
                "answer_regex": r"(?P<value>10)",
                "tolerance": 0.0,
            },
        },
    ]
    events: list[tuple[str, str]] = []

    result = await run_eval(
        run_id="run-sql-error",
        goldens=goldens,
        replay=_ReplayStub({"broken": "10", "healthy": "10"}),
        sql=_SqlByQueryStub(),
        concurrency=2,
        on_case=lambda case_id, state: events.append((case_id, state)),
    )

    broken, healthy = _cases(result)
    assert broken["passed"] is False
    assert broken.get("error_type") == "RuntimeError"
    assert healthy["passed"] is True
    assert ("broken", "fail") in events
    assert ("healthy", "pass") in events
