"""Generic golden replay orchestration with injectable external boundaries."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Protocol

from daimon.adapters.cli.eval.graders import mrkdwn_compliance, numeric_recompute
from daimon.adapters.cli.eval.models import CaseResult, ExpectedSQL, Golden, Grade
from daimon.core.slack_mrkdwn import escape_mrkdwn_preserving_mentions


class ReplayBackend(Protocol):
    """Produce one raw agent answer per golden."""

    async def replay(self, *, case_id: str, question: str) -> str: ...


class SqlExecutor(Protocol):
    """Run the read-only scalar query used by the numeric grader."""

    async def scalar(self, query: str) -> object: ...


async def run_eval(
    *,
    run_id: str,
    goldens: Iterable[Golden],
    replay: ReplayBackend,
    sql: SqlExecutor,
    case_timeout_s: float = 600.0,
    concurrency: int = 1,
    on_case: Callable[[str, str], None] | None = None,
) -> dict[str, object]:
    """Replay and grade goldens without coupling to a live agent or database."""
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    cases = list(goldens)
    semaphore = asyncio.Semaphore(concurrency)

    async def run_case(golden: Golden) -> CaseResult:
        async with semaphore:
            if on_case is not None:
                on_case(golden["id"], "start")

            replay_error_type: str | None = None
            try:
                answer_raw = await asyncio.wait_for(
                    replay.replay(case_id=golden["id"], question=golden["question"]),
                    timeout=case_timeout_s,
                )
            except Exception as error:  # noqa: BLE001 - one case must not abort the run
                answer_raw = ""
                replay_error_type = type(error).__name__

            answer = escape_mrkdwn_preserving_mentions(answer_raw)
            expected_sql: ExpectedSQL | None = golden.get("expected_sql")
            recomputed: object | None = None
            sql_error_type: str | None = None
            if expected_sql is not None:
                try:
                    recomputed = await sql.scalar(expected_sql["query"])
                except Exception as error:  # noqa: BLE001 - one oracle must not abort the run
                    sql_error_type = type(error).__name__
            grades: list[Grade] = [
                {
                    "name": "replay_completion",
                    "passed": replay_error_type is None,
                    "gating": True,
                    "detail": (
                        "agent replay completed" if replay_error_type is None else replay_error_type
                    ),
                },
                mrkdwn_compliance.grade(answer),
                numeric_recompute.grade(
                    answer_raw,
                    expected_sql=expected_sql,
                    recomputed=recomputed,
                ),
            ]
            passed = all(grade["passed"] for grade in grades if grade["gating"])
            result: CaseResult = {
                "id": golden["id"],
                "question": golden["question"],
                "answer": answer,
                "answer_raw": answer_raw,
                "passed": passed,
                "grades": grades,
            }
            error_type = replay_error_type or sql_error_type
            if error_type is not None:
                result["error_type"] = error_type
            if on_case is not None:
                on_case(golden["id"], "pass" if passed else "fail")
            return result

    results = list(await asyncio.gather(*(run_case(golden) for golden in cases)))
    passed_count = sum(case["passed"] for case in results)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "summary": {
            "cases": len(results),
            "passed": passed_count,
            "failed": len(results) - passed_count,
            "all_gating_passed": passed_count == len(results),
        },
        "cases": results,
    }
