"""Offline replay fixtures exercise the production grading path."""

from __future__ import annotations

from pathlib import Path

from daimon.adapters.cli.eval.offline import replay_fixture_runs
from daimon.adapters.cli.eval.runner import load_goldens

_FIXTURES = Path(__file__).parent / "fixtures"


def test_generic_fixture_manifest_replays_without_drift() -> None:
    reports = replay_fixture_runs(_FIXTURES, load_goldens(_FIXTURES / "goldens.jsonl"))

    assert len(reports) == 1
    assert reports[0].mismatches == ()
    summary = reports[0].result["summary"]
    assert isinstance(summary, dict)
    assert summary == {
        "cases": 7,
        "passed": 4,
        "failed": 3,
        "all_gating_passed": False,
    }
