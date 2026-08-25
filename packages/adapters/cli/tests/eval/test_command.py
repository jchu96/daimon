"""`daimon eval replay` command tests."""

from __future__ import annotations

from pathlib import Path

from daimon.adapters.cli.main import app
from typer.testing import CliRunner

_FIXTURES = Path(__file__).parent / "fixtures"


def test_eval_replay_reports_checked_in_fixture_pass() -> None:
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "replay",
            "--fixtures",
            str(_FIXTURES),
            "--goldens",
            str(_FIXTURES / "goldens.jsonl"),
        ],
    )

    assert result.exit_code == 0, result.stdout + str(result.exception)
    assert "eval replay: PASS (7 recorded cases, 1 fixture runs" in result.stdout
    assert "claim-wins-prose" in result.stdout
