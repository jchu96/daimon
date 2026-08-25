"""`daimon eval` CLI commands."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, cast

import typer
from daimon.adapters.cli.eval.offline import replay_fixture_runs
from daimon.adapters.cli.eval.runner import format_grade_table, load_goldens

eval_app = typer.Typer(help="Replay typed-claims evaluation fixtures offline.")


@eval_app.command("replay")
def replay_command(
    fixtures: Annotated[
        Path,
        typer.Option("--fixtures", exists=True, file_okay=False),
    ],
    goldens: Annotated[
        Path,
        typer.Option("--goldens", exists=True, dir_okay=False),
    ],
) -> None:
    """Grade recorded answers and fail on checked-in verdict drift."""
    reports = replay_fixture_runs(fixtures, load_goldens(goldens))
    mismatches: list[str] = []
    for report in reports:
        typer.echo(format_grade_table(report.result, run_label=report.label))
        mismatches.extend(report.mismatches)
    if mismatches:
        for mismatch in mismatches:
            typer.echo(f"FIXTURE REGRESSION: {mismatch}", err=True)
        raise typer.Exit(code=1)
    total_cases = sum(cast(dict[str, int], report.result["summary"])["cases"] for report in reports)
    typer.echo(
        f"eval replay: PASS ({total_cases} recorded cases, {len(reports)} fixture runs; "
        "no network or database)"
    )
