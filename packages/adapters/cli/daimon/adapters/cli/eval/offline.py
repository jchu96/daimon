"""Offline replay of recorded eval answers through the production graders."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from daimon.adapters.cli.eval.models import CaseResult, Golden
from daimon.adapters.cli.eval.runner import grade_answer, validate_golden
from daimon.core.claims_contract import MeasureRegistry

type RawSource = Literal["recorded", "stripped", "real_carrier", "unavailable"]


@dataclass(frozen=True)
class FixtureRunResult:
    """One fixture run and any verdict drift against its checked-in expectations."""

    label: str
    result: dict[str, object]
    mismatches: tuple[str, ...]


def _load_object(path: Path) -> dict[str, object]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path}: invalid fixture JSON: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return cast(dict[str, object], loaded)


def _child_file(root: Path, relative: object, *, location: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{location}: baseline must be a non-empty relative path")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError(f"{location}: baseline escapes the fixtures directory")
    if not candidate.is_file():
        raise ValueError(f"{location}: baseline not found: {candidate}")
    return candidate


def _string_map(value: object, *, location: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{location}: expected an object with string keys")
    raw = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in raw):
        raise ValueError(f"{location}: expected an object with string keys")
    return cast(dict[str, object], raw)


def _validate_raw_source(
    value: object,
    *,
    case_ids: set[str],
    location: str,
) -> dict[str, RawSource]:
    raw = _string_map(value, location=location)
    if set(raw) != case_ids:
        raise ValueError(f"{location}: raw_source keys must equal baseline case ids")
    allowed = {"recorded", "stripped", "real_carrier", "unavailable"}
    invalid = {case_id: status for case_id, status in raw.items() if status not in allowed}
    if invalid:
        raise ValueError(f"{location}: invalid source status: {invalid}")
    return cast(dict[str, RawSource], raw)


def _validate_contract_truth(
    value: object,
    *,
    case_ids: set[str],
    location: str,
) -> dict[str, dict[str, str]]:
    raw = _string_map(value, location=location)
    truth: dict[str, dict[str, str]] = {}
    for case_id, truth_value in raw.items():
        if case_id not in case_ids:
            raise ValueError(f"{location}: unknown contract-truth case {case_id}")
        item = _string_map(truth_value, location=f"{location}.{case_id}")
        if item.get("color") not in {"GREEN", "RED"}:
            raise ValueError(f"{location}.{case_id}: color must be GREEN or RED")
        if not isinstance(item.get("reason"), str) or not item["reason"]:
            raise ValueError(f"{location}.{case_id}: reason must be non-empty")
        truth[case_id] = cast(dict[str, str], item)
    return truth


def _validate_expected(
    value: object,
    *,
    case_ids: set[str],
    location: str,
) -> dict[str, dict[str, bool]]:
    raw = _string_map(value, location=location)
    if set(raw) != case_ids:
        raise ValueError(f"{location}: expected_verdicts keys must equal baseline case ids")
    expected: dict[str, dict[str, bool]] = {}
    for case_id, verdicts_value in raw.items():
        verdicts = _string_map(verdicts_value, location=f"{location}.{case_id}")
        if not verdicts or not all(isinstance(item, bool) for item in verdicts.values()):
            raise ValueError(f"{location}.{case_id}: verdicts must be non-empty booleans")
        expected[case_id] = cast(dict[str, bool], verdicts)
    return expected


def _claims_errors(value: object) -> list[str] | None:
    """Normalize recorded scalar/list error evidence to the current list contract."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        items = cast(list[object], value)
        if all(isinstance(item, str) for item in items):
            return cast(list[str], items)
    return None


def replay_fixture_runs(
    fixtures: Path,
    goldens: list[Golden],
    *,
    measure_registry: MeasureRegistry | None = None,
) -> list[FixtureRunResult]:
    """Replay every recorded fixture answer without constructing network or DB clients."""
    root = fixtures.resolve()
    manifest_path = root / "manifest.json"
    manifest = _load_object(manifest_path)
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{manifest_path}: schema_version must be 1")
    runs_value = manifest.get("runs")
    if not isinstance(runs_value, list) or not runs_value:
        raise ValueError(f"{manifest_path}: runs must be a non-empty list")
    runs = cast(list[object], runs_value)

    golden_by_id = {golden["id"]: golden for golden in goldens}
    reports: list[FixtureRunResult] = []
    for index, run_value in enumerate(runs):
        location = f"{manifest_path}:runs[{index}]"
        run = _string_map(run_value, location=location)
        label = run.get("id")
        if not isinstance(label, str) or not label:
            raise ValueError(f"{location}: id must be a non-empty string")
        override_values = _string_map(
            run.get("golden_overrides", {}), location=f"{location}.golden_overrides"
        )
        overrides: dict[str, Golden] = {}
        for override_id, override_value in override_values.items():
            override_payload = _string_map(
                override_value,
                location=f"{location}.golden_overrides.{override_id}",
            )
            override = validate_golden(
                override_payload,
                location=f"{location}.golden_overrides.{override_id}",
                measure_registry=measure_registry,
            )
            if override["id"] != override_id:
                raise ValueError(
                    f"{location}.golden_overrides.{override_id}: id must match its key"
                )
            overrides[override_id] = override
        scope = run.get("golden_scope", "all")
        if scope not in {"all", "overrides_only"}:
            raise ValueError(f"{location}.golden_scope: must be all or overrides_only")
        run_goldens = dict(golden_by_id if scope == "all" else {})
        run_goldens.update(overrides)
        baseline_path = _child_file(root, run.get("baseline"), location=location)
        baseline = _load_object(baseline_path)
        cases_value = baseline.get("cases")
        if not isinstance(cases_value, list) or not cases_value:
            raise ValueError(f"{baseline_path}: cases must be a non-empty list")
        fixture_cases = cast(list[object], cases_value)
        baseline_cases: list[dict[str, object]] = []
        for case_index, case_value in enumerate(fixture_cases):
            baseline_cases.append(
                _string_map(case_value, location=f"{baseline_path}:cases[{case_index}]")
            )
        case_ids = {str(case.get("id")) for case in baseline_cases}
        if case_ids != set(run_goldens):
            raise ValueError(
                f"{location}: baseline case ids must equal golden ids; "
                f"baseline={sorted(case_ids)} goldens={sorted(run_goldens)}"
            )

        raw_source = _validate_raw_source(
            run.get("raw_source"), case_ids=case_ids, location=f"{location}.raw_source"
        )
        contract_truth = _validate_contract_truth(
            run.get("contract_truth", {}),
            case_ids=case_ids,
            location=f"{location}.contract_truth",
        )
        expected = _validate_expected(
            run.get("expected_verdicts"),
            case_ids=case_ids,
            location=f"{location}.expected_verdicts",
        )
        sql_results = _string_map(run.get("sql_results", {}), location=f"{location}.sql_results")
        suspect_value = run.get("suspect_labels", [])
        if not isinstance(suspect_value, list):
            raise ValueError(f"{location}.suspect_labels: expected a list of strings")
        suspect_items = cast(list[object], suspect_value)
        if not all(isinstance(item, str) for item in suspect_items):
            raise ValueError(f"{location}.suspect_labels: expected a list of strings")
        suspect_labels = set(cast(list[str], suspect_items))

        results: list[CaseResult] = []
        mismatches: list[str] = []
        for baseline_case in baseline_cases:
            case_id = str(baseline_case["id"])
            golden = run_goldens[case_id]
            if golden.get("expected_sql") is not None and case_id not in sql_results:
                raise ValueError(f"{location}.sql_results: missing oracle for {case_id}")
            raw = baseline_case.get("answer_raw")
            if not isinstance(raw, str):
                raise ValueError(f"{baseline_path}:{case_id}: answer_raw must be a string")
            status = raw_source[case_id]
            if status in {"recorded", "real_carrier"} and not raw:
                raise ValueError(f"{baseline_path}:{case_id}: {status} answer_raw is empty")
            if status == "unavailable" and raw:
                raise ValueError(f"{baseline_path}:{case_id}: unavailable answer_raw is non-empty")
            claims_error = _claims_errors(baseline_case.get("claims_error"))
            if status == "stripped" and claims_error is None:
                raise ValueError(
                    f"{baseline_path}:{case_id}: stripped carrier requires claims_error evidence"
                )
            if golden.get("expected_claim") is not None and status != "real_carrier":
                raise ValueError(
                    f"{baseline_path}:{case_id}: typed claims may be graded only from a "
                    "real_carrier source"
                )
            error_type = baseline_case.get("error_type")
            if error_type is not None and not isinstance(error_type, str):
                raise ValueError(f"{baseline_path}:{case_id}: error_type must be a string")
            result = grade_answer(
                golden=golden,
                answer_source=raw,
                recomputed=sql_results.get(case_id),
                suspect_labels=suspect_labels,
                replay_error_type=error_type,
                claims_error_override=claims_error,
                measure_registry=measure_registry,
            )
            results.append(result)
            actual = {grade["name"]: grade["passed"] for grade in result["grades"]}
            if set(actual) != set(expected[case_id]):
                mismatches.append(
                    f"{label}/{case_id}: check set changed "
                    f"expected={sorted(expected[case_id])} actual={sorted(actual)}"
                )
            for check in sorted(set(actual) & set(expected[case_id])):
                if actual[check] is not expected[case_id][check]:
                    mismatches.append(
                        f"{label}/{case_id}/{check}: expected "
                        f"{'PASS' if expected[case_id][check] else 'FAIL'} got "
                        f"{'PASS' if actual[check] else 'FAIL'}"
                    )
            if case_id in contract_truth:
                expected_pass = contract_truth[case_id]["color"] == "GREEN"
                if result["passed"] is not expected_pass:
                    mismatches.append(
                        f"{label}/{case_id}/contract_truth: expected "
                        f"{contract_truth[case_id]['color']} got "
                        f"{'GREEN' if result['passed'] else 'RED'}"
                    )

        passed_count = sum(case["passed"] for case in results)
        result_document: dict[str, object] = {
            "schema_version": 1,
            "run_id": f"fixture:{label}",
            "generated_at": baseline.get("generated_at", "recorded fixture"),
            "summary": {
                "cases": len(results),
                "passed": passed_count,
                "failed": len(results) - passed_count,
                "all_gating_passed": passed_count == len(results),
            },
            "cases": results,
        }
        reports.append(
            FixtureRunResult(label=label, result=result_document, mismatches=tuple(mismatches))
        )
    return reports
