"""Validation and grading for live or recorded evaluation answers."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import cast

from daimon.adapters.cli.eval.graders import numeric_recompute
from daimon.adapters.cli.eval.models import (
    CaseResult,
    ClaimPayload,
    Golden,
    Grade,
)
from daimon.core.claims import (
    ClaimsBlockError,
    parse_claims_block,
    serialize_claims,
    strip_claims_blocks,
)
from daimon.core.claims_contract import (
    ALLOWED_CLAIM_UNITS,
    MeasureRegistry,
    missing_required_phrase_patterns,
)


def _validate_expected_claim(
    payload: dict[str, object],
    *,
    location: str,
    measure_registry: MeasureRegistry | None = None,
) -> None:
    required = {"metric", "value", "tolerance", "unit"}
    allowed = required | {"basis", "display_precision"}
    if not required <= payload.keys() or not payload.keys() <= allowed:
        raise ValueError(f"{location}: expected_claim has missing or unknown fields")
    metric = payload["metric"]
    if not isinstance(metric, str) or not metric:
        raise ValueError(f"{location}: expected_claim.metric must be a non-empty string")
    if measure_registry is not None and metric not in measure_registry.ids:
        raise ValueError(
            f"{location}: expected_claim.metric is not a registered measure id: {metric!r}"
        )
    unit = payload["unit"]
    if not isinstance(unit, str) or unit not in ALLOWED_CLAIM_UNITS:
        raise ValueError(
            f"{location}: expected_claim.unit must be one of {sorted(ALLOWED_CLAIM_UNITS)}"
        )
    basis = payload.get("basis")
    if basis is not None and (not isinstance(basis, str) or not basis):
        raise ValueError(f"{location}: expected_claim.basis must be a non-empty string")
    precision = payload.get("display_precision")
    if precision is not None and (
        isinstance(precision, bool) or not isinstance(precision, int) or precision < 0
    ):
        raise ValueError(
            f"{location}: expected_claim.display_precision must be a non-negative integer"
        )
    value = payload["value"]
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        raise ValueError(f"{location}: expected_claim.value must be numeric")
    try:
        if not Decimal(str(value)).is_finite():
            raise ValueError(f"{location}: expected_claim.value must be finite")
    except InvalidOperation as error:
        raise ValueError(f"{location}: expected_claim.value must be numeric") from error
    tolerance = payload["tolerance"]
    if not isinstance(tolerance, dict):
        raise ValueError(f"{location}: tolerance must contain exactly one of abs or rel")
    typed_tolerance = cast(dict[str, object], tolerance)
    if set(typed_tolerance) not in ({"abs"}, {"rel"}):
        raise ValueError(f"{location}: tolerance must contain exactly one of abs or rel")
    tolerance_value = next(iter(typed_tolerance.values()))
    if isinstance(tolerance_value, bool) or not isinstance(tolerance_value, int | float):
        raise ValueError(f"{location}: tolerance must be numeric")
    decimal_tolerance = Decimal(str(tolerance_value))
    if not decimal_tolerance.is_finite() or decimal_tolerance < 0:
        raise ValueError(f"{location}: tolerance must be finite and non-negative")


def _validate_expected_sql(payload: object, *, location: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{location}: expected_sql must be an object")
    expected = cast(dict[str, object], payload)
    required = {"query", "answer_regex", "tolerance"}
    allowed = required | {"display_precision", "claim_regex"}
    if not required <= expected.keys() or not expected.keys() <= allowed:
        raise ValueError(f"{location}: expected_sql has missing or unknown fields")
    if not isinstance(expected["query"], str) or not expected["query"]:
        raise ValueError(f"{location}: expected_sql.query must be a non-empty string")
    answer_regex = expected["answer_regex"]
    if not isinstance(answer_regex, str):
        raise ValueError(f"{location}: expected_sql.answer_regex must be a string")
    try:
        re.compile(answer_regex)
    except re.error as error:
        raise ValueError(f"{location}: invalid expected_sql.answer_regex: {error}") from error
    tolerance = expected["tolerance"]
    if isinstance(tolerance, bool) or not isinstance(tolerance, int | float) or tolerance < 0:
        raise ValueError(f"{location}: expected_sql.tolerance must be non-negative")


def validate_golden(
    payload: dict[str, object],
    *,
    location: str,
    measure_registry: MeasureRegistry | None = None,
) -> Golden:
    """Validate one decoded golden before any answer is graded."""
    if not isinstance(payload.get("id"), str) or not payload["id"]:
        raise ValueError(f"{location}: id and question are required")
    if not isinstance(payload.get("question"), str) or not payload["question"]:
        raise ValueError(f"{location}: id and question are required")
    if not isinstance(payload.get("expected_properties"), dict):
        raise ValueError(f"{location}: expected_properties must be an object")

    expected_claim = payload.get("expected_claim")
    kind = payload.get("kind")
    if expected_claim is not None:
        if not isinstance(expected_claim, dict):
            raise ValueError(f"{location}: expected_claim must be an object")
        if kind not in {"value", "rejection", "trend"}:
            raise ValueError(f"{location}: typed golden kind must be value, rejection, or trend")
        _validate_expected_claim(
            cast(dict[str, object], expected_claim),
            location=location,
            measure_registry=measure_registry,
        )
    elif kind is not None:
        raise ValueError(f"{location}: kind requires expected_claim")
    if payload.get("expected_sql") is not None:
        _validate_expected_sql(payload["expected_sql"], location=location)
    if expected_claim is None and payload.get("expected_sql") is None:
        raise ValueError(f"{location}: expected_claim or expected_sql is required")

    properties = cast(dict[str, object], payload["expected_properties"])
    allowed_properties = {"required_phrases", "banned_phrases"}
    if not properties.keys() <= allowed_properties:
        raise ValueError(f"{location}: unsupported expected_properties field")
    for key in ("required_phrases", "banned_phrases"):
        patterns = properties.get(key, [])
        if not isinstance(patterns, list) or not all(
            isinstance(pattern, str) for pattern in cast(list[object], patterns)
        ):
            raise ValueError(f"{location}: expected_properties.{key} must be regex strings")
        for pattern in cast(list[str], patterns):
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(
                    f"{location}: expected_properties.{key} has invalid regex {pattern!r}: {error}"
                ) from error
    return cast(Golden, payload)


def load_goldens(path: Path, *, measure_registry: MeasureRegistry | None = None) -> list[Golden]:
    """Load and validate newline-delimited JSON goldens."""
    goldens: list[Golden] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(parsed, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        goldens.append(
            validate_golden(
                cast(dict[str, object], parsed),
                location=f"{path}:{line_number}",
                measure_registry=measure_registry,
            )
        )
    if not goldens:
        raise ValueError(f"{path}: no goldens found")
    ids = [golden["id"] for golden in goldens]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: golden ids must be unique")
    return goldens


def grade_answer(
    *,
    golden: Golden,
    answer_source: str,
    recomputed: object,
    suspect_labels: set[str],  # retained for replay-fixture compatibility
    replay_error_type: str | None = None,
    claims_error_override: str | list[str] | None = None,
    measure_registry: MeasureRegistry | None = None,
) -> CaseResult:
    """Grade one answer, preserving its raw carrier as audit evidence."""
    del suspect_labels
    claims_error: list[str] = []
    try:
        extraction = parse_claims_block(answer_source, registry=measure_registry)
        answer_content, claims = extraction.prose, extraction.claims
        claims_error = list(extraction.errors)
    except ClaimsBlockError as error:
        answer_content = answer_source
        claims = []
        claims_error = [str(error)]
    answer = strip_claims_blocks(answer_content)
    properties = golden["expected_properties"]
    grades: list[Grade] = [
        {
            "name": "replay_completion",
            "passed": replay_error_type is None,
            "gating": True,
            "detail": "agent replay completed" if replay_error_type is None else replay_error_type,
        },
        numeric_recompute.grade(
            answer,
            expected_sql=golden.get("expected_sql"),
            recomputed=recomputed,
            claims=claims,
            expected_claim=golden.get("expected_claim"),
            kind=golden.get("kind", "value"),
        ),
    ]
    required = list(properties.get("required_phrases", []))
    if required:
        missing = missing_required_phrase_patterns(answer, required)
        grades.append(
            {
                "name": "required_properties",
                "passed": not missing,
                "gating": False,
                "detail": (
                    "advisory phrase expectations present"
                    if not missing
                    else f"advisory phrase expectations missing: {missing}"
                ),
            }
        )
    banned = list(properties.get("banned_phrases", []))
    if banned:
        matched = [
            pattern for pattern in banned if not missing_required_phrase_patterns(answer, [pattern])
        ]
        grades.append(
            {
                "name": "banned_properties",
                "passed": not matched,
                "gating": True,
                "detail": (
                    "no golden-declared bans matched"
                    if not matched
                    else f"matched golden-declared bans: {matched}"
                ),
            }
        )

    case_result: CaseResult = {
        "id": golden["id"],
        "question": golden["question"],
        "answer": answer,
        "answer_raw": answer_source,
        "claims": cast(list[ClaimPayload], serialize_claims(claims)),
        "passed": all(grade["passed"] for grade in grades if grade["gating"]),
        "grades": grades,
    }
    if replay_error_type is not None:
        case_result["error_type"] = replay_error_type
    override_errors = (
        [claims_error_override] if isinstance(claims_error_override, str) else claims_error_override
    )
    effective_errors = claims_error or override_errors
    if effective_errors:
        case_result["claims_error"] = effective_errors
    return case_result


def format_grade_table(result: dict[str, object], *, run_label: str | None = None) -> str:
    """Render a stable per-case/per-check table."""
    rows: list[tuple[str, str, str, str]] = []
    for case in cast(list[dict[str, object]], result.get("cases", [])):
        for grade in cast(list[dict[str, object]], case.get("grades", [])):
            rows.append(
                (
                    str(case["id"]),
                    str(grade["name"]),
                    "gate" if grade.get("gating") is True else "advisory",
                    "PASS" if grade.get("passed") is True else "FAIL",
                )
            )
    headers = ("case", "check", "mode", "verdict")
    widths = [max(len(headers[index]), *(len(row[index]) for row in rows)) for index in range(4)]

    def render(row: tuple[str, str, str, str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    lines = [f"eval grades: {run_label}" if run_label else "eval grades", render(headers)]
    lines.append("-+-".join("-" * width for width in widths))
    lines.extend(render(row) for row in rows)
    return "\n".join(lines)
