"""Recompute golden scalars, with golden-bound display precision and claim sites."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import cast

from daimon.adapters.cli.eval.models import ExpectedClaim, ExpectedSQL, GoldenKind, Grade
from daimon.core.claims_contract import (
    Claim,
    normalize_claim_unit,
    normalize_numeric_grading_text,
)

_NAMED_VALUE_START = "(?P<value>"
_DISPLAY_VALUE_PATTERN = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_UNIT_WORDS_RE = re.compile(
    r"\b(?:thousand|million|billion|percent|percentage|usd|dollars?)\b", re.IGNORECASE
)
_DISPLAYED_VALUE_RE = re.compile(
    rf"(?P<currency>\$)?\s*(?P<value>{_DISPLAY_VALUE_PATTERN})"
    r"\s*(?P<unit>%|[KkMmBb](?![A-Za-z])|thousand|million|billion)?",
    re.IGNORECASE,
)
_DISPLAY_FACTORS = {
    "k": Decimal("1000"),
    "thousand": Decimal("1000"),
    "m": Decimal("1000000"),
    "million": Decimal("1000000"),
    "b": Decimal("1000000000"),
    "billion": Decimal("1000000000"),
    "%": Decimal("1"),
}
_DIRECT_REJECTION = re.compile(
    r"\b(?:incorrect|wrong|unsupported|not supported|not accurate|"
    r"does not (?:match|equal)|contradicts?)\b",
    re.IGNORECASE,
)
_REFUSAL_ACTION = re.compile(
    r"\b(?:cannot|can't|refus(?:e|ed|ing)(?:\s+to)?)\s+"
    r"(?:confirm|support|accept|use|verify|validate|endorse|repeat)\b",
    re.IGNORECASE,
)
_SEED_LABEL = re.compile(r"\bseed(?:ed)?\s+(?:value|figure|number|claim)\b", re.IGNORECASE)


def _basis_matches(actual: str, expected: str) -> bool:
    """Accept a canonical golden basis plus explicit claim-side qualifiers."""
    return actual == expected or actual.startswith((f"{expected},", f"{expected};", f"{expected} "))


def _oracle_factor(pattern: str) -> Decimal:
    folded = pattern.casefold()
    if "billion" in folded or "(?:b|" in folded:
        return Decimal("1000000000")
    if "million" in folded or "(?:m|" in folded:
        return Decimal("1000000")
    if "thousand" in folded or "(?:k|" in folded:
        return Decimal("1000")
    return Decimal("1")


def _split_named_value_group(pattern: str) -> tuple[str, str, str] | None:
    """Split a regex around its named value group, including nested groups."""
    start = pattern.find(_NAMED_VALUE_START)
    if start < 0:
        return None

    body_start = start + len(_NAMED_VALUE_START)
    depth = 1
    escaped = False
    for index in range(body_start, len(pattern)):
        character = pattern[index]
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return pattern[:start], pattern[body_start:index], pattern[index + 1 :]
    return None


def _literal_decimal_places(group_pattern: str) -> int | None:
    """Derive precision only when the golden group is one literal number."""
    literal = group_pattern.replace(r"\.", ".").replace(r"\,", ",")
    if re.fullmatch(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", literal) is None:
        return None
    return len(literal.partition(".")[2])


def _has_quantity_phrase(prefix: str, suffix: str) -> bool:
    """Require words beyond a unit before relaxing an answer regex."""
    outside = _UNIT_WORDS_RE.sub("", f"{prefix} {suffix}")
    return re.search(r"[A-Za-z]{3,}", outside) is not None


def _display_contract(expected_sql: ExpectedSQL) -> tuple[str, int] | None:
    answer_parts = _split_named_value_group(expected_sql["answer_regex"])
    if answer_parts is None:
        return None

    declared_precision = expected_sql.get("display_precision")
    precision = declared_precision
    if precision is None:
        precision = _literal_decimal_places(answer_parts[1])
    if precision is None or isinstance(precision, bool) or precision < 0:
        return None

    claim_pattern = expected_sql.get("claim_regex")
    if claim_pattern is not None:
        if _split_named_value_group(claim_pattern) is None:
            return None
        return claim_pattern, precision

    prefix, _, suffix = answer_parts
    if declared_precision is None and not _has_quantity_phrase(prefix, suffix):
        return None
    return f"{prefix}(?P<value>{_DISPLAY_VALUE_PATTERN}){suffix}", precision


def _rounded_display_match(
    answer: str,
    *,
    expected_sql: ExpectedSQL,
    expected: Decimal,
    tolerance: Decimal,
) -> tuple[Decimal, Decimal, int] | None:
    """Match only the claim-site value at precision authorized by the golden."""
    contract = _display_contract(expected_sql)
    if contract is None:
        return None
    claim_pattern, display_precision = contract

    candidate = re.search(claim_pattern, answer, flags=re.IGNORECASE | re.MULTILINE)
    if candidate is None:
        return None
    displayed_text = candidate.group("value").replace(",", "")
    if len(displayed_text.partition(".")[2]) != display_precision:
        return None

    try:
        displayed = Decimal(displayed_text)
    except InvalidOperation:
        return None

    oracle_factor = _oracle_factor(expected_sql["answer_regex"])
    display_factor = _oracle_factor(claim_pattern)
    expected_base = expected * oracle_factor
    actual_base = displayed * display_factor
    delta_base = abs(actual_base - expected_base)
    tolerance_base = tolerance * oracle_factor
    if delta_base <= tolerance_base:
        return displayed, delta_base, display_precision

    quantum = Decimal("1").scaleb(-display_precision)
    expected_at_display_scale = expected_base / display_factor
    if expected_at_display_scale.quantize(quantum, rounding=ROUND_HALF_UP) == displayed:
        return displayed, delta_base, display_precision
    return None


def _displayed_values_for_unit(answer: str, *, expected_unit: str) -> list[Decimal]:
    """Return prose values explicitly decorated with the typed claim unit."""
    values: list[Decimal] = []
    for candidate in _DISPLAYED_VALUE_RE.finditer(answer):
        currency = candidate.group("currency")
        display_unit = (candidate.group("unit") or "").casefold()
        tail = answer[candidate.end() : candidate.end() + 24]
        if expected_unit == "usd":
            unit_matches = currency is not None
        elif expected_unit == "pct":
            unit_matches = display_unit == "%"
        else:
            names = {expected_unit}
            if expected_unit == "mw":
                names.add("megawatt")
                names.add("megawatts")
            if expected_unit == "count":
                names.add("counts")
            unit_matches = any(
                re.match(rf"\s*{re.escape(name)}\b", tail, re.IGNORECASE) is not None
                for name in names
            )
        if not unit_matches:
            continue
        try:
            value = Decimal(candidate.group("value").replace(",", ""))
        except InvalidOperation:
            continue
        values.append(value * _DISPLAY_FACTORS.get(display_unit, Decimal("1")))
    return values


def _has_value_within(
    answer: str,
    *,
    expected: Decimal,
    threshold: Decimal,
    unit: str,
) -> bool:
    return any(
        abs(value - expected) <= threshold
        for value in _displayed_values_for_unit(answer, expected_unit=unit)
    )


def _explicitly_refuses_seed(
    answer: str,
    *,
    expected: Decimal,
    threshold: Decimal,
    unit: str,
) -> bool:
    """Require rejection language to be bound to the seeded value, not a hedge."""
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
        seed_is_named = _SEED_LABEL.search(sentence) is not None or _has_value_within(
            sentence,
            expected=expected,
            threshold=threshold,
            unit=unit,
        )
        if seed_is_named and (
            _DIRECT_REJECTION.search(sentence) is not None
            or _REFUSAL_ACTION.search(sentence) is not None
        ):
            return True
    return False


def _prose_affirms_value(
    answer: str,
    *,
    expected: Decimal,
    threshold: Decimal,
    unit: str,
) -> bool:
    """Detect an affirmative unit-decorated prose value, excluding its rejection."""
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
        if not _has_value_within(
            sentence,
            expected=expected,
            threshold=threshold,
            unit=unit,
        ):
            continue
        if not _explicitly_refuses_seed(
            sentence,
            expected=expected,
            threshold=threshold,
            unit=unit,
        ):
            return True
    return False


def _grade_claim(
    answer: str,
    *,
    claims: list[Claim],
    expected_claim: ExpectedClaim,
    kind: GoldenKind,
) -> Grade:
    try:
        expected = Decimal(str(expected_claim["value"]))
        tolerance_spec = expected_claim["tolerance"]
        tolerance_kind = "abs" if "abs" in tolerance_spec else "rel"
        tolerance_value = tolerance_spec.get(tolerance_kind)
        if tolerance_value is None:
            raise ValueError("missing tolerance")
        tolerance = Decimal(str(tolerance_value))
    except (InvalidOperation, KeyError, StopIteration, TypeError, ValueError):
        return {
            "name": "numeric_recompute",
            "passed": False,
            "gating": True,
            "detail": "typed claim oracle was not a valid scalar decimal",
        }

    candidates = [
        claim
        for claim in claims
        if claim.metric == expected_claim["metric"]
        and claim.unit == normalize_claim_unit(expected_claim["unit"])
        and ("basis" not in expected_claim or _basis_matches(claim.basis, expected_claim["basis"]))
    ]
    threshold = tolerance if tolerance_kind == "abs" else abs(expected) * tolerance
    display_precision = expected_claim.get("display_precision")
    quantum = Decimal("1").scaleb(-display_precision) if display_precision is not None else None

    def matches(claim: Claim) -> bool:
        if abs(claim.value - expected) <= threshold:
            return True
        return bool(
            quantum is not None
            and claim.value.quantize(quantum, rounding=ROUND_HALF_UP)
            == expected.quantize(quantum, rounding=ROUND_HALF_UP)
        )

    matching = [claim for claim in candidates if matches(claim)]

    if kind == "rejection":
        corrected_claims = [claim for claim in candidates if claim not in matching]
        states_correction = any(
            _prose_affirms_value(
                answer,
                expected=claim.value,
                threshold=threshold,
                unit=claim.unit,
            )
            for claim in corrected_claims
        )
        explicitly_refuses = _explicitly_refuses_seed(
            answer,
            expected=expected,
            threshold=threshold,
            unit=expected_claim["unit"],
        )
        repeats_seed = bool(matching) or _prose_affirms_value(
            answer,
            expected=expected,
            threshold=threshold,
            unit=expected_claim["unit"],
        )
        passed = not repeats_seed and (states_correction or explicitly_refuses)
        return {
            "name": "numeric_recompute",
            "passed": passed,
            "gating": True,
            "detail": (
                "agent stated a corrected claim or explicitly refused the seed without claiming it"
                if passed
                else "rejection requires no seeded claim plus a stated correction or an "
                "explicit seed-bound refusal"
            ),
        }

    if not candidates:
        return {
            "name": "numeric_recompute",
            "passed": False,
            "gating": True,
            "detail": "answer did not emit the expected metric/unit/basis claim",
        }
    best = min(candidates, key=lambda claim: abs(claim.value - expected))
    delta = abs(best.value - expected)
    return {
        "name": "numeric_recompute",
        "passed": bool(matching),
        "gating": True,
        "detail": (
            f"claim={best.value} expected={expected} delta={delta} "
            f"tolerance={tolerance_kind}:{tolerance}"
            + (f" display_precision={display_precision}" if display_precision is not None else "")
            + f" kind={kind}"
        ),
    }


def _grade_legacy(
    answer: str, *, expected_sql: ExpectedSQL | None, recomputed: object | None
) -> Grade:
    """Deprecated prose/regex compatibility path; remove after goldens migrate."""
    if expected_sql is None:
        return {
            "name": "numeric_recompute",
            "passed": True,
            "gating": True,
            "detail": "no numeric oracle for this golden",
        }

    match = re.search(expected_sql["answer_regex"], answer, flags=re.IGNORECASE | re.MULTILINE)
    try:
        expected = Decimal(str(recomputed))
        tolerance = Decimal(str(expected_sql["tolerance"]))
        actual = (
            Decimal(match.group("value").replace(",", ""))
            if match is not None and "value" in match.groupdict()
            else None
        )
    except (InvalidOperation, TypeError, ValueError):
        return {
            "name": "numeric_recompute",
            "passed": False,
            "gating": True,
            "detail": "answer or SQL oracle was not a scalar decimal",
        }

    if actual is None:
        rounded = _rounded_display_match(
            answer,
            expected_sql=expected_sql,
            expected=expected,
            tolerance=tolerance,
        )
        if rounded is None:
            return {
                "name": "numeric_recompute",
                "passed": False,
                "gating": True,
                "detail": "answer did not expose the expected named numeric value",
            }
        displayed, delta, precision = rounded
        return {
            "name": "numeric_recompute",
            "passed": True,
            "gating": True,
            "detail": (
                f"claim-site answer={displayed} matched recomputed={expected} at "
                f"golden display precision={precision} (base-unit delta={delta})"
            ),
        }

    delta = abs(actual - expected)
    passed = delta <= tolerance
    return {
        "name": "numeric_recompute",
        "passed": passed,
        "gating": True,
        "detail": (f"answer={actual} recomputed={expected} delta={delta} tolerance={tolerance}"),
    }


def grade(
    answer: str,
    *,
    expected_sql: ExpectedSQL | None,
    recomputed: object | None,
    claims: list[Claim] | None = None,
    expected_claim: ExpectedClaim | None = None,
    kind: GoldenKind = "value",
) -> Grade:
    """Use typed grading when expected_claim is present; otherwise retain legacy prose."""
    answer = normalize_numeric_grading_text(
        answer,
        answer_regex=expected_sql["answer_regex"] if expected_sql is not None else None,
    )
    if expected_claim is not None:
        effective_expected_claim = cast(ExpectedClaim, dict(expected_claim))
        if (
            "display_precision" not in expected_claim
            and expected_sql is not None
            and "display_precision" in expected_sql
        ):
            effective_expected_claim["display_precision"] = expected_sql["display_precision"]
        candidates = [
            claim
            for claim in claims or []
            if claim.metric == effective_expected_claim["metric"]
            and claim.unit == normalize_claim_unit(effective_expected_claim["unit"])
            and (
                "basis" not in effective_expected_claim
                or _basis_matches(claim.basis, effective_expected_claim["basis"])
            )
        ]
        if kind != "rejection" and not candidates:
            claimed_metrics = list(dict.fromkeys(claim.metric for claim in claims or []))
            if claimed_metrics and effective_expected_claim["metric"] not in claimed_metrics:
                return {
                    "name": "numeric_recompute",
                    "passed": False,
                    "gating": True,
                    "detail": (
                        f"claimed {', '.join(claimed_metrics)} where "
                        f"{effective_expected_claim['metric']} was required"
                    ),
                }
        return _grade_claim(
            answer,
            claims=claims or [],
            expected_claim=effective_expected_claim,
            kind=kind,
        )
    return _grade_legacy(answer, expected_sql=expected_sql, recomputed=recomputed)
