"""Recompute golden scalars with golden-bound display precision and claim sites."""

from __future__ import annotations

import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

from daimon.adapters.cli.eval.models import ExpectedSQL, Grade

_NAMED_VALUE_START = "(?P<value>"
_DISPLAY_VALUE_PATTERN = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_UNIT_WORDS_RE = re.compile(
    r"\b(?:thousand|million|billion|percent|percentage|usd|dollars?)\b", re.IGNORECASE
)


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

    precision = expected_sql.get("display_precision")
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
    if not _has_quantity_phrase(prefix, suffix):
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


def grade(answer: str, *, expected_sql: ExpectedSQL | None, recomputed: object | None) -> Grade:
    """Grade the exact oracle, then an explicitly golden-bounded display form."""
    if expected_sql is None:
        return {
            "name": "numeric_recompute",
            "passed": True,
            "gating": True,
            "detail": "no numeric oracle for this golden",
        }

    claim_pattern = expected_sql.get("claim_regex")
    claim_match = (
        re.search(claim_pattern, answer, flags=re.IGNORECASE | re.MULTILINE)
        if claim_pattern is not None
        else None
    )
    exact_scope = (
        claim_match.group(0)
        if claim_pattern is not None and claim_match is not None
        else ("" if claim_pattern is not None else answer)
    )
    match = re.search(
        expected_sql["answer_regex"],
        exact_scope,
        flags=re.IGNORECASE | re.MULTILINE,
    )
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
        "detail": f"answer={actual} recomputed={expected} delta={delta} tolerance={tolerance}",
    }
