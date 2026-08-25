"""Vendor-neutral contracts for machine-readable claims and measure registries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, field_validator

CLAIMS_FENCE_LANGUAGE = "claims"
CLAIMS_FENCE_PATTERN = rf"```{re.escape(CLAIMS_FENCE_LANGUAGE)}\s*\n(?P<body>.*?)```"
CLAIMS_INSTRUCTION_PLACEHOLDER = "{{DAIMON_CLAIMS_CONTRACT}}"
MEASURE_REGISTRY_PLACEHOLDER = "{{DAIMON_MEASURE_REGISTRY}}"

ALLOWED_CLAIM_UNITS = frozenset({"usd", "pct", "count", "days", "mw"})
CLAIM_UNIT_ALIASES = {"$": "usd", "%": "pct"}

_REGISTRY_ID = re.compile(r"[a-z][a-z0-9_.-]*\Z")
_RELATION_COLUMN = re.compile(r"(?:[a-z_][a-z0-9_]*\.){2}[a-z_][a-z0-9_]*\Z")
_DECIMAL_TEXT = re.compile(r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)\Z")
_NUMERIC_TOKEN = r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_NUMERIC_RANGE = re.compile(
    rf"{_NUMERIC_TOKEN}\s*[-\u2010\u2011\u2012\u2013\u2014]\s*{_NUMERIC_TOKEN}"
)
_CLAIMS_TEXT_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00a0": " ",
        "\u2009": " ",
        "\u202f": " ",
    }
)


def normalize_claims_text(value: str) -> str:
    """Normalize contract-significant Unicode punctuation and spacing."""
    return value.translate(_CLAIMS_TEXT_TRANSLATION)


def normalize_numeric_grading_text(value: str, *, answer_regex: str | None) -> str:
    """Normalize text while hiding range operands from scalar extraction.

    A scalar golden must not read either endpoint of a displayed numeric range
    as a standalone value. A range remains visible only when that golden's
    answer regex spans the complete range. This numeric-only view does not
    change the general text normalization used by phrase checks or claim dates.
    """
    normalized = normalize_claims_text(value)
    explicit_spans = (
        [
            match.span()
            for match in re.finditer(answer_regex, normalized, re.IGNORECASE | re.MULTILINE)
        ]
        if answer_regex is not None
        else []
    )
    characters = list(normalized)
    for numeric_range in _NUMERIC_RANGE.finditer(value):
        start, end = numeric_range.span()
        if any(
            match_start <= start and match_end >= end for match_start, match_end in explicit_spans
        ):
            continue
        for index in range(start, end):
            if characters[index].isdigit() or characters[index] in ",.":
                characters[index] = " "
    return "".join(characters)


@dataclass(frozen=True)
class MeasureDefinition:
    """One deployment-supplied measure exposed to a claims carrier."""

    id: str
    source: str
    basis: str
    unit: str
    derivation: str | None = None
    routing: str | None = None


@dataclass(frozen=True)
class MeasureRegistry:
    """An immutable deployment-supplied set of registered measures."""

    measures: tuple[MeasureDefinition, ...]

    @property
    def ids(self) -> frozenset[str]:
        return frozenset(measure.id for measure in self.measures)


def validate_measure_registry(registry: MeasureRegistry) -> None:
    """Fail closed on malformed deployment registry data."""
    ids = [measure.id for measure in registry.measures]
    if not ids:
        raise ValueError("measure registry must contain at least one measure")
    if len(ids) != len(set(ids)):
        raise ValueError("measure registry ids must be unique")
    for measure in registry.measures:
        if _REGISTRY_ID.fullmatch(measure.id) is None:
            raise ValueError(f"invalid measure registry id: {measure.id!r}")
        if _RELATION_COLUMN.fullmatch(measure.source) is None:
            raise ValueError(
                f"measure registry source must be relation.column: {measure.id}={measure.source}"
            )
        if not measure.basis.strip():
            raise ValueError(f"measure registry source/basis must be non-empty: {measure.id}")
        if measure.unit not in ALLOWED_CLAIM_UNITS:
            raise ValueError(f"measure registry unit is not allowed: {measure.id}={measure.unit}")


def normalize_claim_unit(value: str) -> str:
    """Return the canonical claims unit for a textual unit."""
    normalized = normalize_claims_text(value).strip().casefold()
    return CLAIM_UNIT_ALIASES.get(normalized, normalized)


class Claim(BaseModel):
    """One registry-addressed numeric claim emitted by an agent."""

    model_config = ConfigDict(extra="ignore", strict=True)

    metric: str
    value: Decimal
    unit: str
    basis: str
    as_of: date | None
    source: str | None

    @field_validator("metric", mode="before")
    @classmethod
    def _normalize_metric(cls, value: object) -> object:
        return normalize_claims_text(value).strip().casefold() if isinstance(value, str) else value

    @field_validator("metric")
    @classmethod
    def _metric_is_registry_id(cls, value: str) -> str:
        if _REGISTRY_ID.fullmatch(value) is None:
            raise ValueError("metric must be a lowercase registry id")
        return value

    @field_validator("unit", mode="before")
    @classmethod
    def _normalize_unit(cls, value: object) -> object:
        return normalize_claim_unit(value) if isinstance(value, str) else value

    @field_validator("unit")
    @classmethod
    def _unit_is_allowed(cls, value: str) -> str:
        if value not in ALLOWED_CLAIM_UNITS:
            raise ValueError(f"{value} not allowed")
        return value

    @field_validator("basis", mode="before")
    @classmethod
    def _basis_is_present(cls, value: object) -> object:
        if isinstance(value, str):
            value = normalize_claims_text(value)
        if not isinstance(value, str):
            return value
        if not value.strip() or value != value.strip():
            raise ValueError("basis must be non-empty without surrounding whitespace")
        return value

    @field_validator("source", mode="before")
    @classmethod
    def _source_is_present_when_set(cls, value: object) -> object:
        if isinstance(value, str):
            value = normalize_claims_text(value)
        if value is not None and not isinstance(value, str):
            return value
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError("source must be non-empty without surrounding whitespace")
        return value

    @field_validator("value", mode="before")
    @classmethod
    def _normalize_value(cls, value: object) -> object:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            normalized = Decimal(str(value))
            if normalized.is_finite() and normalized == normalized.to_integral():
                return Decimal(int(normalized))
            return normalized
        if not isinstance(value, str):
            return value
        normalized = normalize_claims_text(value).strip()
        if normalized.startswith("$"):
            normalized = normalized[1:].lstrip()
        if _DECIMAL_TEXT.fullmatch(normalized) is None:
            raise ValueError("value must be numeric")
        try:
            return Decimal(normalized.replace(",", ""))
        except ArithmeticError as error:
            raise ValueError("value must be numeric") from error

    @field_validator("value")
    @classmethod
    def _value_is_finite(cls, value: Decimal) -> Decimal:
        if not value.is_finite():
            raise ValueError("value must be finite")
        return value

    @field_validator("as_of", mode="before")
    @classmethod
    def _normalize_as_of(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        normalized = normalize_claims_text(value)
        try:
            return date.fromisoformat(normalized)
        except ValueError:
            return normalized


def render_claims_instruction(registry: MeasureRegistry | None = None) -> str:
    """Generate the agent-facing instruction from the executable contract."""
    fields = ", ".join(Claim.model_fields)
    units = ", ".join(sorted(ALLOWED_CLAIM_UNITS))
    aliases = ", ".join(
        f"{source!r} to {target!r}" for source, target in sorted(CLAIM_UNIT_ALIASES.items())
    )
    metric_instruction = (
        "Use one of the registered measure IDs; an unknown ID is rejected. "
        if registry is not None
        else "Use the registry measure id when one exists, otherwise a lowercase "
        "registry-style descriptor. "
    )
    return (
        "For every headline number, end the final answer with exactly one trailing "
        f"`{CLAIMS_FENCE_LANGUAGE}` JSON fence containing a non-empty list of "
        f"{{{fields}}}. Keep each headline number in the prose; the claims block "
        f"supplements it. {metric_instruction}"
        f"Emit canonical units only from this closed set: {units}. The parser normalizes "
        f"{aliases}; do not rely on those aliases in generated output. Use null for unknown "
        "as_of/source. If you emit another fenced payload, put the claims fence last."
    )


def render_measure_registry(registry: MeasureRegistry) -> str:
    """Generate the complete agent-facing measure list from the typed registry."""
    lines = [
        "<!-- BEGIN GENERATED MEASURE REGISTRY -->",
        "These are the complete registered measure IDs. Use the exact ID and canonical basis",
        "in numeric claims; recompute from the named source at the answer's stated data cut.",
    ]
    for measure in registry.measures:
        detail = (
            f"- **{measure.id}** — source: `{measure.source}`; "
            f"basis: `{measure.basis}`; unit: `{measure.unit}`."
        )
        if measure.derivation is not None:
            detail += f" Derivation: {measure.derivation}."
        if measure.routing is not None:
            detail += f" Routing: {measure.routing}."
        lines.append(detail)
    lines.append("<!-- END GENERATED MEASURE REGISTRY -->")
    return "\n".join(lines)


def render_claims_placeholders(text: str, registry: MeasureRegistry | None = None) -> str:
    """Expand authored claims placeholders from the executable contract."""
    rendered = text.replace(CLAIMS_INSTRUCTION_PLACEHOLDER, render_claims_instruction(registry))
    if MEASURE_REGISTRY_PLACEHOLDER not in rendered:
        return rendered
    if registry is None:
        raise ValueError("measure registry placeholder requires a configured registry")
    return rendered.replace(MEASURE_REGISTRY_PLACEHOLDER, render_measure_registry(registry))


def missing_required_phrase_patterns(answer: str, patterns: list[str]) -> list[str]:
    """Return the required regex patterns that do not match an answer."""
    normalized = normalize_claims_text(answer)
    return [pattern for pattern in patterns if re.search(pattern, normalized) is None]
