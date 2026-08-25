"""Strict machine-readable claims carried in an agent's final answer."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import cast

from daimon.core.claims_contract import CLAIMS_FENCE_PATTERN, Claim, MeasureRegistry
from pydantic import ValidationError

_CLAIMS_FENCE = re.compile(CLAIMS_FENCE_PATTERN, re.DOTALL)


class ClaimsBlockError(ValueError):
    """The answer declared a claims carrier that failed strict validation."""


@dataclass(frozen=True)
class ClaimsExtraction:
    """Parsed prose, accepted claims, and isolated per-row errors."""

    prose: str
    claims: list[Claim]
    errors: tuple[str, ...] = ()


def _format_claims_error(error: ValidationError) -> str:
    detail = error.errors(include_url=False)[0]
    if detail["type"] == "json_invalid":
        return "claims: invalid JSON"
    path = "claims"
    for part in detail["loc"]:
        path += f"[{part}]" if isinstance(part, int) else f".{part}"
    message = str(detail["msg"]).removeprefix("Value error, ")
    return f"{path}: {message}"


def parse_claims_block(text: str, *, registry: MeasureRegistry | None = None) -> ClaimsExtraction:
    """Parse the trailing carrier and isolate invalid rows.

    Only the final trailing fence is the authoritative carrier. A declared
    trailing carrier with malformed JSON, a non-list payload, or an empty list
    raises :class:`ClaimsBlockError`. Schema-invalid rows and, when configured,
    unknown registry IDs are omitted and reported individually while valid
    siblings remain available to graders. Core human-response extractors remove
    every carrier before handing text to renderers.
    """
    matches = list(_CLAIMS_FENCE.finditer(text))
    if not matches:
        return ClaimsExtraction(text, [])
    match = matches[-1]
    if text[match.end() :].strip():
        return ClaimsExtraction(text, [])
    try:
        payload = cast(object, json.loads(match.group("body")))
    except json.JSONDecodeError as error:
        raise ClaimsBlockError("claims: invalid JSON") from error
    if not isinstance(payload, list):
        raise ClaimsBlockError("claims: expected a JSON list")
    if not payload:
        raise ClaimsBlockError("claims block must contain at least one claim")

    accepted: list[Claim] = []
    errors: list[str] = []
    registry_ids = registry.ids if registry is not None else None
    for index, row in enumerate(cast(list[object], payload)):
        try:
            claim = Claim.model_validate(row)
        except ValidationError as error:
            detail = _format_claims_error(error)
            errors.append(detail.replace("claims", f"claims[{index}]", 1))
            continue
        if registry_ids is not None and claim.metric not in registry_ids:
            errors.append(f"claims[{index}].metric: unknown registered measure id {claim.metric!r}")
            continue
        accepted.append(claim)
    return ClaimsExtraction(text[: match.start()].rstrip(), accepted, tuple(errors))


def extract_claims_block(text: str) -> tuple[str, list[Claim]]:
    """Backward-compatible strict extraction without registry membership checks."""
    extraction = parse_claims_block(text)
    if extraction.errors:
        raise ClaimsBlockError(extraction.errors[0])
    return extraction.prose, extraction.claims


def strip_claims_blocks(text: str) -> str:
    """Remove every claims fence from a render surface without validating it."""
    return _CLAIMS_FENCE.sub("", text).rstrip()


def claims_basis(claims: list[Claim]) -> str | None:
    """Return ordered, de-duplicated bases for Slack provenance."""
    bases = list(dict.fromkeys(claim.basis for claim in claims))
    return " · ".join(bases) if bases else None


def serialize_claims(claims: list[Claim]) -> list[dict[str, str | None]]:
    """Return JSON-safe claim dictionaries without losing decimal precision."""
    return [
        {
            "metric": claim.metric,
            "value": str(claim.value),
            "unit": claim.unit,
            "basis": claim.basis,
            "as_of": claim.as_of.isoformat() if claim.as_of is not None else None,
            "source": claim.source,
        }
        for claim in claims
    ]
