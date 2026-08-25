"""Typed contracts shared by the eval runner and grader modules."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict


class ExpectedSQL(TypedDict):
    query: str
    answer_regex: str
    tolerance: float
    display_precision: NotRequired[int]
    claim_regex: NotRequired[str]


class ClaimTolerance(TypedDict, total=False):
    abs: float
    rel: float


class ExpectedClaim(TypedDict):
    metric: str
    value: str | int | float
    tolerance: ClaimTolerance
    unit: str
    basis: NotRequired[str]
    display_precision: NotRequired[int]


type GoldenKind = Literal["value", "rejection", "trend"]


class ClaimPayload(TypedDict):
    metric: str
    value: str
    unit: str
    basis: str
    as_of: str | None
    source: str | None


class VizExpectation(TypedDict):
    tier: str
    form: str
    finding_headline: bool
    panel_annotations: list[str]


class ExpectedProperties(TypedDict):
    top_n_sites: NotRequired[bool]
    requires_gap: NotRequired[bool]
    required_phrases: NotRequired[list[str]]
    banned_phrases: NotRequired[list[str]]
    grain_expectation: NotRequired[str]
    answer_format_expectation: NotRequired[str]
    viz_expectation: NotRequired[VizExpectation]


class Golden(TypedDict):
    id: str
    question: str
    expected_properties: ExpectedProperties
    expected_sql: NotRequired[ExpectedSQL]
    expected_claim: NotRequired[ExpectedClaim]
    kind: NotRequired[GoldenKind]


class Grade(TypedDict):
    name: str
    passed: bool
    gating: bool
    detail: str


class CaseResult(TypedDict):
    id: str
    question: str
    answer: str
    answer_raw: str
    claims: list[ClaimPayload]
    passed: bool
    grades: list[Grade]
    error_type: NotRequired[str]
    claims_error: NotRequired[list[str]]
