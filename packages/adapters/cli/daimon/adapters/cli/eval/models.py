"""Typed contracts shared by the eval runner and grader modules."""

from __future__ import annotations

from typing import NotRequired, TypedDict


class ExpectedSQL(TypedDict):
    query: str
    answer_regex: str
    tolerance: float
    display_precision: NotRequired[int]
    claim_regex: NotRequired[str]


class Golden(TypedDict):
    id: str
    question: str
    expected_sql: NotRequired[ExpectedSQL]


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
    passed: bool
    grades: list[Grade]
    error_type: NotRequired[str]
