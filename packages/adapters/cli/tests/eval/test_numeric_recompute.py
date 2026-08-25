"""Typed-claims primary numeric grader tests."""

from __future__ import annotations

from daimon.adapters.cli.eval.graders.numeric_recompute import grade
from daimon.core.claims import parse_claims_block


def _claims(answer: str) -> tuple[str, list[object]]:
    extraction = parse_claims_block(answer)
    return extraction.prose, list(extraction.claims)


def test_typed_claim_beats_conflicting_prose() -> None:
    answer, claims = _claims(
        """The supported total is 99 MW.

```claims
[{"metric":"installed_mw","value":10,"unit":"mw","basis":"nameplate","as_of":null,"source":null}]
```"""
    )

    result = grade(
        answer,
        expected_sql={
            "query": "SELECT 10",
            "answer_regex": r"(?P<value>10)\s*MW",
            "tolerance": 0,
        },
        recomputed=10,
        claims=claims,  # type: ignore[arg-type]
        expected_claim={
            "metric": "installed_mw",
            "value": 10,
            "tolerance": {"abs": 0.01},
            "unit": "mw",
        },
    )

    assert result["passed"] is True


def test_matching_prose_cannot_rescue_wrong_typed_claim() -> None:
    answer, claims = _claims(
        """The supported total is 10 MW.

```claims
[{"metric":"installed_mw","value":9,"unit":"mw","basis":"nameplate","as_of":null,"source":null}]
```"""
    )

    result = grade(
        answer,
        expected_sql={
            "query": "SELECT 10",
            "answer_regex": r"(?P<value>10)\s*MW",
            "tolerance": 0,
        },
        recomputed=10,
        claims=claims,  # type: ignore[arg-type]
        expected_claim={
            "metric": "installed_mw",
            "value": 10,
            "tolerance": {"abs": 0.01},
            "unit": "mw",
        },
    )

    assert result["passed"] is False
    assert "claim=9" in result["detail"]


def test_typed_claim_honours_golden_display_precision() -> None:
    answer, claims = _claims(
        """Realization was 45.1%.

```claims
[{"metric":"realization_pct","value":45.1,"unit":"pct","basis":"service-month","as_of":null,"source":null}]
```"""
    )

    result = grade(
        answer,
        expected_sql=None,
        recomputed=None,
        claims=claims,  # type: ignore[arg-type]
        expected_claim={
            "metric": "realization_pct",
            "value": 45.052,
            "tolerance": {"abs": 0.001},
            "unit": "pct",
            "display_precision": 1,
        },
    )

    assert result["passed"] is True
    assert "display_precision=1" in result["detail"]


def test_rejection_requires_corrected_claim_or_seed_bound_refusal() -> None:
    expected = {
        "metric": "installed_mw",
        "value": 10,
        "tolerance": {"abs": 0.01},
        "unit": "mw",
    }

    refused = grade(
        "I cannot confirm the seeded 10 MW value.",
        expected_sql=None,
        recomputed=None,
        claims=[],
        expected_claim=expected,
        kind="rejection",
    )
    unrelated_hedge = grade(
        "The unrelated cycle-time field cannot be computed.",
        expected_sql=None,
        recomputed=None,
        claims=[],
        expected_claim=expected,
        kind="rejection",
    )

    assert refused["passed"] is True
    assert unrelated_hedge["passed"] is False


def test_rejection_passes_on_corrected_claim_but_not_seed_repetition() -> None:
    corrected_answer, corrected_claims = _claims(
        """The seeded 10 MW value is incorrect; the supported total is 12 MW.

```claims
[{"metric":"installed_mw","value":12,"unit":"mw","basis":"nameplate","as_of":null,"source":null}]
```"""
    )
    repeated_answer, repeated_claims = _claims(
        """The supported total is 10 MW.

```claims
[{"metric":"installed_mw","value":10,"unit":"mw","basis":"nameplate","as_of":null,"source":null}]
```"""
    )
    expected = {
        "metric": "installed_mw",
        "value": 10,
        "tolerance": {"rel": 0.01},
        "unit": "mw",
    }

    assert (
        grade(
            corrected_answer,
            expected_sql=None,
            recomputed=None,
            claims=corrected_claims,  # type: ignore[arg-type]
            expected_claim=expected,
            kind="rejection",
        )["passed"]
        is True
    )
    assert (
        grade(
            repeated_answer,
            expected_sql=None,
            recomputed=None,
            claims=repeated_claims,  # type: ignore[arg-type]
            expected_claim=expected,
            kind="rejection",
        )["passed"]
        is False
    )


def test_trend_uses_relative_tolerance() -> None:
    answer, claims = _claims(
        """Revenue increased 5.05%.

```claims
[{"metric":"revenue_growth","value":5.05,"unit":"pct","basis":"year over year","as_of":null,"source":null}]
```"""
    )

    result = grade(
        answer,
        expected_sql=None,
        recomputed=None,
        claims=claims,  # type: ignore[arg-type]
        expected_claim={
            "metric": "revenue_growth",
            "value": 5,
            "tolerance": {"rel": 0.02},
            "unit": "pct",
        },
        kind="trend",
    )

    assert result["passed"] is True


def test_legacy_prose_fallback_remains_available() -> None:
    result = grade(
        "The supported total is 10 MW.",
        expected_sql={
            "query": "SELECT 10",
            "answer_regex": r"(?P<value>10)\s*MW",
            "tolerance": 0.01,
        },
        recomputed=10,
    )

    assert result["passed"] is True
