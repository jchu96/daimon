"""Machine-readable final-answer claims contract tests."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

import pytest
from daimon.core.claims import (
    ClaimsBlockError,
    extract_claims_block,
    parse_claims_block,
    serialize_claims,
    strip_claims_blocks,
)
from daimon.core.claims_contract import (
    CLAIMS_INSTRUCTION_PLACEHOLDER,
    Claim,
    MeasureDefinition,
    MeasureRegistry,
    normalize_claims_text,
    normalize_numeric_grading_text,
    render_claims_instruction,
    render_claims_placeholders,
    validate_measure_registry,
)
from pydantic import ValidationError


def _claim_fence(rows: str) -> str:
    return f"Answer prose.\n\n```claims\n{rows}\n```"


def test_parse_claims_block_strips_and_validates_trailing_carrier() -> None:
    extraction = parse_claims_block(
        _claim_fence(
            '[{"metric":"revenue","value":"$ 1,234.50","unit":"$",'
            '"basis":"invoices","as_of":"2026-08-25","source":"ledger.revenue"}]'
        )
    )

    assert extraction.prose == "Answer prose."
    assert extraction.errors == ()
    assert extraction.claims == [
        Claim(
            metric="revenue",
            value=Decimal("1234.50"),
            unit="usd",
            basis="invoices",
            as_of=date(2026, 8, 25),
            source="ledger.revenue",
        )
    ]


@pytest.mark.parametrize(
    "payload, message",
    [
        ("not-json", "claims: invalid JSON"),
        ('{"metric":"revenue"}', "claims: expected a JSON list"),
        ("[]", "claims block must contain at least one claim"),
    ],
)
def test_parse_claims_block_rejects_invalid_carrier_shapes(payload: str, message: str) -> None:
    with pytest.raises(ClaimsBlockError, match=message):
        parse_claims_block(_claim_fence(payload))


def test_claim_rows_isolate_invalid_siblings() -> None:
    extraction = parse_claims_block(
        _claim_fence(
            "["
            '{"metric":"bad metric","value":1,"unit":"usd","basis":"x",'
            '"as_of":null,"source":null},'
            '{"metric":"revenue","value":2,"unit":"usd","basis":"invoices",'
            '"as_of":null,"source":null}'
            "]"
        )
    )

    assert [claim.metric for claim in extraction.claims] == ["revenue"]
    assert extraction.errors == ("claims[0].metric: metric must be a lowercase registry id",)
    with pytest.raises(ClaimsBlockError, match=r"claims\[0\]\.metric"):
        extract_claims_block(
            _claim_fence(
                "["
                '{"metric":"bad metric","value":1,"unit":"usd","basis":"x",'
                '"as_of":null,"source":null},'
                '{"metric":"revenue","value":2,"unit":"usd","basis":"invoices",'
                '"as_of":null,"source":null}'
                "]"
            )
        )


def test_registry_membership_is_optional_and_isolated_per_row() -> None:
    registry = MeasureRegistry(
        (
            MeasureDefinition(
                id="revenue",
                source="semantic.revenue.amount",
                basis="invoices",
                unit="usd",
            ),
        )
    )
    answer = _claim_fence(
        "["
        '{"metric":"unknown","value":1,"unit":"usd","basis":"x",'
        '"as_of":null,"source":null},'
        '{"metric":"revenue","value":2,"unit":"usd","basis":"invoices",'
        '"as_of":null,"source":null}'
        "]"
    )

    open_extraction = parse_claims_block(answer)
    closed_extraction = parse_claims_block(answer, registry=registry)

    assert [claim.metric for claim in open_extraction.claims] == ["unknown", "revenue"]
    assert [claim.metric for claim in closed_extraction.claims] == ["revenue"]
    assert "unknown registered measure id 'unknown'" in closed_extraction.errors[0]


def test_non_trailing_fence_is_not_authoritative_but_all_fences_strip() -> None:
    answer = _claim_fence('[{"metric":"example"}]') + "\n\nMore prose."

    extraction = parse_claims_block(answer)

    assert extraction.prose == answer
    assert extraction.claims == []
    assert strip_claims_blocks(answer) == "Answer prose.\n\n\n\nMore prose."


def test_unicode_claim_normalization_covers_dashes_and_thin_spaces() -> None:
    assert normalize_claims_text("2026\u201108\u201125\u2009USD") == "2026-08-25 USD"
    claim = Claim.model_validate(
        {
            "metric": "REVENUE",
            "value": "$\u202f1,234",
            "unit": "$",
            "basis": "service\u2013month",
            "as_of": "2026\u201108\u201125",
            "source": None,
        }
    )
    assert claim.metric == "revenue"
    assert claim.value == Decimal("1234")
    assert claim.basis == "service-month"
    assert claim.as_of == date(2026, 8, 25)


def test_numeric_normalization_hides_range_endpoints_without_explicit_range_regex() -> None:
    answer = "Expected range: 10\u201312 million; point estimate: 11."

    scalar_view = normalize_numeric_grading_text(answer, answer_regex=None)
    range_view = normalize_numeric_grading_text(answer, answer_regex=r"10-12")

    assert "10" not in scalar_view and "12" not in scalar_view
    assert "point estimate: 11" in scalar_view
    assert "10-12" in range_view


def test_claim_schema_is_strict_but_ignores_presentation_keys() -> None:
    claim = Claim.model_validate(
        {
            "metric": "revenue",
            "value": "1,234",
            "unit": "usd",
            "basis": "invoices",
            "as_of": None,
            "source": None,
            "display": "$1.23k",
        }
    )
    assert claim.value == Decimal("1234")
    with pytest.raises(ValidationError):
        Claim.model_validate(
            {
                "metric": "revenue",
                "value": True,
                "unit": "usd",
                "basis": "invoices",
                "as_of": None,
                "source": None,
            }
        )


def test_contract_generates_instruction_and_expands_placeholder() -> None:
    instruction = render_claims_instruction()

    assert "metric, value, unit, basis, as_of, source" in instruction
    assert "usd" in instruction and "pct" in instruction
    assert render_claims_placeholders(CLAIMS_INSTRUCTION_PLACEHOLDER) == instruction


def test_measure_registry_validation_rejects_duplicate_or_malformed_definitions() -> None:
    valid = MeasureDefinition(
        id="revenue",
        source="semantic.revenue.amount",
        basis="invoices",
        unit="usd",
    )
    validate_measure_registry(MeasureRegistry((valid,)))
    with pytest.raises(ValueError, match="unique"):
        validate_measure_registry(MeasureRegistry((valid, valid)))
    with pytest.raises(ValueError, match="relation.column"):
        validate_measure_registry(
            MeasureRegistry(
                (
                    MeasureDefinition(
                        id="revenue",
                        source="revenue.amount",
                        basis="invoices",
                        unit="usd",
                    ),
                )
            )
        )


def test_serialize_claims_preserves_decimal_precision() -> None:
    claim = Claim(
        metric="revenue",
        value=Decimal("1.2300"),
        unit="usd",
        basis="invoices",
        as_of=None,
        source=None,
    )
    assert serialize_claims([claim])[0]["value"] == "1.2300"
