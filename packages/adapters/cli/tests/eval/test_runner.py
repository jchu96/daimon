"""Golden validation and grading policy tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from daimon.adapters.cli.eval.models import Golden
from daimon.adapters.cli.eval.runner import grade_answer, load_goldens, validate_golden


def _typed_golden(**properties: list[str]) -> Golden:
    return {
        "id": "typed",
        "question": "Capacity?",
        "expected_properties": properties,
        "kind": "value",
        "expected_claim": {
            "metric": "installed_mw",
            "value": 10,
            "tolerance": {"abs": 0.01},
            "unit": "mw",
        },
    }


def test_required_phrases_are_advisory_and_bans_gate() -> None:
    clean = grade_answer(
        golden=_typed_golden(
            required_phrases=["missing advisory phrase"],
            banned_phrases=[r"(?i)secret operator label"],
        ),
        answer_source=(
            "Supported total is 10 MW.\n\n"
            "```claims\n"
            '[{"metric":"installed_mw","value":10,"unit":"mw","basis":"nameplate",'
            '"as_of":null,"source":null}]\n```'
        ),
        recomputed=None,
        suspect_labels=set(),
    )
    leaked = grade_answer(
        golden=_typed_golden(banned_phrases=[r"(?i)secret operator label"]),
        answer_source=(
            "Secret operator label.\n\n"
            "```claims\n"
            '[{"metric":"installed_mw","value":10,"unit":"mw","basis":"nameplate",'
            '"as_of":null,"source":null}]\n```'
        ),
        recomputed=None,
        suspect_labels=set(),
    )

    advisory = next(grade for grade in clean["grades"] if grade["name"] == "required_properties")
    assert advisory["passed"] is False and advisory["gating"] is False
    assert clean["passed"] is True
    assert leaked["passed"] is False


def test_grade_answer_preserves_raw_carrier_and_isolated_claim_errors() -> None:
    raw = (
        "Capacity is 10 MW.\n\n```claims\n["
        '{"metric":"bad metric","value":9,"unit":"mw","basis":"x",'
        '"as_of":null,"source":null},'
        '{"metric":"installed_mw","value":10,"unit":"mw","basis":"nameplate",'
        '"as_of":null,"source":null}]\n```'
    )

    case = grade_answer(
        golden=_typed_golden(),
        answer_source=raw,
        recomputed=None,
        suspect_labels=set(),
    )

    assert case["passed"] is True
    assert case["answer_raw"] == raw
    assert "```claims" not in case["answer"]
    assert [claim["metric"] for claim in case["claims"]] == ["installed_mw"]
    assert case["claims_error"] == ["claims[0].metric: metric must be a lowercase registry id"]


@pytest.mark.parametrize(
    "mutation, message",
    [
        ({"kind": "estimate"}, "kind must be value, rejection, or trend"),
        (
            {"expected_claim": {"metric": "x", "value": 1, "tolerance": {}, "unit": "mw"}},
            "tolerance must contain exactly one",
        ),
        (
            {
                "expected_claim": {
                    "metric": "x",
                    "value": 1,
                    "tolerance": {"abs": 0},
                    "unit": "meters",
                }
            },
            "expected_claim.unit must be one of",
        ),
    ],
)
def test_validate_golden_rejects_contract_mutations(
    mutation: dict[str, object], message: str
) -> None:
    payload: dict[str, object] = dict(_typed_golden())
    payload.update(mutation)
    with pytest.raises(ValueError, match=message):
        validate_golden(payload, location="fixture")


def test_load_goldens_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "goldens.jsonl"
    line = (
        '{"id":"same","question":"Q?","expected_properties":{},"kind":"value",'
        '"expected_claim":{"metric":"x","value":1,"tolerance":{"abs":0},"unit":"count"}}'
    )
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unique"):
        load_goldens(path)
