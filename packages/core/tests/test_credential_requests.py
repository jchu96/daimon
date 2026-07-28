"""Unit tests for the credential-request wire contract (no I/O, no DB)."""

from __future__ import annotations

from daimon.core.credential_requests import (
    CUSTOM_ID_PATTERN,
    MAX_BUTTON_LABEL_CHARS,
    build_button_label,
    build_custom_id,
    mint_request_token,
)


def test_mint_request_token_returns_fresh_string_each_call() -> None:
    first = mint_request_token()
    second = mint_request_token()
    assert first != second, "two mints must never collide"


def test_build_custom_id_prefixes_the_token() -> None:
    token = mint_request_token()
    assert build_custom_id(token) == f"ztc:{token}"


def test_build_custom_id_stays_within_discord_custom_id_limit() -> None:
    custom_id = build_custom_id(mint_request_token())
    assert len(custom_id) <= 100, "custom_id must fit Discord's 100-char cap"


def test_custom_id_pattern_fullmatches_a_minted_custom_id_and_round_trips_the_token() -> None:
    token = mint_request_token()
    custom_id = build_custom_id(token)

    match = CUSTOM_ID_PATTERN.fullmatch(custom_id)

    assert match is not None, "the template must fullmatch a real minted custom_id"
    assert match["token"] == token, "the token group must round-trip the minted token"


def test_custom_id_pattern_rejects_bare_prefix() -> None:
    assert CUSTOM_ID_PATTERN.fullmatch("ztc:") is None, "a token-less custom_id must not match"


def test_custom_id_pattern_rejects_token_with_disallowed_characters() -> None:
    assert CUSTOM_ID_PATTERN.fullmatch("ztc:abc def") is None, (
        "a space is outside the allowed token charset"
    )


def test_custom_id_pattern_rejects_wrong_prefix() -> None:
    token = mint_request_token()
    assert CUSTOM_ID_PATTERN.fullmatch(f"xyz:{token}") is None, (
        "a custom_id minted by an unrelated feature must not match this template"
    )


def test_build_button_label_env() -> None:
    assert build_button_label("env", "OPENAI_API_KEY") == "Add env credential: OPENAI_API_KEY"


def test_build_button_label_mcp() -> None:
    assert build_button_label("mcp", "linear") == "Add MCP credential: linear"


def test_build_button_label_truncates_long_target_within_discord_label_limit() -> None:
    label = build_button_label("mcp", "x" * 200)

    assert len(label) <= MAX_BUTTON_LABEL_CHARS, "label must fit Discord's 80-char button limit"
    assert label.startswith("Add MCP credential: "), (
        "truncation must not lose the human-readable prefix"
    )
