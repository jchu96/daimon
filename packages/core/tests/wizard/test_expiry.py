"""Unit tests for CDN signature parsing and form-expiry derivation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from daimon.core.wizard.expiry import (
    DEFAULT_TTL,
    MIN_TTL,
    SAFETY_MARGIN,
    UNKNOWN_SIGNATURE_TTL,
    derive_expires_at,
    parse_attachment_expiry,
)

_NOW = datetime(2026, 7, 30, 12, 0, 0, tzinfo=UTC)


def _url_with_expiry(expiry: datetime) -> str:
    hex_epoch = format(int(expiry.timestamp()), "x")
    return f"https://cdn.discordapp.com/attachments/1/2/a.png?ex={hex_epoch}&is=1&hm=deadbeef"


def test_parse_attachment_expiry_reads_real_shaped_discord_cdn_url() -> None:
    expected = datetime(2026, 8, 1, 0, 0, 0, tzinfo=UTC)
    url = f"https://cdn.discordapp.com/attachments/123/456/img.png?ex={format(int(expected.timestamp()), 'x')}&is=6890a000&hm=abc123"

    parsed = parse_attachment_expiry(url)

    assert parsed == expected, "parse_attachment_expiry must decode the hex ex parameter as UTC"


def test_parse_attachment_expiry_returns_none_without_ex_param() -> None:
    assert (
        parse_attachment_expiry("https://cdn.discordapp.com/attachments/1/2/a.png?is=1&hm=x")
        is None
    )


def test_parse_attachment_expiry_returns_none_for_non_hex_ex() -> None:
    assert (
        parse_attachment_expiry("https://cdn.discordapp.com/attachments/1/2/a.png?ex=notahex")
        is None
    )


def test_derive_expires_at_with_no_urls_returns_default_ttl() -> None:
    result = derive_expires_at(now=_NOW, attachment_urls=[])

    assert result == _NOW + DEFAULT_TTL


def test_derive_expires_at_with_far_future_expiry_returns_default_ttl() -> None:
    far_future = _NOW + timedelta(days=30)
    result = derive_expires_at(now=_NOW, attachment_urls=[_url_with_expiry(far_future)])

    assert result == _NOW + DEFAULT_TTL, (
        "the default TTL is the binding cap when signatures outlive it"
    )


def test_derive_expires_at_with_near_expiry_returns_expiry_minus_margin() -> None:
    near = _NOW + timedelta(hours=2)
    result = derive_expires_at(now=_NOW, attachment_urls=[_url_with_expiry(near)])

    assert result == near - SAFETY_MARGIN


def test_derive_expires_at_with_mixture_returns_earliest_minus_margin() -> None:
    near = _NOW + timedelta(hours=2)
    far = _NOW + timedelta(hours=8)
    result = derive_expires_at(
        now=_NOW, attachment_urls=[_url_with_expiry(far), _url_with_expiry(near)]
    )

    assert result == near - SAFETY_MARGIN, "the earliest attachment expiry governs the form's TTL"


def test_derive_expires_at_with_unparseable_url_caps_at_unknown_signature_ttl() -> None:
    unparseable = "https://cdn.discordapp.com/attachments/1/2/a.png?is=1&hm=x"
    result = derive_expires_at(now=_NOW, attachment_urls=[unparseable])

    assert result == _NOW + UNKNOWN_SIGNATURE_TTL


def test_derive_expires_at_with_past_expiry_floors_at_min_ttl() -> None:
    past = _NOW - timedelta(hours=1)
    result = derive_expires_at(now=_NOW, attachment_urls=[_url_with_expiry(past)])

    assert result == _NOW + MIN_TTL, (
        "an already-past signature expiry must never yield a past form expiry"
    )
