"""CDN signature parsing and form-expiry derivation.

A wizard's images, when it has any, are attachments Discord serves from a
signed CDN URL — the `ex` (expiry), `is` (issue time) and `hm` (signature)
query parameters. Discord's documentation describes the parameters but
commits to no fixed duration for them, so a form's own expiry cannot be a
constant: the signed URLs a form's images are persisted as expire on their
own schedule, and a form whose buttons stay tappable after its pictures have
gone dark is exactly the state `derive_expires_at` exists to prevent. The
form's TTL must instead be derived from the actual signature expiry the CDN
returned when the images were posted.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Final
from urllib.parse import parse_qs, urlparse

DEFAULT_TTL: Final[timedelta] = timedelta(hours=12)
SAFETY_MARGIN: Final[timedelta] = timedelta(minutes=5)
UNKNOWN_SIGNATURE_TTL: Final[timedelta] = timedelta(hours=1)
MIN_TTL: Final[timedelta] = timedelta(minutes=1)


def parse_attachment_expiry(url: str) -> datetime | None:
    """Return the UTC expiry the CDN signed into `url`'s `ex` parameter.

    Returns `None` when the parameter is absent or not parseable as a
    hexadecimal Unix timestamp. Never raises.
    """
    query = parse_qs(urlparse(url).query)
    values = query.get("ex")
    if not values:
        return None
    try:
        epoch_seconds = int(values[0], 16)
    except ValueError:
        return None
    return datetime.fromtimestamp(epoch_seconds, tz=UTC)


def derive_expires_at(*, now: datetime, attachment_urls: Sequence[str]) -> datetime:
    """Return the UTC instant a wizard form should expire at.

    With no attachments, the form's ceiling is `now + DEFAULT_TTL`.
    Otherwise the result is the minimum of `now + DEFAULT_TTL` and the
    smallest parsed attachment expiry minus `SAFETY_MARGIN`. If any
    attachment's expiry could not be parsed, the result is additionally
    capped at `now + UNKNOWN_SIGNATURE_TTL` — an unreadable signature is
    treated as expiring soon, not as expiring never. The result is always
    floored at `now + MIN_TTL` so an already-past signature expiry never
    produces a form that expires in the past.
    """
    if not attachment_urls:
        return now + DEFAULT_TTL

    parsed_expiries = [parse_attachment_expiry(url) for url in attachment_urls]
    known_expiries = [expiry for expiry in parsed_expiries if expiry is not None]
    has_unknown_signature = len(known_expiries) < len(parsed_expiries)

    candidate = now + DEFAULT_TTL
    if known_expiries:
        candidate = min(candidate, min(known_expiries) - SAFETY_MARGIN)
    if has_unknown_signature:
        candidate = min(candidate, now + UNKNOWN_SIGNATURE_TTL)

    return max(candidate, now + MIN_TTL)
