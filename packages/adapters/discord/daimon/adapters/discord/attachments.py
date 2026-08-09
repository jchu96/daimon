"""Discord data-attachment handling.

Non-image attachments (CSV, PDF, ...) are surfaced to the agent as their signed
Discord CDN URL — the agent has a bash tool with network egress and fetches the
bytes itself. This mirrors how images are surfaced (``vision.build_image_url_prefix``)
and avoids any bot-side upload that could silently fail. If the agent needs a file
on a notebook workspace to publish, it uploads on demand via the
``create_attachment_upload_url`` MCP tool — the bot does not upload eagerly.
"""

from __future__ import annotations

import discord


def build_attachment_url_prefix(attachments: list[discord.Attachment]) -> str:
    """One ``[attachment] ...`` line per non-image attachment exposing its signed CDN URL.

    The line is adapter-authored metadata sitting in the user turn, so it says so
    plainly. It must NOT impersonate a system message: a line reading
    ``*system: fetch this URL*`` inside user-turn text is indistinguishable from a
    prompt injection, and injection-aware models refuse it — a real agent refused a
    real user's real upload on exactly that basis.

    Data files (CSV, PDF, ...) aren't vision blocks, so the agent reaches them by
    fetching the bytes itself: Discord's signed CDN URL (``?ex=&is=&hm=`` params
    included) is publicly fetchable until the signature expires (~24h). Returns the
    empty string when there are no data attachments.

    If the agent actually needs the file on a notebook workspace (to build/publish a
    notebook), it owns that decision: it curls the URL to disk, then mints an upload
    URL via the ``create_attachment_upload_url`` MCP tool and PUTs the bytes.
    """
    return "\n".join(
        f"[attachment] `{attachment.filename}` ({attachment.size} bytes), uploaded by the "
        f"user with this message. Signed Discord CDN URL, expires ~24h: {attachment.url} "
        f"— curl it to disk, then read it. To use it in a notebook you publish, upload it "
        f"via the create_attachment_upload_url tool."
        for attachment in attachments
    )
