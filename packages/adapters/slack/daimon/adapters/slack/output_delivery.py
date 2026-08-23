"""Best-effort delivery of Managed Agents session outputs to Slack.

The Managed Agents Files API contract is the boundary here: listing with the
session ``scope_id`` is treated as the session's output collection, excluding
uploads, credential resources, and files from the working directory. Daimon
does not inspect the sandbox filesystem directly.

Successful-delivery IDs are process-local because this adapter has no durable
delivery-receipt repository. A process restart can therefore re-post a recent
file inside the clock-skew window; the trade-off avoids marking an unposted
file as delivered and silently losing it.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import datetime, timedelta

import structlog
from anthropic import AsyncAnthropic
from anthropic.types.beta import FileMetadata
from daimon.core.media.filenames import sanitize_title
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

log = structlog.get_logger(__name__)

_MAX_FILES_PER_TURN = 5
_MAX_BYTES_PER_FILE = 20 * 1024 * 1024
_CLOCK_SLACK = timedelta(seconds=30)
_INDEX_RETRY_DELAY_S = 1.0

# Session-scoped listings include prior-turn artifacts. Only successful Slack
# uploads count as delivered; age, size, cap, and transient failures do not.
_delivered_file_ids: dict[str, set[str]] = {}


async def _iter_session_files(
    anthropic: AsyncAnthropic, *, session_id: str
) -> AsyncIterator[FileMetadata]:
    """List session files, retrying one empty result for MA indexing lag."""
    for attempt in range(2):
        try:
            page = await anthropic.beta.files.list(
                scope_id=session_id,
                betas=["managed-agents-2026-04-01"],
            )
        except Exception as exc:  # noqa: BLE001 -- post-turn delivery must degrade cleanly
            log.warning(
                "slack.output_delivery.list_failed",
                session_id=session_id,
                error=str(exc)[:300],
            )
            return

        saw_file = False
        try:
            async for file in page:
                saw_file = True
                yield file
        except Exception as exc:  # noqa: BLE001 -- pagination must degrade cleanly
            log.warning(
                "slack.output_delivery.pagination_failed",
                session_id=session_id,
                error=str(exc)[:300],
            )
            return

        if saw_file or attempt == 1:
            return
        log.debug(
            "slack.output_delivery.empty_listing_retry",
            session_id=session_id,
            delay_seconds=_INDEX_RETRY_DELAY_S,
        )
        await asyncio.sleep(_INDEX_RETRY_DELAY_S)


async def deliver_session_outputs(
    anthropic: AsyncAnthropic,
    web_client: AsyncWebClient,
    *,
    session_id: str,
    channel_id: str,
    thread_ts: str,
    turn_started_at: datetime,
) -> None:
    """Upload output files created during this turn without failing the turn."""
    delivered = _delivered_file_ids.setdefault(session_id, set())
    cutoff = turn_started_at - _CLOCK_SLACK

    posted = 0
    try:
        async for file in _iter_session_files(anthropic, session_id=session_id):
            if file.id in delivered:
                log.debug(
                    "slack.output_delivery.already_delivered",
                    session_id=session_id,
                    file_id=file.id,
                )
                continue

            created_at = file.created_at
            if created_at.tzinfo is None and cutoff.tzinfo is not None:
                created_at = created_at.replace(tzinfo=cutoff.tzinfo)
            if created_at < cutoff:
                log.debug(
                    "slack.output_delivery.before_turn",
                    session_id=session_id,
                    file_id=file.id,
                )
                continue
            if file.size_bytes > _MAX_BYTES_PER_FILE:
                log.info(
                    "slack.output_delivery.skipped_oversize",
                    session_id=session_id,
                    file_id=file.id,
                    size_bytes=file.size_bytes,
                    max_bytes=_MAX_BYTES_PER_FILE,
                )
                continue
            if posted >= _MAX_FILES_PER_TURN:
                log.info(
                    "slack.output_delivery.cap_reached",
                    session_id=session_id,
                    max_files=_MAX_FILES_PER_TURN,
                    next_file_id=file.id,
                )
                break

            safe_filename = sanitize_title(file.filename)
            try:
                response = await anthropic.beta.files.download(
                    file.id,
                    betas=["managed-agents-2026-04-01"],
                )
                content = await response.read()
                await web_client.files_upload_v2(  # pyright: ignore[reportUnknownMemberType]
                    channel=channel_id,
                    thread_ts=thread_ts,
                    filename=safe_filename,
                    title=safe_filename,
                    content=content,
                )
            except SlackApiError as exc:
                if exc.response.get("error") == "missing_scope":  # pyright: ignore[reportUnknownMemberType]
                    log.warning(
                        "slack.output_delivery.missing_scope",
                        session_id=session_id,
                        required_scope="files:write",
                    )
                    await web_client.chat_postMessage(  # pyright: ignore[reportUnknownMemberType]
                        channel=channel_id,
                        thread_ts=thread_ts,
                        text=(
                            "I couldn't attach the generated file because this app is missing "
                            "the `files:write` Slack scope. A workspace admin must add that "
                            "scope and reinstall daimon from the install link before file "
                            "delivery can work."
                        ),
                    )
                    return
                log.warning(
                    "slack.output_delivery.upload_failed",
                    session_id=session_id,
                    file_id=file.id,
                    error=str(exc)[:300],
                )
                continue
            except Exception as exc:  # noqa: BLE001 -- one bad file must not block the turn
                log.warning(
                    "slack.output_delivery.upload_failed",
                    session_id=session_id,
                    file_id=file.id,
                    error=str(exc)[:300],
                )
                continue

            delivered.add(file.id)
            posted += 1
            log.info(
                "slack.output_delivery.posted",
                session_id=session_id,
                file_id=file.id,
                filename=safe_filename,
                size_bytes=file.size_bytes,
            )
    except Exception as exc:  # noqa: BLE001 -- pagination/metadata must degrade cleanly
        log.warning(
            "slack.output_delivery.pagination_failed",
            session_id=session_id,
            error=str(exc)[:300],
        )
