from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import structlog
import structlog.testing
from anthropic import AsyncAnthropic
from anthropic.types.beta import FileMetadata
from daimon.adapters.slack.output_delivery import deliver_session_outputs
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

_TURN_STARTED = datetime(2026, 8, 23, 12, 0, tzinfo=UTC)


class _FilePages:
    """Small paginator fake that records which API pages were consumed."""

    def __init__(self, pages: list[list[FileMetadata]]) -> None:
        self.pages = pages
        self.pages_visited = 0

    def __aiter__(self) -> Any:
        return self._items()

    async def _items(self) -> Any:
        for page in self.pages:
            self.pages_visited += 1
            for file in page:
                yield file


def _file(file_id: str, *, size_bytes: int = 10) -> FileMetadata:
    return FileMetadata(
        id=file_id,
        created_at=_TURN_STARTED + timedelta(seconds=1),
        filename=f"{file_id}.txt",
        mime_type="text/plain",
        size_bytes=size_bytes,
        type="file",
    )


def _clients(
    files: list[FileMetadata],
    *,
    upload_side_effect: Exception | None = None,
) -> tuple[AsyncAnthropic, AsyncWebClient, Any, Any]:
    anthropic_mock: Any = MagicMock(spec=AsyncAnthropic)
    anthropic_mock.beta.files.list = AsyncMock(return_value=_FilePages([files]))

    async def _download(file_id: str, **_: Any) -> Any:
        return MagicMock(read=AsyncMock(return_value=f"content:{file_id}".encode()))

    anthropic_mock.beta.files.download = AsyncMock(side_effect=_download)
    web_mock: Any = MagicMock(spec=AsyncWebClient)
    web_mock.files_upload_v2 = AsyncMock(side_effect=upload_side_effect)
    web_mock.chat_postMessage = AsyncMock()
    return (
        cast(AsyncAnthropic, anthropic_mock),
        cast(AsyncWebClient, web_mock),
        anthropic_mock,
        web_mock,
    )


async def _deliver(
    anthropic: AsyncAnthropic,
    web_client: AsyncWebClient,
    *,
    session_id: str,
    turn_started_at: datetime = _TURN_STARTED,
) -> None:
    await deliver_session_outputs(
        anthropic,
        web_client,
        session_id=session_id,
        channel_id="C123",
        thread_ts="171234.5678",
        turn_started_at=turn_started_at,
    )


async def test_posts_each_file_once_into_the_thread() -> None:
    anthropic, web_client, anthropic_mock, web_mock = _clients([_file("F1"), _file("F2")])

    await _deliver(anthropic, web_client, session_id="session-once")
    await _deliver(anthropic, web_client, session_id="session-once")

    assert anthropic_mock.beta.files.download.await_count == 2
    assert web_mock.files_upload_v2.await_count == 2
    assert web_mock.files_upload_v2.await_args_list[0].kwargs == {
        "channel": "C123",
        "thread_ts": "171234.5678",
        "filename": "F1.txt",
        "title": "F1.txt",
        "content": b"content:F1",
        "content_type": "text/plain",
    }
    assert web_mock.files_upload_v2.await_args_list[1].kwargs["filename"] == "F2.txt"


async def test_empty_output_listing_is_a_no_op() -> None:
    anthropic, web_client, anthropic_mock, web_mock = _clients([])

    await _deliver(anthropic, web_client, session_id="session-empty")

    anthropic_mock.beta.files.download.assert_not_awaited()
    web_mock.files_upload_v2.assert_not_awaited()


async def test_retries_empty_output_listing_once_for_indexing_lag() -> None:
    anthropic, web_client, anthropic_mock, web_mock = _clients([])
    anthropic_mock.beta.files.list.side_effect = [
        _FilePages([[]]),
        _FilePages([[_file("F-LATE")]]),
    ]

    with patch(
        "daimon.adapters.slack.output_delivery.asyncio.sleep", new_callable=AsyncMock
    ) as sleep:
        await _deliver(anthropic, web_client, session_id="session-index-lag")

    assert anthropic_mock.beta.files.list.await_count == 2
    sleep.assert_awaited_once_with(1.0)
    web_mock.files_upload_v2.assert_awaited_once()


async def test_missing_files_write_scope_posts_actionable_thread_message() -> None:
    missing_scope = SlackApiError(
        message="missing_scope",
        response={"ok": False, "error": "missing_scope", "needed": "files:write"},
    )
    anthropic, web_client, _, web_mock = _clients(
        [_file("F-SCOPE-1"), _file("F-SCOPE-2")], upload_side_effect=missing_scope
    )
    capture = structlog.testing.LogCapture()
    structlog.configure(processors=[capture])
    try:
        await _deliver(anthropic, web_client, session_id="session-missing-scope")
    finally:
        structlog.reset_defaults()

    scope_events = [
        event
        for event in capture.entries
        if event["event"] == "slack.output_delivery.missing_scope"
    ]
    assert len(scope_events) == 1
    assert web_mock.files_upload_v2.await_count == 1, (
        "missing files:write must stop later upload attempts after one warning"
    )
    web_mock.chat_postMessage.assert_awaited_once_with(
        channel="C123",
        thread_ts="171234.5678",
        text=(
            "I couldn't attach the generated file because this app is missing the "
            "`files:write` Slack scope. A workspace admin must add that scope and "
            "reinstall daimon from the install link before file delivery can work."
        ),
    )


async def test_oversize_and_excess_files_are_bounded_and_logged() -> None:
    files = [_file("F-BIG", size_bytes=20 * 1024 * 1024 + 1)] + [_file(f"F{i}") for i in range(6)]
    anthropic, web_client, anthropic_mock, web_mock = _clients(files)
    capture = structlog.testing.LogCapture()
    structlog.configure(processors=[capture])
    try:
        await _deliver(anthropic, web_client, session_id="session-bounded")
    finally:
        structlog.reset_defaults()

    assert anthropic_mock.beta.files.download.await_count == 5
    assert web_mock.files_upload_v2.await_count == 5
    events = [event["event"] for event in capture.entries]
    assert "slack.output_delivery.skipped_oversize" in events
    assert "slack.output_delivery.cap_reached" in events


async def test_one_upload_failure_is_logged_and_later_files_continue() -> None:
    anthropic, web_client, _, web_mock = _clients([_file("F1"), _file("F2")])
    web_mock.files_upload_v2.side_effect = [RuntimeError("upload failed"), None]
    capture = structlog.testing.LogCapture()
    structlog.configure(processors=[capture])
    try:
        await _deliver(anthropic, web_client, session_id="session-continue")
    finally:
        structlog.reset_defaults()

    assert web_mock.files_upload_v2.await_count == 2
    assert any(event["event"] == "slack.output_delivery.upload_failed" for event in capture.entries)


async def test_auto_paginates_until_new_turn_files_on_second_page() -> None:
    old_files = []
    for index in range(20):
        file = _file(f"F-OLD-{index}")
        file.created_at = _TURN_STARTED - timedelta(minutes=5)
        old_files.append(file)
    new_files = [_file("F-NEW-1"), _file("F-NEW-2")]
    anthropic, web_client, anthropic_mock, web_mock = _clients([])
    pages = _FilePages([old_files, new_files])
    anthropic_mock.beta.files.list.return_value = pages

    await _deliver(anthropic, web_client, session_id="session-two-pages")

    assert pages.pages_visited == 2
    assert [call.args[0] for call in anthropic_mock.beta.files.download.await_args_list] == [
        "F-NEW-1",
        "F-NEW-2",
    ]
    assert web_mock.files_upload_v2.await_count == 2


async def test_old_file_is_not_mistaken_for_already_delivered() -> None:
    file = _file("F-OLD-THEN-ELIGIBLE")
    file.created_at = _TURN_STARTED - timedelta(minutes=1)
    anthropic, web_client, anthropic_mock, web_mock = _clients([file])

    await _deliver(anthropic, web_client, session_id="session-old-not-delivered")
    await _deliver(
        anthropic,
        web_client,
        session_id="session-old-not-delivered",
        turn_started_at=_TURN_STARTED - timedelta(minutes=2),
    )

    anthropic_mock.beta.files.download.assert_awaited_once_with(
        "F-OLD-THEN-ELIGIBLE", betas=["managed-agents-2026-04-01"]
    )
    web_mock.files_upload_v2.assert_awaited_once()


async def test_sanitizes_filename_and_title_before_upload() -> None:
    file = _file("F-UNSAFE")
    file.filename = "../../reports/Q3 forecast.csv"
    anthropic, web_client, _, web_mock = _clients([file])

    await _deliver(anthropic, web_client, session_id="session-safe-filename")

    upload = web_mock.files_upload_v2.await_args.kwargs
    assert upload["filename"] == "reports_Q3_forecast.csv"
    assert upload["title"] == "reports_Q3_forecast.csv"
