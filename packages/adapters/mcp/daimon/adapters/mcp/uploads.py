"""PUT route that receives agent-produced file bytes.

Mounted on the FastMCP-derived ASGI app via `add_route` in server.py — a
sibling of /healthz and /oauth/*, and like them it bypasses IdentityMiddleware
because the single-use upload token IS the auth.

This route exists so file bytes never transit the model's token stream: the
agent's sandbox curls the file here directly and the model only ever handles a
short handle id.

Catch at boundaries only: this module IS the catch boundary.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import structlog
from daimon.core.stores.file_uploads import (
    MAX_UPLOAD_BYTES,
    UploadTooLargeError,
    store_upload_content,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

log = structlog.get_logger(__name__)


def build_upload_route(
    session_factory: async_sessionmaker[AsyncSession],
) -> Callable[[Request], Awaitable[Response]]:
    """Build the `PUT /uploads/{token}` handler."""

    async def upload(request: Request) -> Response:
        token = request.path_params["token"]

        # Refuse on the declared length before reading the body, so an oversize
        # PUT costs one header rather than a full transfer into memory.
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
            return PlainTextResponse(
                f"upload exceeds the {MAX_UPLOAD_BYTES} byte limit", status_code=413
            )

        body = await request.body()
        async with session_factory() as session:
            try:
                row = await store_upload_content(
                    session, upload_token=token, data=body, now=datetime.now(UTC)
                )
            except UploadTooLargeError as err:
                return PlainTextResponse(str(err), status_code=413)
            if row is None:
                # Unknown, already burnt, or past TTL — deliberately one answer,
                # since the caller is unauthenticated apart from the token.
                log.info("upload.token_unusable")
                return PlainTextResponse("unknown or expired upload token", status_code=404)
            await session.commit()

        log.info("upload.stored", handle_id=row.id, bytes=len(body))
        return PlainTextResponse("", status_code=204)

    return upload
