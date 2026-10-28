from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from logging_broadcast import get_log_broadcast

router = APIRouter(prefix="/logs", tags=["logs"])

_LEVEL_PRIORITY = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}


async def _log_sse(
    queue: asyncio.Queue,
    min_priority: int,
) -> AsyncGenerator[str, None]:
    try:
        while True:
            entry = await queue.get()
            entry_priority = _LEVEL_PRIORITY.get(entry.get("level", "INFO"), 0)
            if entry_priority >= min_priority:
                yield f"event: log\ndata: {json.dumps(entry, default=str)}\n\n"
    except asyncio.CancelledError:
        return


@router.get("/stream", summary="Stream structured logs via SSE")
async def log_stream(
    level: str | None = Query(None, description="Minimum log level: DEBUG, INFO, WARNING, ERROR"),
) -> StreamingResponse:
    broadcast = get_log_broadcast()
    if broadcast is None:
        return StreamingResponse(
            iter(["event: error\ndata: {\"error\": \"Log broadcast not initialized\"}\n\n"]),
            media_type="text/event-stream",
        )

    min_priority = _LEVEL_PRIORITY.get(level.upper(), 0) if level else 0
    queue = broadcast.subscribe()

    async def stream() -> AsyncGenerator[str, None]:
        try:
            async for chunk in _log_sse(queue, min_priority):
                yield chunk
        finally:
            broadcast.unsubscribe(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
