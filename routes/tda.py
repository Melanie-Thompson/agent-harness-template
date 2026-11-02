import asyncio
import json

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from tda.stream import get_tda_stream

router = APIRouter(prefix="/tda", tags=["tda"])


async def _tda_sse(catchup: bool = True):
    stream = get_tda_stream()
    if stream is None:
        yield "event: error\ndata: {\"error\": \"TDA not enabled\"}\n\n"
        return

    q = stream.subscribe(catchup=catchup)
    try:
        while True:
            snapshot = await q.get()
            data = snapshot.model_dump_json()
            yield f"event: topology\ndata: {data}\n\n"
    except asyncio.CancelledError:
        pass
    finally:
        stream.unsubscribe(q)


@router.get("/stream", summary="Stream TDA topology updates via SSE")
async def tda_stream(catchup: bool = Query(True)):
    return StreamingResponse(
        _tda_sse(catchup=catchup),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/snapshot", summary="Get current TDA topology snapshot")
async def tda_snapshot():
    stream = get_tda_stream()
    if stream is None:
        return {"error": "TDA not enabled"}
    snapshot = stream.get_current_snapshot()
    if snapshot is None:
        return {"status": "waiting", "message": "Not enough log events yet (need at least 3)"}
    return snapshot.model_dump()
