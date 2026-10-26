from __future__ import annotations

import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ai.agent import invoke_agent
from ai.pending import resolve_pending
from ai.tools.base import get_tools
from ai.tools.db import set_db_context
from auth import require
from db.connect import DbConnect
from db.mongo import get_mongo_db
from schemas.agent import AgentChatRequest, AgentEvent, ConfirmRequest
from schemas.session import MessageResponse, SessionData

router = APIRouter(prefix="/agent", tags=["agent"])


async def _sse_stream(events: AsyncGenerator[AgentEvent, None]) -> AsyncGenerator[str, None]:
    async for event in events:
        yield f"event: {event.type}\ndata: {json.dumps(event.data)}\n\n"


@router.post("/chat", summary="Stream an agent conversation via SSE")
async def agent_chat(
    body: AgentChatRequest,
    session: SessionData = Depends(require("agent:use")),
) -> StreamingResponse:
    db = DbConnect(session.user_id, get_mongo_db())
    set_db_context(db)

    tools = get_tools(permissions=session.permissions)
    events = invoke_agent(
        body.prompt,
        tools=tools,
        user_id=session.user_id,
        session_id=body.session_id,
        conversation_id=body.conversation_id,
        model=body.model,
    )

    return StreamingResponse(
        _sse_stream(events),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/confirm/{pending_id}", summary="Approve or deny a gated tool call")
async def confirm_tool(
    pending_id: str,
    body: ConfirmRequest,
    session: SessionData = Depends(require("agent:use")),
) -> MessageResponse:
    action = await resolve_pending(pending_id, body.approved)
    return MessageResponse(message=f"Tool call {action}")
