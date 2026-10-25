from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class AgentChatRequest(BaseModel):
    prompt: str
    conversation_id: str | None = None
    session_id: str | None = None
    model: str | None = None


class ConfirmRequest(BaseModel):
    approved: bool


class AgentEvent(BaseModel):
    type: Literal[
        "thinking",
        "tool_call",
        "tool_result",
        "tool_pending",
        "tool_confirmed",
        "tool_denied",
        "response",
        "error",
        "blocked",
    ]
    data: dict[str, Any] = {}
