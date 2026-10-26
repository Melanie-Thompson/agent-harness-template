from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel


class AgentChatRequest(BaseModel):
    prompt: str
    conversation_id: str | None = None
    session_id: str | None = None
    model: str | None = None


class ConfirmRequest(BaseModel):
    approved: bool


class PendingActionMeta(BaseModel):
    pending_id: str
    tool_name: str
    tool_args: dict[str, Any]
    user_id: str
    created_at: datetime


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
        "eval_retry",
    ]
    data: dict[str, Any] = {}
