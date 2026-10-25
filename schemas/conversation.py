from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class TokenUsage(BaseModel):
    input: int = 0
    output: int = 0


class ConversationMessage(BaseModel):
    conversation_id: str
    message_id: str
    session_id: str
    user_id: str
    role: Role
    timestamp: datetime
    content: str
    model: str | None = None
    tokens: TokenUsage | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationSummary(BaseModel):
    conversation_id: str
    user_id: str
    created_at: datetime
    message_count: int
    last_activity: datetime
