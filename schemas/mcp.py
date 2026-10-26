from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class McpTransport(str, Enum):
    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable_http"


class McpServerConfig(BaseModel):
    name: str
    transport: McpTransport
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)


class McpToolPolicy(BaseModel):
    gated_tools: set[str] = Field(default_factory=set)
    tool_permissions: dict[str, list[str]] = Field(default_factory=dict)
