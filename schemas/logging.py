from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class LogEntry(BaseModel):
    timestamp: str
    level: str
    logger: str
    event: str
    function: str | None = None
    params: dict[str, Any] | None = None
    duration_ms: float | None = None
    error: str | None = None
    extra: dict[str, Any] | None = None
