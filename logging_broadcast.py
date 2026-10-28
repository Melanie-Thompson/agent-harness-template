from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from config import Config

_broadcast: LogBroadcast | None = None


class LogBroadcast:
    def __init__(self, buffer_size: int) -> None:
        self._buffer: deque[dict[str, Any]] = deque(maxlen=buffer_size)
        self._subscribers: set[asyncio.Queue[dict[str, Any]]] = set()

    def publish(self, event: dict[str, Any]) -> None:
        self._buffer.append(event)
        dead: list[asyncio.Queue[dict[str, Any]]] = []
        for queue in self._subscribers:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(queue)
        for q in dead:
            self._subscribers.discard(q)

    def subscribe(self, catchup: int = 50) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=500)
        recent = list(self._buffer)[-catchup:]
        for entry in recent:
            try:
                queue.put_nowait(entry)
            except asyncio.QueueFull:
                break
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self._subscribers.discard(queue)


def init_log_broadcast() -> None:
    global _broadcast
    _broadcast = LogBroadcast(Config.LOG_BROADCAST_BUFFER_SIZE)


def get_log_broadcast() -> LogBroadcast | None:
    return _broadcast


def close_log_broadcast() -> None:
    global _broadcast
    _broadcast = None
