from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from config import Config
from db.sessions import get_dragonfly
from errors import PendingActionNotFoundError
from schemas.agent import PendingActionMeta

_pending_futures: dict[str, asyncio.Future[bool]] = {}

_KEY_PREFIX = "pending:"


async def create_pending(
    pending_id: str,
    *,
    tool_name: str,
    tool_args: dict,
    user_id: str,
) -> asyncio.Future[bool]:
    meta = PendingActionMeta(
        pending_id=pending_id,
        tool_name=tool_name,
        tool_args=tool_args,
        user_id=user_id,
        created_at=datetime.now(timezone.utc),
    )
    r = get_dragonfly()
    await r.set(f"{_KEY_PREFIX}{pending_id}", meta.model_dump_json(), ex=Config.AGENT_PENDING_TTL)

    loop = asyncio.get_running_loop()
    future: asyncio.Future[bool] = loop.create_future()
    _pending_futures[pending_id] = future
    return future


async def resolve_pending(pending_id: str, approved: bool) -> str:
    r = get_dragonfly()
    key = f"{_KEY_PREFIX}{pending_id}"
    raw = await r.get(key)
    if raw is None:
        _pending_futures.pop(pending_id, None)
        raise PendingActionNotFoundError(pending_id)

    await r.delete(key)

    future = _pending_futures.pop(pending_id, None)
    if future is not None and not future.done():
        future.set_result(approved)

    return "approved" if approved else "denied"


async def cleanup_pending(pending_id: str) -> None:
    _pending_futures.pop(pending_id, None)
    r = get_dragonfly()
    await r.delete(f"{_KEY_PREFIX}{pending_id}")


async def get_pending_meta(pending_id: str) -> PendingActionMeta | None:
    r = get_dragonfly()
    raw = await r.get(f"{_KEY_PREFIX}{pending_id}")
    if raw is None:
        return None
    return PendingActionMeta.model_validate_json(raw)
