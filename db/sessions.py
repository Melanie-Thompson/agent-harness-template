import json
import secrets

import redis.asyncio as aioredis

from config import Config

_dragonfly: aioredis.Redis | None = None


async def init_dragonfly(url: str = Config.DRAGONFLY_URL) -> None:
    global _dragonfly
    _dragonfly = aioredis.from_url(url, decode_responses=True)


async def close_dragonfly() -> None:
    global _dragonfly
    if _dragonfly:
        await _dragonfly.aclose()
        _dragonfly = None


def get_dragonfly() -> aioredis.Redis:
    assert _dragonfly is not None, "Dragonfly not initialized — call init_dragonfly() first"
    return _dragonfly


async def create_session(user_id: str, permissions: list[str]) -> str:
    session_id = secrets.token_urlsafe(32)
    payload = json.dumps({"user_id": user_id, "permissions": permissions})
    await get_dragonfly().set(f"session:{session_id}", payload, ex=Config.SESSION_EXPIRE_SECONDS)
    return session_id


async def get_session(session_id: str) -> dict | None:
    raw = await get_dragonfly().get(f"session:{session_id}")
    if not raw:
        return None
    return json.loads(raw)


async def delete_session(session_id: str) -> None:
    await get_dragonfly().delete(f"session:{session_id}")