import json
import secrets

import redis.asyncio as aioredis

from config import Config

_redis: aioredis.Redis | None = None


async def init_redis(url: str = Config.REDIS_URL) -> None:
    global _redis
    _redis = aioredis.from_url(url, decode_responses=True)


async def close_redis() -> None:
    global _redis
    if _redis:
        await _redis.aclose()
        _redis = None


def get_redis() -> aioredis.Redis:
    assert _redis is not None, "Redis not initialized — call init_redis() first"
    return _redis


async def create_session(user_id: str, permissions: list[str]) -> str:
    session_id = secrets.token_urlsafe(32)
    payload = json.dumps({"user_id": user_id, "permissions": permissions})
    await get_redis().set(f"session:{session_id}", payload, ex=Config.SESSION_EXPIRE_SECONDS)
    return session_id


async def get_session(session_id: str) -> dict | None:
    raw = await get_redis().get(f"session:{session_id}")
    if not raw:
        return None
    return json.loads(raw)


async def delete_session(session_id: str) -> None:
    await get_redis().delete(f"session:{session_id}")