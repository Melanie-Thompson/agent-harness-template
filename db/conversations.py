import json
import time
from datetime import datetime, timezone
from uuid import uuid4

import redis.asyncio as aioredis

from config import Config
from schemas.conversation import ConversationMessage, ConversationSummary

_store: aioredis.Redis | None = None


def _make_id() -> str:
    return f"{int(time.time() * 1000):013d}-{uuid4().hex[:12]}"


async def init_conversation_store(url: str = Config.CONVERSATION_STORE_URL) -> None:
    global _store
    _store = aioredis.from_url(url, decode_responses=True)


async def close_conversation_store() -> None:
    global _store
    if _store:
        await _store.aclose()
        _store = None


def get_conversation_store() -> aioredis.Redis:
    assert _store is not None, "Conversation store not initialized — call init_conversation_store() first"
    return _store


async def create_conversation(user_id: str, session_id: str) -> str:
    conversation_id = _make_id()
    now = datetime.now(timezone.utc)
    r = get_conversation_store()
    ttl = Config.CONVERSATION_TTL

    await r.hset(f"conv_meta:{conversation_id}", mapping={
        "user_id": user_id,
        "session_id": session_id,
        "created_at": now.isoformat(),
    })
    await r.expire(f"conv_meta:{conversation_id}", ttl)

    score = now.timestamp() * 1000
    await r.zadd(f"user_convs:{user_id}", {conversation_id: score})
    await r.expire(f"user_convs:{user_id}", ttl)

    return conversation_id


async def add_message(msg: ConversationMessage) -> str:
    r = get_conversation_store()
    ttl = Config.CONVERSATION_TTL
    score = msg.timestamp.timestamp() * 1000

    await r.zadd(f"conv:{msg.conversation_id}", {msg.model_dump_json(): score})

    pipe = r.pipeline()
    pipe.expire(f"conv:{msg.conversation_id}", ttl)
    pipe.expire(f"conv_meta:{msg.conversation_id}", ttl)
    pipe.zadd(f"user_convs:{msg.user_id}", {msg.conversation_id: score})
    pipe.expire(f"user_convs:{msg.user_id}", ttl)
    await pipe.execute()

    return msg.message_id


async def get_messages(
    conversation_id: str,
    *,
    after: datetime | None = None,
    before: datetime | None = None,
    limit: int = 100,
) -> list[ConversationMessage]:
    r = get_conversation_store()
    min_score = after.timestamp() * 1000 if after else "-inf"
    max_score = before.timestamp() * 1000 if before else "+inf"

    raw = await r.zrangebyscore(
        f"conv:{conversation_id}",
        min=min_score,
        max=max_score,
        start=0,
        num=limit,
    )
    return [ConversationMessage.model_validate_json(item) for item in raw]


async def get_conversation_meta(conversation_id: str) -> dict | None:
    r = get_conversation_store()
    data = await r.hgetall(f"conv_meta:{conversation_id}")
    return data or None


async def list_conversations(user_id: str) -> list[ConversationSummary]:
    r = get_conversation_store()
    conv_ids = await r.zrevrange(f"user_convs:{user_id}", 0, -1, withscores=True)
    summaries = []

    for conv_id, last_score in conv_ids:
        meta = await r.hgetall(f"conv_meta:{conv_id}")
        if not meta:
            continue
        count = await r.zcard(f"conv:{conv_id}")
        summaries.append(ConversationSummary(
            conversation_id=conv_id,
            user_id=user_id,
            created_at=datetime.fromisoformat(meta["created_at"]),
            message_count=count,
            last_activity=datetime.fromtimestamp(last_score / 1000, tz=timezone.utc),
        ))

    return summaries


async def delete_conversation(conversation_id: str, user_id: str) -> bool:
    r = get_conversation_store()
    pipe = r.pipeline()
    pipe.delete(f"conv:{conversation_id}")
    pipe.delete(f"conv_meta:{conversation_id}")
    pipe.zrem(f"user_convs:{user_id}", conversation_id)
    results = await pipe.execute()
    return any(results)
