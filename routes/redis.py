from __future__ import annotations

import json

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from db.sessions import get_dragonfly
from db.conversations import get_conversation_store
from logging_decorator import log_call

router = APIRouter(prefix="/redis", tags=["redis"])


async def _scan_db(conn, db_label: str) -> list[dict]:
    entries = []
    cursor = 0
    while True:
        cursor, keys = await conn.scan(cursor, count=100)
        for key in keys:
            key_type = await conn.type(key)
            ttl = await conn.ttl(key)
            if key_type == "string":
                raw = await conn.get(key)
                try:
                    value = json.loads(raw) if raw else None
                except (json.JSONDecodeError, TypeError):
                    value = raw
            elif key_type == "list":
                raw_list = await conn.lrange(key, 0, 50)
                value = []
                for item in raw_list:
                    try:
                        value.append(json.loads(item))
                    except (json.JSONDecodeError, TypeError):
                        value.append(item)
            elif key_type == "hash":
                value = await conn.hgetall(key)
            elif key_type == "set":
                value = list(await conn.smembers(key))
            else:
                value = f"<{key_type}>"

            entries.append({
                "db": db_label,
                "key": key,
                "type": key_type,
                "ttl": ttl if ttl >= 0 else None,
                "value": value,
            })
        if cursor == 0:
            break
    return entries


@log_call
@router.get("/keys", summary="List all Redis/Dragonfly keys across both DBs")
async def list_keys() -> list[dict]:
    sessions_db = get_dragonfly()
    conversations_db = get_conversation_store()

    session_entries = await _scan_db(sessions_db, "sessions")
    conversation_entries = await _scan_db(conversations_db, "conversations")

    return session_entries + conversation_entries
