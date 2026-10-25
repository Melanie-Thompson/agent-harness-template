from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Any, Literal

from ai.tools.base import harness_tool
from config import Config
from db.connect import DbConnect
from errors import CollectionNotAllowedError, NoDbContextError

_db_context: ContextVar[DbConnect | None] = ContextVar("_db_context", default=None)


def set_db_context(db: DbConnect) -> None:
    _db_context.set(db)


def get_db_context() -> DbConnect:
    db = _db_context.get()
    if db is None:
        raise NoDbContextError()
    return db


def _sanitise_filter(filter: dict[str, Any] | list | Any) -> Any:
    if isinstance(filter, dict):
        cleaned = {}
        for key, value in filter.items():
            if key.startswith("_meta"):
                continue
            cleaned[key] = _sanitise_filter(value)
        return cleaned
    if isinstance(filter, list):
        return [_sanitise_filter(item) for item in filter]
    return filter


def _check_collection(collection: str) -> None:
    allowed = Config.AGENT_ALLOWED_COLLECTIONS
    if allowed and collection not in allowed:
        raise CollectionNotAllowedError(collection, allowed)


@harness_tool(
    "db_query",
    description="Query the database. Use operation 'find' to get multiple documents or 'find_one' to get a single document. Provide a collection name and optional filter.",
    permissions=["db:read"],
)
async def db_query(
    operation: Literal["find", "find_one"],
    collection: str,
    filter: dict[str, Any] | None = None,
) -> str:
    _check_collection(collection)
    db = get_db_context()
    safe_filter = _sanitise_filter(filter or {})

    from schemas.db import QueryInput

    q = QueryInput(operation=operation, collection=collection, filter=safe_filter)
    result = await db.query(q)
    return json.dumps(result.model_dump(), default=str)


@harness_tool(
    "db_mutate",
    description="Modify database records. Supports insert_one (provide document), update_one (provide filter and update), or delete_one (provide filter). All operations are scoped to the current user.",
    permissions=["db:write"],
    gated=True,
)
async def db_mutate(
    operation: Literal["insert_one", "update_one", "delete_one"],
    collection: str,
    filter: dict[str, Any] | None = None,
    document: dict[str, Any] | None = None,
    update: dict[str, Any] | None = None,
) -> str:
    _check_collection(collection)
    db = get_db_context()
    safe_filter = _sanitise_filter(filter or {})

    from schemas.db import QueryInput

    q = QueryInput(
        operation=operation,
        collection=collection,
        filter=safe_filter,
        document=document,
        update=update,
    )
    result = await db.query(q)
    return json.dumps(result.model_dump(), default=str)
