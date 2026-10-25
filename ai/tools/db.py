from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Any

from ai.tools.base import harness_tool
from config import Config
from db.connect import DbConnect
from errors import CollectionNotAllowedError, NoDbContextError
from schemas.db import WriteOperation

_db_context: ContextVar[DbConnect | None] = ContextVar("_db_context", default=None)


def set_db_context(db: DbConnect) -> None:
    _db_context.set(db)


def get_db_context() -> DbConnect:
    db = _db_context.get()
    if db is None:
        raise NoDbContextError()
    return db


def _sanitise_doc(doc: dict[str, Any] | list | Any) -> Any:
    if isinstance(doc, dict):
        return {k: _sanitise_doc(v) for k, v in doc.items() if not k.startswith("_meta")}
    if isinstance(doc, list):
        return [_sanitise_doc(item) for item in doc]
    return doc


def _check_collection(collection: str) -> None:
    allowed = Config.AGENT_ALLOWED_COLLECTIONS
    if allowed and collection not in allowed:
        raise CollectionNotAllowedError(collection, allowed)


@harness_tool(
    "db_read",
    description=(
        "Query the database using a MongoDB aggregation pipeline. "
        "Provide the collection name and a pipeline (array of stage objects). "
        "User scoping is applied automatically. "
        'Examples: db_read(collection="items", pipeline=[{"$match": {"status": "active"}}]) '
        'or db_read(collection="orders", pipeline=[{"$match": {"total": {"$gt": 100}}}, {"$sort": {"total": -1}}, {"$limit": 5}])'
    ),
    permissions=["db:read"],
)
async def db_read(collection: str, pipeline: list[dict[str, Any]]) -> str:
    _check_collection(collection)
    sanitised = _sanitise_doc(pipeline)
    db = get_db_context()
    result = await db.aggregate(collection, sanitised)
    return json.dumps(result.model_dump(), default=str)


@harness_tool(
    "db_write",
    description=(
        "Write to the database. Provide operation, collection, and the relevant arguments. "
        "Operations: insert_one (document), insert_many (documents), "
        "update_one (filter + update), update_many (filter + update), "
        "delete_one (filter), delete_many (filter). "
        "User scoping and audit metadata are applied automatically."
    ),
    permissions=["db:write"],
    gated=True,
)
async def db_write(
    operation: str,
    collection: str,
    filter: dict[str, Any] | None = None,
    document: dict[str, Any] | None = None,
    documents: list[dict[str, Any]] | None = None,
    update: dict[str, Any] | None = None,
) -> str:
    _check_collection(collection)
    op = WriteOperation(operation)

    _SANITISE = {"filter"}
    raw = {"filter": filter, "document": document, "documents": documents, "update": update}
    kwargs = {k: (_sanitise_doc(v) if k in _SANITISE else v) for k, v in raw.items() if v is not None}

    db = get_db_context()
    result = await db.write(op, collection, **kwargs)
    return json.dumps(result.model_dump(), default=str)
