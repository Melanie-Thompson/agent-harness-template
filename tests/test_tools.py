from __future__ import annotations

import json

import pytest

from ai.tools.base import _registry, clear_registry, get_tool, get_tools, harness_tool
from ai.tools.db import _sanitise_doc, db_read, db_write, set_db_context
from config import Config
from db.connect import DbConnect
from db.meta import build_meta_create
from db.mongo import get_mongo_db


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = dict(_registry)
    yield
    _registry.clear()
    _registry.update(saved)


@pytest.fixture()
def db() -> DbConnect:
    return DbConnect("test_user", get_mongo_db())


# --- Registry tests ---


def test_harness_tool_registers():
    @harness_tool("test_echo", description="Echo input")
    async def echo(text: str) -> str:
        return text

    assert "test_echo" in _registry
    assert _registry["test_echo"].name == "test_echo"


def test_get_tools_returns_all():
    tools = get_tools()
    names = [t.name for t in tools]
    assert "db_read" in names
    assert "db_write" in names


def test_get_tools_filters_by_permissions():
    tools = get_tools(permissions=["db:read"])
    names = [t.name for t in tools]
    assert "db_read" in names
    assert "db_write" not in names


def test_get_tools_all_permissions():
    tools = get_tools(permissions=["db:read", "db:write"])
    names = [t.name for t in tools]
    assert "db_read" in names
    assert "db_write" in names


def test_get_tool_by_name():
    assert get_tool("db_read") is not None
    assert get_tool("db_write") is not None
    assert get_tool("nonexistent") is None


def test_tool_metadata():
    read_tool = get_tool("db_read")
    assert read_tool.metadata["gated"] is False
    assert read_tool.metadata["permissions"] == ["db:read"]

    write_tool = get_tool("db_write")
    assert write_tool.metadata["gated"] is True
    assert write_tool.metadata["permissions"] == ["db:write"]


def test_tool_schema_generation():
    tool = get_tool("db_read")
    schema = tool.args_schema.model_json_schema()
    assert "properties" in schema
    assert "collection" in schema["properties"]
    assert "pipeline" in schema["properties"]


# --- Document sanitisation ---


def test_sanitise_doc_strips_meta():
    f = {"name": "test", "_meta.user_id": "attacker", "_meta.deleted_at": None}
    cleaned = _sanitise_doc(f)
    assert "_meta.user_id" not in cleaned
    assert "_meta.deleted_at" not in cleaned
    assert cleaned["name"] == "test"


def test_sanitise_doc_strips_nested_meta():
    f = {"$and": [{"name": "test"}, {"_meta.user_id": "attacker"}]}
    cleaned = _sanitise_doc(f)
    inner = cleaned["$and"][1]
    assert "_meta.user_id" not in inner


def test_sanitise_doc_preserves_normal_keys():
    f = {"status": "active", "count": {"$gt": 5}}
    cleaned = _sanitise_doc(f)
    assert cleaned == f


def test_sanitise_doc_deeply_nested():
    f = {"outer": {"_meta.secret": "bad", "ok": "good"}}
    cleaned = _sanitise_doc(f)
    assert cleaned == {"outer": {"ok": "good"}}


# --- Collection allowlist ---


def test_collection_allowlist_blocks(monkeypatch):
    monkeypatch.setattr(Config, "AGENT_ALLOWED_COLLECTIONS", ["items", "orders"])

    from errors import CollectionNotAllowedError

    with pytest.raises(CollectionNotAllowedError, match="not in the allowed list"):
        from ai.tools.db import _check_collection
        _check_collection("secrets")


def test_collection_allowlist_allows(monkeypatch):
    monkeypatch.setattr(Config, "AGENT_ALLOWED_COLLECTIONS", ["items", "orders"])

    from ai.tools.db import _check_collection
    _check_collection("items")


def test_collection_allowlist_empty_allows_all(monkeypatch):
    monkeypatch.setattr(Config, "AGENT_ALLOWED_COLLECTIONS", [])

    from ai.tools.db import _check_collection
    _check_collection("anything")


# --- DB tool integration: reads ---


async def test_db_read_find_all(db: DbConnect):
    set_db_context(db)
    coll = get_mongo_db()["items"]
    await coll.insert_one({"name": "widget", **build_meta_create("test_user")})

    result_json = await db_read.ainvoke({"collection": "items", "pipeline": []})
    result = json.loads(result_json)
    assert result["operation"] == "aggregate"
    assert len(result["data"]) == 1
    assert result["data"][0]["name"] == "widget"


async def test_db_read_with_match(db: DbConnect):
    set_db_context(db)
    coll = get_mongo_db()["items"]
    await coll.insert_one({"name": "a", "price": 5, **build_meta_create("test_user")})
    await coll.insert_one({"name": "b", "price": 50, **build_meta_create("test_user")})

    result_json = await db_read.ainvoke({
        "collection": "items",
        "pipeline": [{"$match": {"price": {"$gt": 10}}}],
    })
    result = json.loads(result_json)
    assert len(result["data"]) == 1
    assert result["data"][0]["name"] == "b"


async def test_db_read_compound_pipeline(db: DbConnect):
    set_db_context(db)
    coll = get_mongo_db()["items"]
    for i in range(5):
        await coll.insert_one({"name": f"item_{i}", "price": i * 10, **build_meta_create("test_user")})

    result_json = await db_read.ainvoke({
        "collection": "items",
        "pipeline": [
            {"$match": {"price": {"$gte": 20}}},
            {"$sort": {"price": -1}},
            {"$limit": 2},
        ],
    })
    result = json.loads(result_json)
    assert len(result["data"]) == 2
    assert result["data"][0]["price"] == 40
    assert result["data"][1]["price"] == 30


async def test_db_read_user_scoped(db: DbConnect):
    set_db_context(db)
    coll = get_mongo_db()["items"]
    await coll.insert_one({"name": "mine", **build_meta_create("test_user")})
    await coll.insert_one({"name": "theirs", **build_meta_create("other_user")})

    result_json = await db_read.ainvoke({"collection": "items", "pipeline": []})
    result = json.loads(result_json)
    names = [d["name"] for d in result["data"]]
    assert "mine" in names
    assert "theirs" not in names


async def test_db_read_group_aggregation(db: DbConnect):
    set_db_context(db)
    coll = get_mongo_db()["items"]
    await coll.insert_one({"category": "A", "price": 10, **build_meta_create("test_user")})
    await coll.insert_one({"category": "A", "price": 20, **build_meta_create("test_user")})
    await coll.insert_one({"category": "B", "price": 30, **build_meta_create("test_user")})

    result_json = await db_read.ainvoke({
        "collection": "items",
        "pipeline": [
            {"$group": {"_id": "$category", "total": {"$sum": "$price"}}},
            {"$sort": {"_id": 1}},
        ],
    })
    result = json.loads(result_json)
    assert len(result["data"]) == 2
    assert result["data"][0]["_id"] == "A"
    assert result["data"][0]["total"] == 30
    assert result["data"][1]["_id"] == "B"
    assert result["data"][1]["total"] == 30


# --- DB tool integration: writes ---


async def test_db_write_insert_one(db: DbConnect):
    set_db_context(db)

    result_json = await db_write.ainvoke({
        "operation": "insert_one",
        "collection": "items",
        "document": {"name": "widget"},
    })
    result = json.loads(result_json)
    assert result["created"] == 1
    assert result["data"]["inserted_id"]


async def test_db_write_update_one(db: DbConnect):
    set_db_context(db)
    coll = get_mongo_db()["items"]
    await coll.insert_one({"name": "old", "price": 10, **build_meta_create("test_user")})

    result_json = await db_write.ainvoke({
        "operation": "update_one",
        "collection": "items",
        "filter": {"name": "old"},
        "update": {"$set": {"name": "new", "price": 20}},
    })
    result = json.loads(result_json)
    assert result["updated"] == 1


async def test_db_write_delete_one(db: DbConnect):
    set_db_context(db)
    coll = get_mongo_db()["items"]
    await coll.insert_one({"name": "doomed", **build_meta_create("test_user")})

    result_json = await db_write.ainvoke({
        "operation": "delete_one",
        "collection": "items",
        "filter": {"name": "doomed"},
    })
    result = json.loads(result_json)
    assert result["deleted"] == 1
