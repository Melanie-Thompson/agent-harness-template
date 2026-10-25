from __future__ import annotations

import json

import pytest

from ai.tools.base import _registry, clear_registry, get_tool, get_tools, harness_tool
from ai.tools.db import _sanitise_filter, db_mutate, db_query, set_db_context
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
    assert "db_query" in names
    assert "db_mutate" in names


def test_get_tools_filters_by_permissions():
    tools = get_tools(permissions=["db:read"])
    names = [t.name for t in tools]
    assert "db_query" in names
    assert "db_mutate" not in names


def test_get_tools_all_permissions():
    tools = get_tools(permissions=["db:read", "db:write"])
    names = [t.name for t in tools]
    assert "db_query" in names
    assert "db_mutate" in names


def test_get_tool_by_name():
    assert get_tool("db_query") is not None
    assert get_tool("nonexistent") is None


def test_tool_metadata():
    tool = get_tool("db_mutate")
    assert tool is not None
    assert tool.metadata["gated"] is True
    assert tool.metadata["permissions"] == ["db:write"]


def test_tool_schema_generation():
    tool = get_tool("db_query")
    schema = tool.args_schema.model_json_schema() if hasattr(tool.args_schema, "model_json_schema") else {}
    assert "properties" in schema
    assert "operation" in schema["properties"]
    assert "collection" in schema["properties"]


# --- Filter sanitisation ---


def test_sanitise_filter_strips_meta():
    f = {"name": "test", "_meta.user_id": "attacker", "_meta.deleted_at": None}
    cleaned = _sanitise_filter(f)
    assert "_meta.user_id" not in cleaned
    assert "_meta.deleted_at" not in cleaned
    assert cleaned["name"] == "test"


def test_sanitise_filter_strips_nested_meta():
    f = {"$and": [{"name": "test"}, {"_meta.user_id": "attacker"}]}
    cleaned = _sanitise_filter(f)
    assert "_meta.user_id" not in cleaned.get("$and", [{}])[1] if isinstance(cleaned.get("$and"), list) else True


def test_sanitise_filter_preserves_normal_keys():
    f = {"status": "active", "count": {"$gt": 5}}
    cleaned = _sanitise_filter(f)
    assert cleaned == f


def test_sanitise_filter_deeply_nested():
    f = {"outer": {"_meta.secret": "bad", "ok": "good"}}
    cleaned = _sanitise_filter(f)
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


# --- DB tool integration ---


async def test_db_query_find(db: DbConnect):
    set_db_context(db)
    coll = get_mongo_db()["items"]
    doc = {"name": "widget", **build_meta_create("test_user")}
    await coll.insert_one(doc)

    result_json = await db_query.ainvoke({"operation": "find", "collection": "items"})
    result = json.loads(result_json)
    assert result["operation"] == "find"
    assert len(result["data"]) == 1
    assert result["data"][0]["name"] == "widget"


async def test_db_query_find_one(db: DbConnect):
    set_db_context(db)
    coll = get_mongo_db()["items"]
    doc = {"name": "gadget", **build_meta_create("test_user")}
    await coll.insert_one(doc)

    result_json = await db_query.ainvoke({"operation": "find_one", "collection": "items", "filter": {"name": "gadget"}})
    result = json.loads(result_json)
    assert result["data"]["name"] == "gadget"


async def test_db_query_user_scoped(db: DbConnect):
    set_db_context(db)
    coll = get_mongo_db()["items"]
    await coll.insert_one({"name": "mine", **build_meta_create("test_user")})
    await coll.insert_one({"name": "theirs", **build_meta_create("other_user")})

    result_json = await db_query.ainvoke({"operation": "find", "collection": "items"})
    result = json.loads(result_json)
    names = [d["name"] for d in result["data"]]
    assert "mine" in names
    assert "theirs" not in names
