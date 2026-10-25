from datetime import timezone

from db.meta import build_meta_create, build_meta_delete, build_meta_update
from models.meta import Change, Meta, MongoDocument


def test_build_meta_create():
    result = build_meta_create("user_1")
    meta = result["_meta"]
    assert meta["user_id"] == "user_1"
    assert meta["created_by"] == "user_1"
    assert meta["updated_by"] == "user_1"
    assert meta["deleted_at"] is None
    assert len(meta["changes"]) == 1
    assert meta["changes"][0]["op"] == "create"
    assert meta["changes"][0]["delta"] is None
    assert meta["created_at"].tzinfo == timezone.utc


def test_build_meta_update():
    delta = {"name": "new_name", "price": 42}
    result = build_meta_update("user_1", delta)
    assert "_meta.updated_at" in result["$set"]
    assert result["$set"]["_meta.updated_by"] == "user_1"
    change = result["$push"]["_meta.changes"]
    assert change["op"] == "update"
    assert change["delta"] == delta


def test_build_meta_delete():
    result = build_meta_delete("user_1")
    assert result["$set"]["_meta.deleted_at"] is not None
    assert result["$set"]["_meta.updated_by"] == "user_1"
    change = result["$push"]["_meta.changes"]
    assert change["op"] == "delete"
    assert change["delta"] is None


def test_pydantic_models():
    change = Change(ts="2026-01-01T00:00:00Z", by="user_1", op="create")
    assert change.op == "create"

    meta = Meta(
        user_id="user_1",
        created_at="2026-01-01T00:00:00Z",
        created_by="user_1",
        updated_at="2026-01-01T00:00:00Z",
        updated_by="user_1",
        changes=[change],
    )
    assert meta.user_id == "user_1"
    assert meta.deleted_at is None


def test_mongo_document_from_raw():
    raw = {
        "_id": "507f1f77bcf86cd799439011",
        "_meta": {
            "user_id": "user_1",
            "created_at": "2026-01-01T00:00:00Z",
            "created_by": "user_1",
            "updated_at": "2026-01-01T00:00:00Z",
            "updated_by": "user_1",
            "deleted_at": None,
            "changes": [],
        },
    }
    doc = MongoDocument(**raw)
    assert doc.id == "507f1f77bcf86cd799439011"
    assert doc.meta.user_id == "user_1"
