import pytest

from db.connect import DbConnect
from db.mongo import get_mongo_db
from schemas.db import QueryInput


@pytest.fixture()
def db() -> DbConnect:
    return DbConnect("test_user", get_mongo_db())


@pytest.fixture()
def other_db() -> DbConnect:
    return DbConnect("other_user", get_mongo_db())


async def test_insert_and_find(db: DbConnect):
    result = await db.query(QueryInput(
        operation="insert_one",
        collection="items",
        document={"name": "widget"},
    ))
    assert result.created == 1
    assert result.data["inserted_id"]

    result = await db.query(QueryInput(
        operation="find",
        collection="items",
    ))
    assert len(result.data) == 1
    assert result.data[0]["name"] == "widget"
    assert result.data[0]["_meta"]["user_id"] == "test_user"
    assert result.data[0]["_meta"]["created_by"] == "test_user"
    assert result.data[0]["_meta"]["deleted_at"] is None
    assert len(result.data[0]["_meta"]["changes"]) == 1
    assert result.data[0]["_meta"]["changes"][0]["op"] == "create"


async def test_insert_many(db: DbConnect):
    result = await db.query(QueryInput(
        operation="insert_many",
        collection="items",
        documents=[{"name": "a"}, {"name": "b"}, {"name": "c"}],
    ))
    assert result.created == 3

    result = await db.query(QueryInput(
        operation="find",
        collection="items",
    ))
    assert len(result.data) == 3


async def test_find_one(db: DbConnect):
    await db.query(QueryInput(
        operation="insert_one",
        collection="items",
        document={"name": "target"},
    ))
    result = await db.query(QueryInput(
        operation="find_one",
        collection="items",
        filter={"name": "target"},
    ))
    assert result.data is not None
    assert result.data["name"] == "target"


async def test_find_by_id(db: DbConnect):
    insert_result = await db.query(QueryInput(
        operation="insert_one",
        collection="items",
        document={"name": "findme"},
    ))
    doc_id = insert_result.data["inserted_id"]

    found = await db.find_by_id("items", doc_id)
    assert found is not None
    assert found["name"] == "findme"


async def test_find_by_id_not_found(db: DbConnect):
    found = await db.find_by_id("items", "000000000000000000000000")
    assert found is None


async def test_update_one(db: DbConnect):
    await db.query(QueryInput(
        operation="insert_one",
        collection="items",
        document={"name": "old", "price": 10},
    ))
    result = await db.query(QueryInput(
        operation="update_one",
        collection="items",
        filter={"name": "old"},
        update={"$set": {"name": "new", "price": 20}},
    ))
    assert result.updated == 1

    result = await db.query(QueryInput(
        operation="find_one",
        collection="items",
        filter={"name": "new"},
    ))
    assert result.data["price"] == 20
    assert len(result.data["_meta"]["changes"]) == 2
    assert result.data["_meta"]["changes"][1]["op"] == "update"
    assert result.data["_meta"]["changes"][1]["delta"] == {"name": "new", "price": 20}


async def test_soft_delete(db: DbConnect):
    await db.query(QueryInput(
        operation="insert_one",
        collection="items",
        document={"name": "doomed"},
    ))
    result = await db.query(QueryInput(
        operation="delete_one",
        collection="items",
        filter={"name": "doomed"},
    ))
    assert result.deleted == 1

    result = await db.query(QueryInput(
        operation="find",
        collection="items",
    ))
    assert len(result.data) == 0


async def test_user_scoping(db: DbConnect, other_db: DbConnect):
    await db.query(QueryInput(
        operation="insert_one",
        collection="items",
        document={"name": "mine"},
    ))
    await other_db.query(QueryInput(
        operation="insert_one",
        collection="items",
        document={"name": "theirs"},
    ))

    my_items = await db.query(QueryInput(operation="find", collection="items"))
    assert len(my_items.data) == 1
    assert my_items.data[0]["name"] == "mine"

    their_items = await other_db.query(QueryInput(operation="find", collection="items"))
    assert len(their_items.data) == 1
    assert their_items.data[0]["name"] == "theirs"


async def test_cannot_update_other_users_doc(db: DbConnect, other_db: DbConnect):
    await db.query(QueryInput(
        operation="insert_one",
        collection="items",
        document={"name": "protected"},
    ))
    result = await other_db.query(QueryInput(
        operation="update_one",
        collection="items",
        filter={"name": "protected"},
        update={"$set": {"name": "hacked"}},
    ))
    assert result.updated == 0

    result = await db.query(QueryInput(
        operation="find_one",
        collection="items",
        filter={"name": "protected"},
    ))
    assert result.data["name"] == "protected"
