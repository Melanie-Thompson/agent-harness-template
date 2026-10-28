from copy import deepcopy
from typing import Any

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from auth import require
from db.meta import build_meta_create, build_meta_delete, build_meta_update
from db.mongo import get_mongo_db
from db.objectid import serialise_doc, to_objectid
from logging_decorator import log_call
from schemas.db import QueryResult, WriteOperation
from schemas.session import SessionData


class DbConnect:
    def __init__(self, user_id: str, db: AsyncIOMotorDatabase):
        self.user_id = user_id
        self.db = db

    def _scope_filter(self) -> dict:
        return {"_meta.user_id": self.user_id, "_meta.deleted_at": None}

    def _scoped_filter(self, filter: dict) -> dict:
        scoped = {**self._scope_filter(), **filter}
        if "_id" in scoped and isinstance(scoped["_id"], str):
            scoped["_id"] = to_objectid(scoped["_id"])
        return scoped

    async def find_by_id(self, collection: str, doc_id: str) -> dict | None:
        coll = self.db[collection]
        doc = await coll.find_one(self._scoped_filter({"_id": doc_id}))
        return serialise_doc(doc) if doc else None

    @log_call
    async def aggregate(self, collection: str, pipeline: list[dict]) -> QueryResult:
        coll = self.db[collection]
        scoped_pipeline = [{"$match": self._scope_filter()}, *pipeline]
        cursor = coll.aggregate(scoped_pipeline)
        data = serialise_doc(await cursor.to_list(length=None))
        return QueryResult(operation="aggregate", collection=collection, data=data)

    async def _insert_one(self, collection: str, *, document: dict, **_: Any) -> QueryResult:
        coll = self.db[collection]
        doc = {**document, **build_meta_create(self.user_id)}
        res = await coll.insert_one(doc)
        return QueryResult(
            operation="insert_one", collection=collection,
            created=1 if res.inserted_id else 0,
            data={"inserted_id": str(res.inserted_id)},
        )

    async def _insert_many(self, collection: str, *, documents: list[dict], **_: Any) -> QueryResult:
        coll = self.db[collection]
        docs = [{**d, **build_meta_create(self.user_id)} for d in documents]
        res = await coll.insert_many(docs)
        return QueryResult(
            operation="insert_many", collection=collection,
            created=len(res.inserted_ids),
            data={"inserted_ids": [str(i) for i in res.inserted_ids]},
        )

    async def _do_update(self, collection: str, fn_name: str, *, filter: dict, update: dict, **_: Any) -> QueryResult:
        coll = self.db[collection]
        delta = update.get("$set", {})
        meta_update = build_meta_update(self.user_id, delta)
        merged = deepcopy(update)
        merged.setdefault("$set", {}).update(meta_update["$set"])
        merged.setdefault("$push", {}).update(meta_update["$push"])
        fn = getattr(coll, fn_name)
        res = await fn(self._scoped_filter(filter), merged)
        return QueryResult(operation=fn_name, collection=collection, updated=res.modified_count)

    async def _update_one(self, collection: str, **kwargs: Any) -> QueryResult:
        return await self._do_update(collection, "update_one", **kwargs)

    async def _update_many(self, collection: str, **kwargs: Any) -> QueryResult:
        return await self._do_update(collection, "update_many", **kwargs)

    async def _do_delete(self, collection: str, *, many: bool, filter: dict, **_: Any) -> QueryResult:
        coll = self.db[collection]
        meta_delete = build_meta_delete(self.user_id)
        fn = coll.update_many if many else coll.update_one
        op = "delete_many" if many else "delete_one"
        res = await fn(self._scoped_filter(filter), meta_delete)
        return QueryResult(operation=op, collection=collection, deleted=res.modified_count)

    async def _delete_one(self, collection: str, **kwargs: Any) -> QueryResult:
        return await self._do_delete(collection, many=False, **kwargs)

    async def _delete_many(self, collection: str, **kwargs: Any) -> QueryResult:
        return await self._do_delete(collection, many=True, **kwargs)

    _WRITE_DISPATCH: dict[WriteOperation, str] = {
        WriteOperation.INSERT_ONE: "_insert_one",
        WriteOperation.INSERT_MANY: "_insert_many",
        WriteOperation.UPDATE_ONE: "_update_one",
        WriteOperation.UPDATE_MANY: "_update_many",
        WriteOperation.DELETE_ONE: "_delete_one",
        WriteOperation.DELETE_MANY: "_delete_many",
    }

    @log_call
    async def write(self, operation: WriteOperation, collection: str, **kwargs: Any) -> QueryResult:
        handler = getattr(self, self._WRITE_DISPATCH[operation])
        return await handler(collection, **kwargs)


def get_db(session: SessionData = Depends(require())) -> DbConnect:
    return DbConnect(session.user_id, get_mongo_db())
