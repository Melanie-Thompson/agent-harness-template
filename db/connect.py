from copy import deepcopy
from typing import Any

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from auth import require
from db.meta import build_meta_create, build_meta_delete, build_meta_update
from db.mongo import get_mongo_db
from db.objectid import serialise_doc, to_objectid
from errors import UnsupportedOperationError
from schemas.db import QueryInput, QueryResult
from schemas.session import SessionData


class DbConnect:
    def __init__(self, user_id: str, db: AsyncIOMotorDatabase):
        self.user_id = user_id
        self.db = db

    def _scoped_filter(self, filter: dict) -> dict:
        scoped = {"_meta.user_id": self.user_id, "_meta.deleted_at": None}
        scoped.update(filter)
        if "_id" in scoped and isinstance(scoped["_id"], str):
            scoped["_id"] = to_objectid(scoped["_id"])
        return scoped

    async def find_by_id(self, collection: str, doc_id: str) -> dict | None:
        coll = self.db[collection]
        doc = await coll.find_one(self._scoped_filter({"_id": doc_id}))
        return serialise_doc(doc) if doc else None

    async def _find(self, q: QueryInput) -> QueryResult:
        coll = self.db[q.collection]
        cursor = coll.find(self._scoped_filter(q.filter))
        data = serialise_doc(await cursor.to_list(length=None))
        return QueryResult(operation=q.operation.value, collection=q.collection, data=data)

    async def _find_one(self, q: QueryInput) -> QueryResult:
        coll = self.db[q.collection]
        doc = await coll.find_one(self._scoped_filter(q.filter))
        data = serialise_doc(doc) if doc else None
        return QueryResult(operation=q.operation.value, collection=q.collection, data=data)

    async def _insert_one(self, q: QueryInput) -> QueryResult:
        coll = self.db[q.collection]
        doc = {**(q.document or {}), **build_meta_create(self.user_id)}
        res = await coll.insert_one(doc)
        return QueryResult(
            operation=q.operation.value, collection=q.collection,
            created=1 if res.inserted_id else 0,
            data={"inserted_id": str(res.inserted_id)},
        )

    async def _insert_many(self, q: QueryInput) -> QueryResult:
        coll = self.db[q.collection]
        docs = [{**d, **build_meta_create(self.user_id)} for d in (q.documents or [])]
        res = await coll.insert_many(docs)
        return QueryResult(
            operation=q.operation.value, collection=q.collection,
            created=len(res.inserted_ids),
            data={"inserted_ids": [str(i) for i in res.inserted_ids]},
        )

    async def _update(self, q: QueryInput, *, many: bool = False) -> QueryResult:
        coll = self.db[q.collection]
        update = q.update or {}
        delta = update.get("$set", {})
        meta_update = build_meta_update(self.user_id, delta)
        merged = deepcopy(update)
        merged.setdefault("$set", {}).update(meta_update["$set"])
        merged.setdefault("$push", {}).update(meta_update["$push"])
        fn = coll.update_many if many else coll.update_one
        res = await fn(self._scoped_filter(q.filter), merged)
        return QueryResult(
            operation=q.operation.value, collection=q.collection,
            updated=res.modified_count,
        )

    async def _update_one(self, q: QueryInput) -> QueryResult:
        return await self._update(q, many=False)

    async def _update_many(self, q: QueryInput) -> QueryResult:
        return await self._update(q, many=True)

    async def _delete(self, q: QueryInput, *, many: bool = False) -> QueryResult:
        coll = self.db[q.collection]
        meta_delete = build_meta_delete(self.user_id)
        fn = coll.update_many if many else coll.update_one
        res = await fn(self._scoped_filter(q.filter), meta_delete)
        return QueryResult(
            operation=q.operation.value, collection=q.collection,
            deleted=res.modified_count,
        )

    async def _delete_one(self, q: QueryInput) -> QueryResult:
        return await self._delete(q, many=False)

    async def _delete_many(self, q: QueryInput) -> QueryResult:
        return await self._delete(q, many=True)

    _DISPATCH: dict[str, str] = {
        "find": "_find",
        "find_one": "_find_one",
        "insert_one": "_insert_one",
        "insert_many": "_insert_many",
        "update_one": "_update_one",
        "update_many": "_update_many",
        "delete_one": "_delete_one",
        "delete_many": "_delete_many",
    }

    async def query(self, q: QueryInput) -> QueryResult:
        method_name = self._DISPATCH.get(q.operation.value)
        if not method_name:
            raise UnsupportedOperationError(q.operation.value)
        handler = getattr(self, method_name)
        return await handler(q)


def get_db(session: SessionData = Depends(require())) -> DbConnect:
    return DbConnect(session.user_id, get_mongo_db())
