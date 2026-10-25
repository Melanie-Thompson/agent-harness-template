from copy import deepcopy

from fastapi import Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from auth import require
from db.meta import build_meta_create, build_meta_delete, build_meta_update
from db.mongo import get_mongo_db
from db.objectid import serialise_doc, to_objectid
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

    async def query(self, q: QueryInput) -> QueryResult:
        coll = self.db[q.collection]
        result = QueryResult(operation=q.operation.value, collection=q.collection)

        if q.operation == "find":
            cursor = coll.find(self._scoped_filter(q.filter))
            result.data = serialise_doc(await cursor.to_list(length=None))

        elif q.operation == "find_one":
            doc = await coll.find_one(self._scoped_filter(q.filter))
            result.data = serialise_doc(doc) if doc else None

        elif q.operation == "insert_one":
            doc = {**(q.document or {}), **build_meta_create(self.user_id)}
            res = await coll.insert_one(doc)
            result.created = 1 if res.inserted_id else 0
            result.data = {"inserted_id": str(res.inserted_id)}

        elif q.operation == "insert_many":
            docs = [{**d, **build_meta_create(self.user_id)} for d in (q.documents or [])]
            res = await coll.insert_many(docs)
            result.created = len(res.inserted_ids)
            result.data = {"inserted_ids": [str(i) for i in res.inserted_ids]}

        elif q.operation in ("update_one", "update_many"):
            update = q.update or {}
            delta = update.get("$set", {})
            meta_update = build_meta_update(self.user_id, delta)
            merged = deepcopy(update)
            merged.setdefault("$set", {}).update(meta_update["$set"])
            merged.setdefault("$push", {}).update(meta_update["$push"])
            if q.operation == "update_one":
                res = await coll.update_one(self._scoped_filter(q.filter), merged)
            else:
                res = await coll.update_many(self._scoped_filter(q.filter), merged)
            result.updated = res.modified_count

        elif q.operation in ("delete_one", "delete_many"):
            meta_delete = build_meta_delete(self.user_id)
            if q.operation == "delete_one":
                res = await coll.update_one(self._scoped_filter(q.filter), meta_delete)
            else:
                res = await coll.update_many(self._scoped_filter(q.filter), meta_delete)
            result.deleted = res.modified_count

        else:
            raise ValueError(f"Unsupported operation: {q.operation}")

        return result


def get_db(session: SessionData = Depends(require())) -> DbConnect:
    return DbConnect(session.user_id, get_mongo_db())
