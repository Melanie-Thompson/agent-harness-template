from models.meta import Change, Meta, _now


def build_meta_create(user_id: str) -> dict:
    now = _now()
    meta = Meta(
        user_id=user_id,
        created_at=now,
        created_by=user_id,
        updated_at=now,
        updated_by=user_id,
        deleted_at=None,
        changes=[Change(ts=now, by=user_id, op="create", delta=None)],
    )
    return {"_meta": meta.model_dump()}


def build_meta_update(user_id: str, delta: dict) -> dict:
    now = _now()
    change = Change(ts=now, by=user_id, op="update", delta=delta)
    return {
        "$set": {
            "_meta.updated_at": now,
            "_meta.updated_by": user_id,
        },
        "$push": {
            "_meta.changes": change.model_dump(),
        },
    }


def build_meta_delete(user_id: str) -> dict:
    now = _now()
    change = Change(ts=now, by=user_id, op="delete", delta=None)
    return {
        "$set": {
            "_meta.deleted_at": now,
            "_meta.updated_at": now,
            "_meta.updated_by": user_id,
        },
        "$push": {
            "_meta.changes": change.model_dump(),
        },
    }
