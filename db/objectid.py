from typing import Annotated, Any

from bson import ObjectId
from pydantic import BeforeValidator


def to_str(v: Any) -> str:
    return str(v)


def to_objectid(v: str) -> ObjectId:
    return ObjectId(v)


def serialise_doc(doc: dict | list | Any) -> Any:
    if isinstance(doc, list):
        return [serialise_doc(d) for d in doc]
    if not isinstance(doc, dict):
        return doc
    return {k: to_str(v) if isinstance(v, ObjectId) else serialise_doc(v) for k, v in doc.items()}


PyObjectId = Annotated[str, BeforeValidator(to_str)]
