from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class WriteOperation(str, Enum):
    INSERT_ONE = "insert_one"
    INSERT_MANY = "insert_many"
    UPDATE_ONE = "update_one"
    UPDATE_MANY = "update_many"
    DELETE_ONE = "delete_one"
    DELETE_MANY = "delete_many"


class QueryResult(BaseModel):
    operation: str
    collection: str
    created: int = 0
    updated: int = 0
    deleted: int = 0
    data: Any = None
