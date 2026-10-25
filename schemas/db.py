from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Operation(str, Enum):
    FIND = "find"
    FIND_ONE = "find_one"
    INSERT_ONE = "insert_one"
    INSERT_MANY = "insert_many"
    UPDATE_ONE = "update_one"
    UPDATE_MANY = "update_many"
    DELETE_ONE = "delete_one"
    DELETE_MANY = "delete_many"


class QueryInput(BaseModel):
    operation: Operation
    collection: str
    filter: dict[str, Any] = Field(default_factory=dict)
    document: dict[str, Any] | None = None
    documents: list[dict[str, Any]] | None = None
    update: dict[str, Any] | None = None


class QueryResult(BaseModel):
    operation: str
    collection: str
    created: int = 0
    updated: int = 0
    deleted: int = 0
    data: Any = None
