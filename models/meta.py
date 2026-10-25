from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from db.objectid import PyObjectId


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Change(BaseModel):
    ts: datetime
    by: str = Field(min_length=1)
    op: Literal["create", "update", "delete"]
    delta: dict | None = None


class Meta(BaseModel):
    user_id: str = Field(min_length=1)
    created_at: datetime
    created_by: str = Field(min_length=1)
    updated_at: datetime
    updated_by: str = Field(min_length=1)
    deleted_at: datetime | None = None
    changes: list[Change] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_timestamps(self):
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot be before created_at")
        if self.deleted_at is not None and self.deleted_at < self.created_at:
            raise ValueError("deleted_at cannot be before created_at")
        return self


class MongoDocument(BaseModel):
    id: PyObjectId = Field(default=None, alias="_id")
    meta: Meta = Field(alias="_meta")

    model_config = {"populate_by_name": True}
