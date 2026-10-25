from pydantic import BaseModel


class SessionData(BaseModel):
    user_id: str
    permissions: list[str]


class MessageResponse(BaseModel):
    message: str


class MeResponse(BaseModel):
    user_id: str
    permissions: list[str]


class HealthResponse(BaseModel):
    status: str
