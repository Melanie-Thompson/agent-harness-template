from fastapi import APIRouter, Depends, Request, Response

from auth import require
from db.sessions import delete_session
from logging_decorator import log_call
from schemas.session import MeResponse, MessageResponse, SessionData

router = APIRouter(prefix="/sessions", tags=["sessions"])


@log_call
@router.post("/logout", response_model=MessageResponse, summary="Log out")
async def logout(request: Request, response: Response) -> MessageResponse:
    session_id = request.cookies.get("session_id")
    if session_id:
        await delete_session(session_id)
    response.delete_cookie(key="session_id")
    return MessageResponse(message="Logged out successfully")


@log_call
@router.get("/me", response_model=MeResponse, summary="Current user info")
async def me(session: SessionData = Depends(require("read:self"))) -> MeResponse:
    return MeResponse(user_id=session.user_id, permissions=session.permissions)