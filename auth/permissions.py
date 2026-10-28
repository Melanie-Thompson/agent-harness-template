from fastapi import HTTPException, Request

from db.sessions import get_session
from logging_config import get_logger
from schemas.session import SessionData

_log = get_logger("auth.permissions")


def require(*permissions: str):
    async def dependency(request: Request) -> SessionData:
        session_id = request.cookies.get("session_id")
        if not session_id:
            raise HTTPException(status_code=401, detail="Not authenticated")

        session = await get_session(session_id)
        if not session:
            _log.warning("auth.session_invalid", session_id=session_id[:8] + "...")
            raise HTTPException(status_code=401, detail="Session expired or invalid")

        data = SessionData(**session)

        if permissions:
            missing = [p for p in permissions if p not in data.permissions]
            if missing:
                _log.warning("auth.permission_denied", user_id=data.user_id, missing=missing)
                raise HTTPException(status_code=403, detail=f"Missing permissions: {', '.join(missing)}")

        return data

    return dependency
