from fastapi import HTTPException, Request

from db.sessions import get_session
from schemas.session import SessionData


def require(*permissions: str):
    async def dependency(request: Request) -> SessionData:
        session_id = request.cookies.get("session_id")
        if not session_id:
            raise HTTPException(status_code=401, detail="Not authenticated")

        session = await get_session(session_id)
        if not session:
            raise HTTPException(status_code=401, detail="Session expired or invalid")

        data = SessionData(**session)

        if permissions:
            missing = [p for p in permissions if p not in data.permissions]
            if missing:
                raise HTTPException(status_code=403, detail=f"Missing permissions: {', '.join(missing)}")

        return data

    return dependency
