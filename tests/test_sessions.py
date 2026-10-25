from httpx import AsyncClient

from db.sessions import create_session


async def test_me_authenticated(client: AsyncClient):
    session_id = await create_session("user_1", ["read:self"])
    resp = await client.get("/sessions/me", cookies={"session_id": session_id})
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "user_1"
    assert "read:self" in data["permissions"]


async def test_me_unauthenticated(client: AsyncClient):
    resp = await client.get("/sessions/me")
    assert resp.status_code == 401


async def test_logout(client: AsyncClient):
    session_id = await create_session("user_1", ["read:self"])
    resp = await client.post("/sessions/logout", cookies={"session_id": session_id})
    assert resp.status_code == 200
    assert resp.json()["message"] == "Logged out successfully"

    resp = await client.get("/sessions/me", cookies={"session_id": session_id})
    assert resp.status_code == 401


async def test_expired_session_cookie(client: AsyncClient):
    resp = await client.get("/sessions/me", cookies={"session_id": "bogus_value"})
    assert resp.status_code == 401