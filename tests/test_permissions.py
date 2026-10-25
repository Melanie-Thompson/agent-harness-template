from httpx import AsyncClient

from db.sessions import create_session


async def test_permission_granted(client: AsyncClient):
    session_id = await create_session("user_1", ["read:self"])
    resp = await client.get("/sessions/me", cookies={"session_id": session_id})
    assert resp.status_code == 200


async def test_permission_denied(client: AsyncClient):
    session_id = await create_session("user_1", ["write:other"])
    resp = await client.get("/sessions/me", cookies={"session_id": session_id})
    assert resp.status_code == 403
    assert "read:self" in resp.json()["detail"]


async def test_multiple_permissions(client: AsyncClient):
    session_id = await create_session("user_1", ["read:self", "write:reports", "admin:all"])
    resp = await client.get("/sessions/me", cookies={"session_id": session_id})
    assert resp.status_code == 200
    assert set(resp.json()["permissions"]) == {"read:self", "write:reports", "admin:all"}
