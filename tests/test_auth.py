from unittest.mock import AsyncMock, patch

from httpx import AsyncClient

from auth.passwordless import verify_magic_link, _get_serializer
from db.mongo import get_mongo_db
from db.sessions import get_redis


async def test_otp_send(client: AsyncClient):
    with patch("auth.passwordless.send_email", new_callable=AsyncMock) as mock_send:
        resp = await client.post("/auth/otp/send", json={"email": "test@example.com"})
        assert resp.status_code == 200
        assert resp.json()["message"] == "Verification code sent"
        mock_send.assert_called_once()

    r = get_redis()
    code = await r.get("otp:test@example.com")
    assert code is not None
    assert len(code) == 6


async def test_otp_verify_success(client: AsyncClient):
    r = get_redis()
    await r.set("otp:test@example.com", "123456", ex=300)

    resp = await client.post("/auth/otp/verify", json={"email": "test@example.com", "code": "123456"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["message"] == "Logged in successfully"
    assert data["user_id"]

    stored = await r.get("otp:test@example.com")
    assert stored is None


async def test_otp_verify_wrong_code(client: AsyncClient):
    r = get_redis()
    await r.set("otp:test@example.com", "123456", ex=300)

    resp = await client.post("/auth/otp/verify", json={"email": "test@example.com", "code": "999999"})
    assert resp.status_code == 400


async def test_otp_verify_expired(client: AsyncClient):
    resp = await client.post("/auth/otp/verify", json={"email": "nobody@example.com", "code": "123456"})
    assert resp.status_code == 400


async def test_magic_link_send(client: AsyncClient):
    with patch("auth.passwordless.send_email", new_callable=AsyncMock) as mock_send:
        resp = await client.post("/auth/magic-link", json={"email": "test@example.com"})
        assert resp.status_code == 200
        assert resp.json()["message"] == "Magic link sent"
        mock_send.assert_called_once()


async def test_magic_link_verify_success(client: AsyncClient):
    serializer = _get_serializer()
    token = serializer.dumps("test@example.com", salt="magic-link")

    resp = await client.get(f"/auth/magic/verify?token={token}", follow_redirects=False)
    assert resp.status_code == 307
    assert "session_id" in resp.cookies


async def test_magic_link_verify_invalid(client: AsyncClient):
    resp = await client.get("/auth/magic/verify?token=bogus-token")
    assert resp.status_code == 400


async def test_otp_creates_user_in_mongo(client: AsyncClient):
    r = get_redis()
    await r.set("otp:newuser@example.com", "111111", ex=300)

    resp = await client.post("/auth/otp/verify", json={"email": "newuser@example.com", "code": "111111"})
    assert resp.status_code == 200

    db = get_mongo_db()
    user = await db.users.find_one({"email": "newuser@example.com"})
    assert user is not None
    assert user["email_verified"] is True
    assert user["auth_provider"] == "otp"


async def test_otp_reuses_existing_user(client: AsyncClient):
    r = get_redis()
    await r.set("otp:repeat@example.com", "111111", ex=300)
    resp1 = await client.post("/auth/otp/verify", json={"email": "repeat@example.com", "code": "111111"})

    await r.set("otp:repeat@example.com", "222222", ex=300)
    resp2 = await client.post("/auth/otp/verify", json={"email": "repeat@example.com", "code": "222222"})

    assert resp1.json()["user_id"] == resp2.json()["user_id"]

    db = get_mongo_db()
    count = await db.users.count_documents({"email": "repeat@example.com"})
    assert count == 1


async def test_session_works_after_otp_login(client: AsyncClient):
    r = get_redis()
    await r.set("otp:flow@example.com", "123456", ex=300)

    resp = await client.post("/auth/otp/verify", json={"email": "flow@example.com", "code": "123456"})
    cookies = dict(resp.cookies)

    resp = await client.get("/sessions/me", cookies=cookies)
    assert resp.status_code == 200
    assert resp.json()["user_id"]