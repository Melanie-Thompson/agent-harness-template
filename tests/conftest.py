import os

os.environ["MONGO_URL"] = "mongodb://localhost:27018"
os.environ["MONGO_DB"] = "tda_harness_test"
os.environ["DRAGONFLY_URL"] = "redis://localhost:6379/1"
os.environ["HARNESS_CONVERSATION_STORE_URL"] = "redis://localhost:6379/2"
os.environ["COOKIE_SECURE"] = "false"

import pytest
import redis.asyncio as aioredis
from httpx import ASGITransport, AsyncClient

from db.conversations import init_conversation_store, close_conversation_store, get_conversation_store
from db.mongo import get_mongo_db, init_mongo, close_mongo
from db.sessions import init_dragonfly, close_dragonfly, get_dragonfly
from main import app


@pytest.fixture(scope="session", autouse=True)
async def init_services():
    await init_dragonfly()
    await init_conversation_store()
    await init_mongo()
    yield
    await close_mongo()
    await close_conversation_store()
    await close_dragonfly()


@pytest.fixture(autouse=True)
async def clean_between_tests():
    db = get_mongo_db()
    collections = await db.list_collection_names()
    for coll in collections:
        await db.drop_collection(coll)
    r = get_dragonfly()
    await r.flushdb()
    cs = get_conversation_store()
    await cs.flushdb()
    yield


@pytest.fixture()
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.fixture()
async def auth_cookies(client: AsyncClient) -> dict:
    resp = await client.post("/sessions/login")
    assert resp.status_code == 200
    return dict(resp.cookies)
