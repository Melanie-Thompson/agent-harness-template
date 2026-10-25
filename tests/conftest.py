import os

os.environ["MONGO_URL"] = "mongodb://localhost:27018"
os.environ["MONGO_DB"] = "tda_harness_test"
os.environ["REDIS_URL"] = "redis://localhost:6379/1"
os.environ["COOKIE_SECURE"] = "false"

import pytest
import redis.asyncio as aioredis
from httpx import ASGITransport, AsyncClient

from db.mongo import get_mongo_db, init_mongo, close_mongo
from db.sessions import init_redis, close_redis, get_redis
from main import app


@pytest.fixture(scope="session", autouse=True)
async def init_services():
    await init_redis()
    await init_mongo()
    yield
    await close_mongo()
    await close_redis()


@pytest.fixture(autouse=True)
async def clean_between_tests():
    db = get_mongo_db()
    collections = await db.list_collection_names()
    for coll in collections:
        await db.drop_collection(coll)
    r = get_redis()
    await r.flushdb()
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
