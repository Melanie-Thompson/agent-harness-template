from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from config import Config
from errors import NotInitializedError

_client: AsyncIOMotorClient | None = None


async def init_mongo() -> None:
    global _client
    _client = AsyncIOMotorClient(Config.MONGO_URL)


async def close_mongo() -> None:
    global _client
    if _client:
        _client.close()
        _client = None


def get_mongo_db() -> AsyncIOMotorDatabase:
    if _client is None:
        raise NotInitializedError("MongoDB")
    return _client[Config.MONGO_DB]
