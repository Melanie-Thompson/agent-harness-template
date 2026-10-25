from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from config import Config
from db.mongo import close_mongo, init_mongo
from db.sessions import close_redis, init_redis
from routes import router
from schemas.session import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_redis()
    await init_mongo()
    yield
    await close_mongo()
    await close_redis()


app = FastAPI(lifespan=lifespan)

app.add_middleware(SessionMiddleware, secret_key=Config.MAGIC_LINK_SECRET)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")