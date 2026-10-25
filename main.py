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


app = FastAPI(
    title="Basic Harness API",
    description=(
        "Reusable FastAPI template with OAuth (Google, GitHub), "
        "passwordless auth (magic links, OTP), Redis-backed sessions, "
        "and MongoDB persistence."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(SessionMiddleware, secret_key=Config.MAGIC_LINK_SECRET)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    """Return a simple status payload to confirm the service is running."""
    return HealthResponse(status="ok")