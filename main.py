from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware

from config import Config
from errors import HarnessException
from db.conversations import close_conversation_store, init_conversation_store
from db.mongo import close_mongo, init_mongo
from db.sessions import close_dragonfly, init_dragonfly
from routes import router
from schemas.session import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_dragonfly()
    await init_conversation_store()
    await init_mongo()
    yield
    await close_mongo()
    await close_conversation_store()
    await close_dragonfly()


app = FastAPI(
    title="Agent Harness API",
    description=(
        "AI agent harness with tool-calling, SSE streaming, "
        "configurable guardrails, gated writes, and production-ready "
        "auth, sessions, and MongoDB persistence."
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


@app.exception_handler(HarnessException)
async def harness_exception_handler(request: Request, exc: HarnessException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    """Return a simple status payload to confirm the service is running."""
    return HealthResponse(status="ok")