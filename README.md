# Agent Harness Template

An AI agent harness built on FastAPI. Gives LLMs the ability to call tools — starting with user-scoped database operations — through a multi-turn execution loop with SSE streaming, configurable guardrails, and gated write confirmation. Production-ready auth, permissions, and audit trail included.

## What it does

A user sends a prompt to `POST /agent/chat`. The harness:

1. Checks the prompt against configurable **input guardrails**
2. Loads **conversation history** from Dragonfly and injects it as context
3. Binds the user's **permitted tools** to the LLM via LangChain's `bind_tools`
4. Runs a **tool-calling loop** — the LLM reasons, calls tools, gets results, repeats until it has a final answer
5. Checks the response against **output guardrails**
6. **Streams** every step (thinking, tool calls, results, response) as Server-Sent Events
7. **Stores** all messages (user, assistant, tool) in the conversation

Write operations require **explicit user confirmation** before executing — the stream pauses until the client approves or denies.

## Features

### Agent & Tools
- **Tool-calling agent loop** — multi-turn LLM execution with LangChain `bind_tools`, SSE streaming, max iteration limits
- **Declarative tool registry** — `@harness_tool` decorator with auto-discovery, permission filtering, gated write support
- **Database tools** — user-scoped `db_query` (read) and `db_mutate` (gated write) with filter sanitisation and collection allowlisting
- **Gated writes** — write tools yield a pending event; execution pauses until the client confirms via `/agent/confirm/{id}`
- **Extensible** — add a tool by dropping a file in `ai/tools/` with a decorated async function

### LLM
- **Multi-provider** — Gemini (default), Anthropic, OpenAI via LangChain
- **Structured output** — `invoke_structured` returns typed Pydantic models
- **Configurable guardrails** — YAML-driven input/output safety checks with per-check toggles and optional cheaper guardrail model

### Conversations
- **Message store** — Dragonfly sorted sets with sliding TTL, per-user indexing
- **Context management** — automatic history injection into LLM calls with token/message budget trimming

### Infrastructure
- **Auth** — OAuth (Google + GitHub), magic link, email OTP. No passwords.
- **Sessions** — Dragonfly-backed with httponly cookies
- **Permissions** — Fine-grained RBAC via `Depends(require("scope:action"))`
- **MongoDB** — Async Motor with audited `_meta` (timestamps, user tracking, change deltas)
- **User scoping** — all queries automatically filtered to the logged-in user
- **Soft deletes** — sets `_meta.deleted_at`, excluded from queries automatically
- **Auto-discovery** — drop a file in `routes/` or `ai/tools/`, it's registered automatically

## Project Structure

```
main.py                  # FastAPI app, lifespan, middleware
config.py                # Env-backed Config class
ai/
  agent.py               # Multi-turn tool-calling execution loop (SSE)
  llm.py                 # LangChain LLM connector (Gemini, Anthropic, OpenAI)
  guardrails.py          # Configurable input/output safety checks
  context.py             # Conversation context builder for LLM calls
  tools/
    base.py              # @harness_tool decorator, registry, permission filtering
    db.py                # Database query/mutate tools with filter sanitisation
auth/
  oauth.py               # Google + GitHub OAuth via Authlib
  passwordless.py        # Magic link + OTP logic
  email.py               # Email sending abstraction (Resend)
  permissions.py         # require() dependency for RBAC
db/
  connect.py             # DbConnect class + universal query actualiser
  conversations.py       # Conversation message store (sorted sets)
  meta.py                # Meta builder functions (create/update/delete)
  mongo.py               # Motor client lifecycle
  objectid.py            # ObjectId serialisation (single source of truth)
  sessions.py            # Dragonfly session CRUD
models/
  meta.py                # Pydantic models: Change, Meta, MongoDocument
  user.py                # User model
schemas/
  agent.py               # AgentChatRequest, ConfirmRequest, AgentEvent
  ai.py                  # GuardrailConfig, GuardrailRulesConfig
  auth.py                # Auth request/response models
  conversation.py        # ConversationMessage, ConversationSummary, Role
  db.py                  # QueryInput, QueryResult, Operation enum
  session.py             # SessionData, MessageResponse, MeResponse, HealthResponse
routes/
  __init__.py            # Auto-discovers all routers in this package
  agent.py               # Agent chat (SSE) + gated tool confirmation
  auth.py                # Auth endpoints (OAuth, magic link, OTP)
  sessions.py            # Logout, /me
guardrails.example.yaml  # Example guardrail rules config
tests/                   # 128 tests across 12 files
```

## Setup

### Prerequisites

- Python 3.13+
- Dragonfly (or Redis — wire-compatible)
- MongoDB

### Install

```bash
uv sync
```

### Configure

```bash
cp .env.example .env
```

Edit `.env` with your credentials. For OAuth, set `GOOGLE_CLIENT_ID`/`SECRET` and `GITHUB_CLIENT_ID`/`SECRET`. For email, set `RESEND_API_KEY`. For the LLM, set `GOOGLE_API_KEY` (default provider is Gemini). Generate a secret for magic links:

```bash
openssl rand -hex 32
```

### Run

```bash
uv run uvicorn main:app --reload
```

Server starts at `http://localhost:8000`.

### Test

```bash
uv sync --extra test
uv run pytest tests/ -v
```

## API Endpoints

### Health
- `GET /health` — `{"status": "ok"}`

### Agent
- `POST /agent/chat` — SSE-streamed agent conversation with tool calling (requires `agent:use`)
- `POST /agent/confirm/{pending_id}` — approve or deny a gated tool call (requires `agent:use`)

### Auth
- `GET /auth/google` — Google OAuth login
- `GET /auth/github` — GitHub OAuth login
- `POST /auth/magic-link` — `{"email": "..."}` → sends magic link
- `POST /auth/otp/send` — `{"email": "..."}` → sends 6-digit code
- `POST /auth/otp/verify` — `{"email": "...", "code": "123456"}` → creates session

### Sessions
- `GET /sessions/me` — current user info (requires `read:self`)
- `POST /sessions/logout` — destroys session

## Adding Tools

Drop a file in `ai/tools/` — it's auto-discovered on import:

```python
# ai/tools/weather.py
from ai.tools.base import harness_tool

@harness_tool("get_weather", description="Get current weather for a city")
async def get_weather(city: str, units: str = "celsius") -> str:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://api.weather.com/{city}")
        data = resp.json()
    return f"{data['temp']}°{units[0].upper()}, {data['condition']}"
```

The `@harness_tool` decorator:
- Generates a LangChain tool schema from type hints (works with all providers)
- Registers the tool in the global registry
- Supports `permissions=["scope:action"]` for RBAC filtering
- Supports `gated=True` for write confirmation

### Built-in database tools

| Tool | Operations | Gated | Permission | Description |
|------|-----------|-------|------------|-------------|
| `db_query` | `find`, `find_one` | No | `db:read` | Read documents |
| `db_mutate` | `insert_one`, `update_one`, `delete_one` | Yes | `db:write` | Modify documents |

**Security**: All database tools run through `DbConnect`, which enforces user scoping via `_meta.user_id`. LLM-generated filters are recursively sanitised to strip `_meta` keys, preventing scope escape. A configurable collection allowlist restricts which collections the LLM can target.

## SSE Event Stream

The `/agent/chat` endpoint streams events as the agent works:

| Event | Description |
|-------|-------------|
| `thinking` | Agent is processing (includes iteration number) |
| `tool_call` | LLM requested a tool call (tool name + args) |
| `tool_result` | Tool executed successfully (result data) |
| `tool_pending` | Gated tool awaiting confirmation (pending_id) |
| `tool_confirmed` | User approved a gated tool |
| `tool_denied` | User denied a gated tool |
| `response` | Final agent response |
| `blocked` | Guardrail blocked input or output |
| `error` | Error (tool not found, max iterations, etc.) |

## Adding Endpoints

Create a file in `routes/` with a `router`:

```python
from fastapi import APIRouter, Depends
from auth import require
from db.connect import DbConnect, get_db

router = APIRouter(prefix="/items", tags=["items"])

@router.get("/")
async def list_items(db: DbConnect = Depends(get_db)):
    return await db.query(QueryInput(operation="find", collection="items"))
```

It's auto-registered — no changes to `main.py` needed.

## Adding Models

Extend `MongoDocument` in `models/`:

```python
from models.meta import MongoDocument

class Order(MongoDocument):
    product: str
    quantity: int
    total: float
```

The `_meta` audit trail and `_id` are inherited automatically.

## LLM & Guardrails

### Structured LLM output

```python
from pydantic import BaseModel
from ai.llm import invoke_structured

class Sentiment(BaseModel):
    label: str
    score: float

result = await invoke_structured("Analyse: 'I love this'", Sentiment)
# result.label == "positive", result.score == 0.95
```

### Guarded calls

`invoke_guarded` wraps the LLM call with configurable input/output safety checks:

```python
from ai.guardrails import invoke_guarded

resp = await invoke_guarded("user prompt", Sentiment)
if resp.blocked:
    print(resp.block_reason)
else:
    print(resp.result)
```

### Configuring guardrails

Copy `guardrails.example.yaml` and point to it:

```bash
cp guardrails.example.yaml guardrails.yaml
# Edit with your rules, set GUARDRAILS_CONFIG_PATH=guardrails.yaml in .env
```

The YAML file lets you enable/disable input and output checks independently, replace the default rejection categories, and add custom rules per deployment.

## Conversation Store

Conversations are stored in Dragonfly sorted sets with Unix-millisecond scores for ordering. Each message is a full JSON document (role, content, timestamp, token usage, model, metadata). All keys use a sliding TTL — activity refreshes expiry.

Pass `conversation_id` to `invoke_structured`, `invoke_guarded`, or the agent loop and conversation history is automatically managed.

```python
from db.conversations import create_conversation, add_message
from schemas.conversation import ConversationMessage, Role
from datetime import datetime, timezone

conv_id = await create_conversation(user_id="user-1", session_id="sess-1")

await add_message(ConversationMessage(
    conversation_id=conv_id,
    message_id="msg-001",
    session_id="sess-1",
    user_id="user-1",
    role=Role.USER,
    timestamp=datetime.now(timezone.utc),
    content="What is the capital of France?",
))
```

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| **Agent** | | |
| `AGENT_MAX_ITERATIONS` | `10` | Max tool-calling loop iterations |
| `AGENT_ALLOWED_COLLECTIONS` | _(empty = all)_ | Comma-separated collection allowlist for DB tools |
| `AGENT_SYSTEM_PROMPT` | _(default)_ | System prompt for the agent |
| **LLM** | | |
| `LLM_PROVIDER` | `gemini` | LLM provider: `gemini`, `anthropic`, `openai` |
| `LLM_MODEL` | _(provider default)_ | Model override |
| `LLM_TEMPERATURE` | `0` | Temperature |
| `GOOGLE_API_KEY` | | Gemini API key |
| `ANTHROPIC_API_KEY` | | Anthropic API key |
| `OPENAI_API_KEY` | | OpenAI API key |
| **Guardrails** | | |
| `GUARDRAILS_ENABLED` | `true` | Global kill switch |
| `GUARDRAILS_CONFIG_PATH` | _(empty)_ | Path to YAML rules file |
| `GUARDRAILS_MODEL` | _(empty)_ | Cheaper model for safety checks |
| **Conversations** | | |
| `DRAGONFLY_URL` | `redis://localhost:6379/0` | Dragonfly connection (falls back to `REDIS_URL`) |
| `HARNESS_CONVERSATION_STORE_URL` | _(same as DRAGONFLY_URL)_ | Separate DB for conversations |
| `HARNESS_CONVERSATION_TTL` | `1800` | Sliding TTL in seconds |
| `CONTEXT_MAX_MESSAGES` | `50` | Max messages to include as context |
| `CONTEXT_MAX_TOKENS` | `4000` | Approximate token budget for context |
