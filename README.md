# Basic Harness

A starting-point FastAPI template for building more elaborate harnesses. Production-ready auth, audited MongoDB, Dragonfly sessions, conversation storage, and permission-based access control — ready to extend.

## Features

- **Auth** — OAuth (Google + GitHub), magic link, email OTP. No passwords.
- **Sessions** — Secure server-side sessions in Dragonfly with httponly cookies
- **Conversation store** — Message-level conversation history in Dragonfly sorted sets with TTL-based sliding window
- **Context management** — Automatic conversation history injection into LLM calls with token/message budget trimming
- **Permissions** — Fine-grained RBAC. Each endpoint declares required permissions via `Depends(require("read:reports"))`
- **MongoDB** — Async via Motor. Every document carries audited `_meta` (created/updated/deleted timestamps, user tracking, change deltas)
- **User scoping** — All queries automatically filtered to the logged-in user
- **Soft deletes** — Deletes set `_meta.deleted_at`, excluded from queries automatically
- **Pydantic everywhere** — Validated request/response schemas, domain models, and Meta audit trail
- **Auto-discovered routes** — Drop a file in `routes/` with a `router`, it's registered automatically
- **LLM integration** — LangChain connector with structured Pydantic output (Gemini, Anthropic, OpenAI)
- **Configurable guardrails** — LLM-based input/output safety checks with YAML-driven rules, per-check toggles, and optional cheaper guardrail model
- **Tool registry** — Decorator-based `@harness_tool` with auto-discovery, permission filtering, and gated write support
- **Agent loop** — Multi-turn tool-calling execution with SSE streaming, input/output guardrails, and conversation history
- **Database tools** — LLM can query/mutate MongoDB through user-scoped, filter-sanitised, collection-allowlisted tools

## Project Structure

```
main.py                  # FastAPI app, lifespan, middleware
config.py                # Env-backed Config class
ai/
  llm.py                 # LangChain LLM connector (Gemini, Anthropic, OpenAI)
  guardrails.py          # Configurable input/output safety checks
  context.py             # Conversation context builder for LLM calls
  agent.py               # Multi-turn tool-calling execution loop (SSE)
  tools/
    base.py              # @harness_tool decorator, registry, permission filtering
    db.py                # Database query/mutate tools with filter sanitisation
auth/
  oauth.py               # Google + GitHub OAuth via Authlib
  passwordless.py         # Magic link + OTP logic
  email.py               # Email sending abstraction (Resend)
  permissions.py          # require() dependency for RBAC
db/
  connect.py             # DbConnect class + universal query actualiser
  meta.py                # Meta builder functions (create/update/delete)
  mongo.py               # Motor client lifecycle
  objectid.py            # ObjectId serialisation (single source of truth)
  sessions.py            # Dragonfly session CRUD
  conversations.py       # Conversation message store (sorted sets)
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
tests/                   # Tests across 12 files
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

### Auth
- `GET /auth/google` — Google OAuth login
- `GET /auth/github` — GitHub OAuth login
- `POST /auth/magic-link` — `{"email": "..."}` → sends magic link
- `POST /auth/otp/send` — `{"email": "..."}` → sends 6-digit code
- `POST /auth/otp/verify` — `{"email": "...", "code": "123456"}` → creates session

### Sessions
- `GET /sessions/me` — current user info (requires `read:self`)
- `POST /sessions/logout` — destroys session

### Agent
- `POST /agent/chat` — SSE-streamed agent conversation with tool calling (requires `agent:use`)
- `POST /agent/confirm/{pending_id}` — approve or deny a gated tool call (requires `agent:use`)

## Adding New Endpoints

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

## Adding New Models

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

Set env vars in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GUARDRAILS_ENABLED` | `true` | Global kill switch |
| `GUARDRAILS_CONFIG_PATH` | _(empty)_ | Path to YAML rules file |
| `GUARDRAILS_MODEL` | _(empty)_ | Cheaper model for safety checks |

For custom rules, copy `guardrails.example.yaml` and point to it:

```bash
cp guardrails.example.yaml guardrails.yaml
# Edit guardrails.yaml with your rules
# Set GUARDRAILS_CONFIG_PATH=guardrails.yaml in .env
```

The YAML file lets you enable/disable input and output checks independently, replace the default rejection categories, and add custom rules per deployment.

## Conversation Store & Context Management

### How it works

Conversations are stored in Dragonfly sorted sets, keyed by conversation ID with Unix-millisecond scores for ordering. Each message is a full JSON document (role, content, timestamp, token usage, model, metadata). All keys use a sliding TTL — activity refreshes expiry.

Key schema:
- `conv:{id}` — sorted set of messages
- `conv_meta:{id}` — hash with `user_id`, `session_id`, `created_at`
- `user_convs:{user_id}` — sorted set index of conversations per user

### Storing messages

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

### Context-aware LLM calls

Pass `conversation_id` to `invoke_structured` or `invoke_guarded` and prior conversation history is automatically prepended to the LLM prompt:

```python
from ai.llm import invoke_structured

result = await invoke_structured(
    "Follow-up question here",
    MySchema,
    conversation_id=conv_id,
)
```

The context builder trims from oldest messages first to fit within the configured token and message budgets.

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DRAGONFLY_URL` | `redis://localhost:6379/0` | Dragonfly connection (falls back to `REDIS_URL`) |
| `HARNESS_CONVERSATION_STORE_URL` | _(same as DRAGONFLY_URL)_ | Separate DB for conversations |
| `HARNESS_CONVERSATION_TTL` | `1800` | Sliding TTL in seconds for conversation keys |
| `CONTEXT_MAX_MESSAGES` | `50` | Max messages to include as context |
| `CONTEXT_MAX_TOKENS` | `4000` | Approximate token budget for context |

## Tool Registry & Agent

### How the tool-calling loop works

The agent endpoint (`POST /agent/chat`) streams events via SSE as the LLM reasons and calls tools:

1. **Input guardrail** checks the user prompt
2. **Conversation history** is loaded and injected as context
3. The LLM is bound to the user's permitted tools via `bind_tools`
4. **Loop**: LLM responds → if it requests tool calls, execute them, send results back, repeat
5. **Output guardrail** checks the final response
6. All messages (user, assistant, tool) are stored in the conversation

### Adding a custom tool

Drop a file in `ai/tools/` — it's auto-discovered:

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

The decorator handles LangChain schema generation from type hints, tool registration, and permission metadata.

### Built-in database tools

| Tool | Operation | Gated | Permission |
|------|-----------|-------|------------|
| `db_query` | `find`, `find_one` | No | `db:read` |
| `db_mutate` | `insert_one`, `update_one`, `delete_one` | Yes | `db:write` |

**Security**: All database tools are user-scoped through `DbConnect`, with `_meta` keys recursively stripped from LLM-generated filters to prevent scope escape. An optional collection allowlist restricts which collections the LLM can access.

### Gated writes

Write operations (`db_mutate`) don't execute immediately. The agent yields a `tool_pending` SSE event with the proposed action. The client must call `POST /agent/confirm/{pending_id}` to approve or deny before execution continues.

### SSE event types

| Event | Description |
|-------|-------------|
| `thinking` | Agent is processing (includes iteration number) |
| `tool_call` | LLM requested a tool call (includes tool name and args) |
| `tool_result` | Tool executed successfully (includes result) |
| `tool_pending` | Gated tool awaiting confirmation (includes pending_id) |
| `tool_confirmed` | User approved a gated tool |
| `tool_denied` | User denied a gated tool |
| `response` | Final agent response |
| `blocked` | Guardrail blocked input or output |
| `error` | Error occurred (tool not found, max iterations, etc.) |

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MAX_ITERATIONS` | `10` | Max tool-calling loop iterations |
| `AGENT_ALLOWED_COLLECTIONS` | _(empty = all)_ | Comma-separated collection allowlist |
| `AGENT_SYSTEM_PROMPT` | _(default prompt)_ | System prompt for the agent |
