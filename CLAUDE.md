# Basic Harness — Code Architecture Rules

## Project

Reusable FastAPI template for AI harness projects. Python 3.13+, async everywhere, Pydantic validation at all boundaries.

## Stack

- **FastAPI** with async lifespan, CORS, SessionMiddleware
- **MongoDB** via Motor (async) — all documents inherit `MongoDocument` for `_meta` audit trail
- **Dragonfly** (Redis wire-compatible) — sessions, OTP, conversation store
- **LangChain** — LLM connector (Gemini, Anthropic, OpenAI) with structured output, tool calling
- **pytest-asyncio** with `asyncio_mode = "auto"`, pytest-cov for coverage

## Run & Test

```bash
uv run uvicorn main:app --reload     # dev server on :8000
uv run pytest tests/ -v              # all tests (needs local Dragonfly + MongoDB)
```

## Architecture Rules

### No long if/elif chains — use dispatch mappings

Use O(1) dispatch dicts or registry lookups instead of conditional branches for operation selection, tool routing, or action handling.

```python
# Bad
if operation == "find": ...
elif operation == "insert": ...

# Good
DISPATCH = {Operation.FIND: db_find, Operation.INSERT: db_insert}
handler = DISPATCH.get(operation)
if not handler:
    raise ValueError(f"Unsupported: {operation}")
return await handler(...)
```

### Structured exception hierarchy

Never raise bare `Exception` or return raw error dicts. Use typed exceptions with status codes. Catch domain exceptions at the FastAPI exception handler level.

### Declarative tool registry

Every LLM tool uses `@harness_tool` from `ai/tools/base.py` with Pydantic-typed args. Tools live in `ai/tools/` and are auto-discovered. Never embed tool functions in route handlers.

```python
@harness_tool("tool_name", description="...", permissions=["scope:action"], gated=False)
async def tool_name(arg: str) -> str: ...
```

### Pydantic & async first

- All inputs/outputs pass through Pydantic models
- All I/O is async (Motor, Dragonfly, HTTP clients, LLM calls)
- Config via `Config` class in `config.py` with `os.getenv` pattern

### User scoping & audit

- All MongoDB documents inherit `MongoDocument` for automatic `_meta` tracking
- All queries go through `DbConnect` which enforces `_meta.user_id` scoping
- LLM-generated filters are sanitised (strip `_meta` keys) before reaching `DbConnect`

### Auto-discovery patterns

- **Routes**: drop a file in `routes/` with a `router` — auto-registered via `routes/__init__.py`
- **Tools**: drop a file in `ai/tools/` with `@harness_tool` decorators — auto-registered via `ai/tools/__init__.py`

### Singleton lifecycle

Database connections follow `init_*()` / `get_*()` / `close_*()` pattern. Init in `main.py` lifespan, close on shutdown.

### Testing conventions

- `pytest-asyncio` with auto mode — no `@pytest.mark.asyncio` needed
- Mock LLMs with `MagicMock` + `AsyncMock`, inject via `model=` parameter
- Real MongoDB/Dragonfly for integration tests (test conftest handles init/cleanup)
- DB index isolation: sessions on DB 1, conversations on DB 2

## Key File Locations

| What | Where |
|------|-------|
| Config | `config.py` |
| LLM connector | `ai/llm.py` |
| Guardrails | `ai/guardrails.py` |
| Tool registry | `ai/tools/base.py` |
| DB tools | `ai/tools/db.py` |
| Agent loop | `ai/agent.py` |
| Context builder | `ai/context.py` |
| DB scoped queries | `db/connect.py` |
| Conversation store | `db/conversations.py` |
| Auth/permissions | `auth/permissions.py` |
