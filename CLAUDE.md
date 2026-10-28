# Agent Harness Template — Code Architecture Rules

## Project

AI agent harness built on FastAPI. LLMs call tools through a multi-turn execution loop with SSE streaming, configurable eval pipeline, gated write confirmation, and user-scoped database access. Python 3.13+, async everywhere, Pydantic validation at all boundaries.

## Stack

- **FastAPI** with async lifespan, CORS, SessionMiddleware
- **MongoDB** via Motor (async) — all documents inherit `MongoDocument` for `_meta` audit trail
- **Dragonfly** (Redis wire-compatible) — sessions, OTP, conversation store, pending action TTL
- **LangChain** — LLM connector (Gemini, Anthropic, OpenAI) with structured output, tool calling
- **MCP** via `langchain-mcp-adapters` — external tool servers (stdio, SSE, streamable HTTP)
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

### Declarative eval registry

Every evaluator uses `@evaluator` from `ai/evals/base.py` with stage bindings. Evaluators live in `ai/evals/` and are auto-discovered. The agent loop calls `run_eval_stage()` at three points: input, tool_result, and output. Existing guardrails are wrapped as built-in evaluators in `ai/evals/builtin.py`.

```python
@evaluator("eval_name", stages=[EvalStage.OUTPUT], description="...", max_retries=1)
async def eval_name(ctx: EvalContext) -> EvalResult: ...
```

Evaluators return `EvalVerdict.PASS`, `FAIL` (blocks), or `RETRY` (with optional `corrective_context`). Pipeline config lives in `GuardrailConfig.eval_pipeline` (YAML or defaults). Empty evaluator list = run all registered for that stage.

### MCP integration

External MCP servers provide tools alongside built-in `@harness_tool` tools. Static default servers are configured via `MCP_SERVERS` (JSON env var) and connected at startup. Per-request servers can be passed in `AgentChatRequest.mcp_servers` and are cleaned up when the SSE stream ends. Tool merge precedence: per-request > static MCP > built-in. MCP tool metadata (`gated`, `permissions`) is injected from `MCP_GATED_TOOLS` and `MCP_TOOL_PERMISSIONS` config. The agent loop resolves tools from the merged `tools` parameter (not the built-in registry), so MCP and built-in tools are handled identically.

### Gated writes & pending actions

Write tools set `gated=True` in `@harness_tool`. The agent loop stores pending action metadata in Dragonfly with TTL (`AGENT_PENDING_TTL`), creates an asyncio.Future for in-process signaling, and yields a `tool_pending` SSE event. The `/agent/confirm/{pending_id}` endpoint resolves or returns 404 (idempotent). Pending action logic lives in `ai/pending.py`.

### Auto-discovery patterns

- **Routes**: drop a file in `routes/` with a `router` — auto-registered via `routes/__init__.py`
- **Tools**: drop a file in `ai/tools/` with `@harness_tool` decorators — auto-registered via `ai/tools/__init__.py`
- **Evaluators**: drop a file in `ai/evals/` with `@evaluator` decorators — auto-registered via `ai/evals/__init__.py`

### Structured logging

`structlog` provides structured JSON logging across the entire codebase. Every public function is instrumented with `@log_call` from `logging_decorator.py`, which logs entry (function name, params), exit (duration_ms), and errors. Critical functions like `invoke_agent` and `run_eval_stage` have additional inline contextual logs.

Logs stream to connected frontends in real-time via `GET /logs/stream` (SSE). The broadcast channel (`logging_broadcast.py`) fans out log events from a structlog processor to per-client async queues with a ring buffer for catch-up.

```python
from logging_decorator import log_call

@log_call
async def my_function(arg: str) -> str: ...
```

Config: `LOG_LEVEL`, `LOG_FORMAT` (json/console), `LOG_BROADCAST_BUFFER_SIZE`.

### Singleton lifecycle

Database, MCP, and logging connections follow `init_*()` / `get_*()` / `close_*()` pattern. Init in `main.py` lifespan, close on shutdown.

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
| Eval registry | `ai/evals/base.py` |
| Built-in evaluators | `ai/evals/builtin.py` |
| Eval pipeline | `ai/evals/pipeline.py` |
| Pending actions | `ai/pending.py` |
| MCP client | `ai/mcp.py` |
| MCP schemas | `schemas/mcp.py` |
| Tool registry | `ai/tools/base.py` |
| DB tools | `ai/tools/db.py` |
| Agent loop | `ai/agent.py` |
| Context builder | `ai/context.py` |
| DB scoped queries | `db/connect.py` |
| Conversation store | `db/conversations.py` |
| Auth/permissions | `auth/permissions.py` |
| Logging config | `logging_config.py` |
| Log broadcast | `logging_broadcast.py` |
| Log decorator | `logging_decorator.py` |
| Log SSE endpoint | `routes/logs.py` |
