from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from ai.agent import invoke_agent
from ai.pending import resolve_pending, _pending_futures
from ai.tools.base import get_tool, get_tools, harness_tool, _registry, clear_registry
from config import Config
from schemas.agent import AgentEvent


@pytest.fixture(autouse=True)
def _reset_config(monkeypatch):
    monkeypatch.setattr(Config, "GUARDRAILS_ENABLED", False)
    monkeypatch.setattr(Config, "AGENT_SYSTEM_PROMPT", "")
    monkeypatch.setattr(Config, "AGENT_MAX_ITERATIONS", 10)
    monkeypatch.setattr(Config, "AGENT_ALLOWED_COLLECTIONS", [])


@pytest.fixture(autouse=True)
def _clean_pending():
    _pending_futures.clear()
    yield
    _pending_futures.clear()


def _mock_llm_responses(*responses):
    mock_llm = MagicMock()
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)
    mock_llm.ainvoke = AsyncMock(side_effect=list(responses))
    return mock_llm


async def _collect_events(gen) -> list[AgentEvent]:
    events = []
    async for event in gen:
        events.append(event)
    return events


async def test_agent_direct_response():
    ai_msg = AIMessage(content="Hello!")
    mock_llm = _mock_llm_responses(ai_msg)

    with patch("ai.agent.get_llm", return_value=mock_llm):
        events = await _collect_events(invoke_agent(
            "Hi",
            tools=[],
            user_id="user-1",
            session_id="sess-1",
        ))

    types = [e.type for e in events]
    assert "thinking" in types
    assert "response" in types
    assert events[-1].data["content"] == "Hello!"


async def test_agent_tool_call_and_response():
    @harness_tool("test_add", description="Add two numbers")
    async def test_add(a: int, b: int) -> str:
        return str(a + b)

    try:
        tool_call_msg = AIMessage(content="", tool_calls=[{"name": "test_add", "args": {"a": 2, "b": 3}, "id": "call-1"}])
        final_msg = AIMessage(content="The answer is 5")
        mock_llm = _mock_llm_responses(tool_call_msg, final_msg)

        with patch("ai.agent.get_llm", return_value=mock_llm):
            events = await _collect_events(invoke_agent(
                "What is 2+3?",
                tools=[test_add],
                user_id="user-1",
                session_id="sess-1",
            ))

        types = [e.type for e in events]
        assert "tool_call" in types
        assert "tool_result" in types
        assert "response" in types

        tool_result_event = next(e for e in events if e.type == "tool_result")
        assert "5" in tool_result_event.data["result"]
    finally:
        _registry.pop("test_add", None)


async def test_agent_max_iterations():
    tool_call_msg = AIMessage(content="", tool_calls=[{"name": "db_read", "args": {"collection": "items", "pipeline": []}, "id": "call-1"}])
    mock_llm = _mock_llm_responses(*([tool_call_msg] * 5))

    from ai.tools.db import set_db_context
    from db.connect import DbConnect
    from db.mongo import get_mongo_db
    set_db_context(DbConnect("test_user", get_mongo_db()))

    with patch("ai.agent.get_llm", return_value=mock_llm):
        events = await _collect_events(invoke_agent(
            "Loop forever",
            tools=get_tools(permissions=["db:read"]),
            user_id="user-1",
            session_id="sess-1",
            max_iterations=3,
        ))

    types = [e.type for e in events]
    assert "error" in types
    assert "maximum iterations" in events[-1].data.get("error", "")


async def test_agent_input_eval_blocks():
    from schemas.ai import EvalResult, EvalStage, EvalVerdict

    async def mock_eval(stage, ctx):
        if stage == EvalStage.INPUT:
            return EvalResult(verdict=EvalVerdict.FAIL, reason="Dangerous input")
        return EvalResult(verdict=EvalVerdict.PASS)

    with patch("ai.agent.run_eval_stage", AsyncMock(side_effect=mock_eval)):
        events = await _collect_events(invoke_agent(
            "bad prompt",
            tools=[],
            user_id="user-1",
            session_id="sess-1",
        ))

    assert events[0].type == "blocked"
    assert events[0].data["stage"] == "input"


async def test_agent_output_eval_blocks():
    from schemas.ai import EvalResult, EvalStage, EvalVerdict

    async def mock_eval(stage, ctx):
        if stage == EvalStage.OUTPUT:
            return EvalResult(verdict=EvalVerdict.FAIL, reason="Bad output")
        return EvalResult(verdict=EvalVerdict.PASS)

    ai_msg = AIMessage(content="something harmful")
    mock_llm = _mock_llm_responses(ai_msg)

    with (
        patch("ai.agent.get_llm", return_value=mock_llm),
        patch("ai.agent.run_eval_stage", AsyncMock(side_effect=mock_eval)),
    ):
        events = await _collect_events(invoke_agent(
            "prompt",
            tools=[],
            user_id="user-1",
            session_id="sess-1",
        ))

    assert any(e.type == "blocked" and e.data["stage"] == "output" for e in events)


async def test_agent_tool_result_eval_veto():
    from schemas.ai import EvalResult, EvalStage, EvalVerdict

    @harness_tool("test_vetoed", description="Will be vetoed")
    async def test_vetoed(x: str) -> str:
        return f"result: {x}"

    try:
        call_count = 0

        async def mock_eval(stage, ctx):
            nonlocal call_count
            if stage == EvalStage.TOOL_RESULT:
                call_count += 1
                return EvalResult(verdict=EvalVerdict.FAIL, reason="Dangerous result")
            return EvalResult(verdict=EvalVerdict.PASS)

        tool_call_msg = AIMessage(content="", tool_calls=[
            {"name": "test_vetoed", "args": {"x": "hello"}, "id": "call-v1"},
        ])
        final_msg = AIMessage(content="OK")
        mock_llm = _mock_llm_responses(tool_call_msg, final_msg)

        with (
            patch("ai.agent.get_llm", return_value=mock_llm),
            patch("ai.agent.run_eval_stage", AsyncMock(side_effect=mock_eval)),
        ):
            events = await _collect_events(invoke_agent(
                "Do vetoed thing",
                tools=[test_vetoed],
                user_id="user-1",
                session_id="sess-1",
            ))

        types = [e.type for e in events]
        assert "blocked" in types
        blocked_event = next(e for e in events if e.type == "blocked")
        assert blocked_event.data["stage"] == "tool_result"
    finally:
        _registry.pop("test_vetoed", None)


async def test_agent_tool_result_eval_retry():
    from schemas.ai import EvalResult, EvalStage, EvalVerdict

    @harness_tool("test_retry", description="Will trigger retry")
    async def test_retry(x: str) -> str:
        return f"result: {x}"

    try:
        async def mock_eval(stage, ctx):
            if stage == EvalStage.TOOL_RESULT:
                return EvalResult(
                    verdict=EvalVerdict.RETRY,
                    reason="Try again",
                    corrective_context="Use a different approach",
                )
            return EvalResult(verdict=EvalVerdict.PASS)

        tool_call_msg = AIMessage(content="", tool_calls=[
            {"name": "test_retry", "args": {"x": "hello"}, "id": "call-r1"},
        ])
        final_msg = AIMessage(content="Done after retry")
        mock_llm = _mock_llm_responses(tool_call_msg, final_msg)

        with (
            patch("ai.agent.get_llm", return_value=mock_llm),
            patch("ai.agent.run_eval_stage", AsyncMock(side_effect=mock_eval)),
        ):
            events = await _collect_events(invoke_agent(
                "Do retry thing",
                tools=[test_retry],
                user_id="user-1",
                session_id="sess-1",
            ))

        types = [e.type for e in events]
        assert "eval_retry" in types
        retry_event = next(e for e in events if e.type == "eval_retry")
        assert retry_event.data["stage"] == "tool_result"
    finally:
        _registry.pop("test_retry", None)


async def test_agent_gated_tool_pending():
    @harness_tool("test_gated", description="Gated action", gated=True)
    async def test_gated(x: str) -> str:
        return f"executed: {x}"

    try:
        tool_call_msg = AIMessage(content="", tool_calls=[{"name": "test_gated", "args": {"x": "hello"}, "id": "call-g1"}])
        final_msg = AIMessage(content="Done")
        mock_llm = _mock_llm_responses(tool_call_msg, final_msg)

        async def approve_after_yield():
            await asyncio.sleep(0.05)
            for pid in list(_pending_futures.keys()):
                await resolve_pending(pid, True)

        with patch("ai.agent.get_llm", return_value=mock_llm):
            task = asyncio.create_task(approve_after_yield())
            events = await _collect_events(invoke_agent(
                "Do gated thing",
                tools=[test_gated],
                user_id="user-1",
                session_id="sess-1",
            ))
            await task

        types = [e.type for e in events]
        assert "tool_pending" in types
        assert "tool_confirmed" in types
        assert "tool_result" in types
    finally:
        _registry.pop("test_gated", None)


async def test_agent_gated_tool_denied():
    @harness_tool("test_deny", description="Will be denied", gated=True)
    async def test_deny(x: str) -> str:
        return f"executed: {x}"

    try:
        tool_call_msg = AIMessage(content="", tool_calls=[{"name": "test_deny", "args": {"x": "hello"}, "id": "call-d1"}])
        final_msg = AIMessage(content="OK denied")
        mock_llm = _mock_llm_responses(tool_call_msg, final_msg)

        async def deny_after_yield():
            await asyncio.sleep(0.05)
            for pid in list(_pending_futures.keys()):
                await resolve_pending(pid, False)

        with patch("ai.agent.get_llm", return_value=mock_llm):
            task = asyncio.create_task(deny_after_yield())
            events = await _collect_events(invoke_agent(
                "Do denied thing",
                tools=[test_deny],
                user_id="user-1",
                session_id="sess-1",
            ))
            await task

        types = [e.type for e in events]
        assert "tool_pending" in types
        assert "tool_denied" in types
    finally:
        _registry.pop("test_deny", None)


async def test_agent_unknown_tool():
    tool_call_msg = AIMessage(content="", tool_calls=[{"name": "nonexistent_tool", "args": {}, "id": "call-u1"}])
    final_msg = AIMessage(content="Sorry")
    mock_llm = _mock_llm_responses(tool_call_msg, final_msg)

    with patch("ai.agent.get_llm", return_value=mock_llm):
        events = await _collect_events(invoke_agent(
            "Use missing tool",
            tools=[],
            user_id="user-1",
            session_id="sess-1",
        ))

    assert any(e.type == "error" and "not found" in e.data.get("error", "") for e in events)


async def test_agent_stores_conversation(monkeypatch):
    from db.conversations import create_conversation, get_messages

    cid = await create_conversation("user-1", "sess-1")

    ai_msg = AIMessage(content="Stored response")
    mock_llm = _mock_llm_responses(ai_msg)

    with patch("ai.agent.get_llm", return_value=mock_llm):
        events = await _collect_events(invoke_agent(
            "Store this",
            tools=[],
            user_id="user-1",
            session_id="sess-1",
            conversation_id=cid,
        ))

    msgs = await get_messages(cid)
    roles = [m.role.value for m in msgs]
    assert "user" in roles
    assert "assistant" in roles


async def test_resolve_pending_unknown_id():
    from errors import PendingActionNotFoundError
    with pytest.raises(PendingActionNotFoundError):
        await resolve_pending("nonexistent", True)


async def test_resolve_pending_idempotent():
    from ai.pending import create_pending

    future = await create_pending(
        "test-idempotent",
        tool_name="test_tool",
        tool_args={"x": 1},
        user_id="user-1",
    )

    action = await resolve_pending("test-idempotent", True)
    assert action == "approved"
    assert future.result() is True

    from errors import PendingActionNotFoundError
    with pytest.raises(PendingActionNotFoundError):
        await resolve_pending("test-idempotent", True)
