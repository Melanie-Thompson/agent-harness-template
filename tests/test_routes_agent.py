from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from routes.agent import _merge_tools, _sse_with_cleanup
from schemas.agent import AgentEvent


def _mock_tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    return tool


def test_merge_tools_precedence():
    builtin = [_mock_tool("shared"), _mock_tool("builtin_only")]
    static_mcp = [_mock_tool("shared"), _mock_tool("mcp_only")]
    request_mcp = [_mock_tool("shared"), _mock_tool("req_only")]

    merged = _merge_tools(builtin, static_mcp, request_mcp)
    merged_map = {t.name: t for t in merged}

    assert set(merged_map.keys()) == {"shared", "builtin_only", "mcp_only", "req_only"}
    assert merged_map["shared"] is request_mcp[0]


def test_merge_tools_empty_inputs():
    builtin = [_mock_tool("a")]
    merged = _merge_tools(builtin, [], [])
    assert len(merged) == 1
    assert merged[0].name == "a"


async def test_sse_with_cleanup_calls_cleanup():
    events_list = [
        AgentEvent(type="thinking", data={"iteration": 1}),
        AgentEvent(type="response", data={"content": "done"}),
    ]

    async def mock_events():
        for e in events_list:
            yield e

    cleanup = AsyncMock()
    collected = []
    async for chunk in _sse_with_cleanup(mock_events(), cleanup):
        collected.append(chunk)

    assert len(collected) == 2
    cleanup.assert_awaited_once()


async def test_sse_with_cleanup_calls_cleanup_on_error():
    async def failing_events():
        yield AgentEvent(type="thinking", data={"iteration": 1})
        raise RuntimeError("boom")

    cleanup = AsyncMock()
    collected = []
    try:
        async for chunk in _sse_with_cleanup(failing_events(), cleanup):
            collected.append(chunk)
    except RuntimeError:
        pass

    cleanup.assert_awaited_once()


async def test_sse_with_cleanup_no_cleanup():
    async def mock_events():
        yield AgentEvent(type="response", data={"content": "ok"})

    collected = []
    async for chunk in _sse_with_cleanup(mock_events(), None):
        collected.append(chunk)

    assert len(collected) == 1
