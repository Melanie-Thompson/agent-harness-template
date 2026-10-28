from __future__ import annotations

from unittest.mock import MagicMock

from routes.agent import _merge_tools


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
