from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from errors import McpConnectionError
from schemas.mcp import McpServerConfig, McpToolPolicy, McpTransport


@pytest.fixture(autouse=True)
def _reset_mcp_state():
    from ai import mcp
    orig_client = mcp._static_client
    mcp._static_client = None
    yield
    mcp._static_client = orig_client


def _mock_tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.metadata = {}
    return tool


async def test_init_mcp_no_servers(monkeypatch):
    from ai.mcp import get_mcp_tools, init_mcp

    monkeypatch.setattr("ai.mcp.Config.MCP_SERVERS", "[]")
    await init_mcp()
    assert await get_mcp_tools() == []


async def test_init_mcp_with_servers(monkeypatch):
    from ai.mcp import get_mcp_tools, init_mcp

    server_json = json.dumps([{"name": "test", "transport": "sse", "url": "http://localhost:9000/sse"}])
    monkeypatch.setattr("ai.mcp.Config.MCP_SERVERS", server_json)
    monkeypatch.setattr("ai.mcp.Config.MCP_GATED_TOOLS", [])
    monkeypatch.setattr("ai.mcp.Config.MCP_TOOL_PERMISSIONS", "{}")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[_mock_tool("remote_tool")])

    with patch("ai.mcp.MultiServerMCPClient", return_value=mock_client) as mock_cls:
        await init_mcp()
        mock_cls.assert_called_once()

    tools = await get_mcp_tools()
    assert len(tools) == 1
    assert tools[0].name == "remote_tool"


async def test_transport_dispatch_covers_all():
    from ai.mcp import _TRANSPORT_CONFIG

    assert set(_TRANSPORT_CONFIG.keys()) == set(McpTransport)


async def test_connect_request_mcp_returns_tools(monkeypatch):
    monkeypatch.setattr("ai.mcp.Config.MCP_GATED_TOOLS", [])
    monkeypatch.setattr("ai.mcp.Config.MCP_TOOL_PERMISSIONS", "{}")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(return_value=[_mock_tool("req_tool")])

    with patch("ai.mcp.MultiServerMCPClient", return_value=mock_client):
        from ai.mcp import connect_request_mcp

        servers = [McpServerConfig(name="req", transport=McpTransport.SSE, url="http://localhost:9000/sse")]
        tools = await connect_request_mcp(servers)

    assert len(tools) == 1
    assert tools[0].name == "req_tool"


async def test_tool_metadata_injection(monkeypatch):
    monkeypatch.setattr("ai.mcp.Config.MCP_GATED_TOOLS", ["gated_tool"])
    monkeypatch.setattr("ai.mcp.Config.MCP_TOOL_PERMISSIONS", json.dumps({"perm_tool": ["db:read"]}))

    mock_client = MagicMock()
    gated = _mock_tool("gated_tool")
    perm = _mock_tool("perm_tool")
    plain = _mock_tool("plain_tool")
    mock_client.get_tools = AsyncMock(return_value=[gated, perm, plain])

    server_json = json.dumps([{"name": "test", "transport": "stdio", "command": "echo"}])
    monkeypatch.setattr("ai.mcp.Config.MCP_SERVERS", server_json)

    with patch("ai.mcp.MultiServerMCPClient", return_value=mock_client):
        from ai.mcp import init_mcp, get_mcp_tools
        await init_mcp()

    tools = await get_mcp_tools()
    tool_map = {t.name: t for t in tools}

    assert tool_map["gated_tool"].metadata["gated"] is True
    assert tool_map["plain_tool"].metadata["gated"] is False
    assert tool_map["perm_tool"].metadata["permissions"] == ["db:read"]
    assert tool_map["plain_tool"].metadata["permissions"] == []


async def test_mcp_connection_error(monkeypatch):
    monkeypatch.setattr("ai.mcp.Config.MCP_GATED_TOOLS", [])
    monkeypatch.setattr("ai.mcp.Config.MCP_TOOL_PERMISSIONS", "{}")

    mock_client = MagicMock()
    mock_client.get_tools = AsyncMock(side_effect=ConnectionError("refused"))

    with patch("ai.mcp.MultiServerMCPClient", return_value=mock_client):
        from ai.mcp import connect_request_mcp

        servers = [McpServerConfig(name="broken", transport=McpTransport.SSE, url="http://localhost:1/sse")]
        with pytest.raises(McpConnectionError, match="connection failed"):
            await connect_request_mcp(servers)


async def test_close_mcp_idempotent():
    from ai import mcp
    from ai.mcp import close_mcp

    mcp._static_client = MagicMock()

    await close_mcp()
    assert mcp._static_client is None

    await close_mcp()


async def test_build_server_params_all_transports():
    from ai.mcp import _build_server_params

    servers = [
        McpServerConfig(name="s1", transport=McpTransport.STDIO, command="python", args=["-m", "server"]),
        McpServerConfig(name="s2", transport=McpTransport.SSE, url="http://localhost:9000/sse"),
        McpServerConfig(name="s3", transport=McpTransport.STREAMABLE_HTTP, url="http://localhost:9001/mcp"),
    ]
    params = _build_server_params(servers)

    assert params["s1"]["transport"] == "stdio"
    assert params["s1"]["command"] == "python"
    assert params["s1"]["args"] == ["-m", "server"]

    assert params["s2"]["transport"] == "sse"
    assert params["s2"]["url"] == "http://localhost:9000/sse"

    assert params["s3"]["transport"] == "streamable_http"
    assert params["s3"]["url"] == "http://localhost:9001/mcp"
