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
    orig_tools = mcp._static_tools
    mcp._static_client = None
    mcp._static_tools = []
    yield
    mcp._static_client = orig_client
    mcp._static_tools = orig_tools


def _mock_tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
    tool.metadata = {}
    return tool


async def test_init_mcp_no_servers(monkeypatch):
    from ai.mcp import get_mcp_tools, init_mcp

    monkeypatch.setattr("ai.mcp.Config.MCP_SERVERS", "[]")
    await init_mcp()
    assert get_mcp_tools() == []


async def test_init_mcp_with_servers(monkeypatch):
    from ai.mcp import get_mcp_tools, init_mcp

    server_json = json.dumps([{"name": "test", "transport": "sse", "url": "http://localhost:9000/sse"}])
    monkeypatch.setattr("ai.mcp.Config.MCP_SERVERS", server_json)
    monkeypatch.setattr("ai.mcp.Config.MCP_GATED_TOOLS", [])
    monkeypatch.setattr("ai.mcp.Config.MCP_TOOL_PERMISSIONS", "{}")

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock()
    mock_client.get_tools = MagicMock(return_value=[_mock_tool("remote_tool")])

    with patch("ai.mcp.MultiServerMCPClient", return_value=mock_client) as mock_cls:
        await init_mcp()
        mock_cls.assert_called_once()

    tools = get_mcp_tools()
    assert len(tools) == 1
    assert tools[0].name == "remote_tool"


async def test_transport_dispatch_covers_all():
    from ai.mcp import _TRANSPORT_CONFIG

    assert set(_TRANSPORT_CONFIG.keys()) == set(McpTransport)


async def test_connect_request_mcp_returns_tools_and_cleanup(monkeypatch):
    monkeypatch.setattr("ai.mcp.Config.MCP_GATED_TOOLS", [])
    monkeypatch.setattr("ai.mcp.Config.MCP_TOOL_PERMISSIONS", "{}")

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock()
    mock_client.get_tools = MagicMock(return_value=[_mock_tool("req_tool")])

    with patch("ai.mcp.MultiServerMCPClient", return_value=mock_client):
        from ai.mcp import connect_request_mcp

        servers = [McpServerConfig(name="req", transport=McpTransport.SSE, url="http://localhost:9000/sse")]
        tools, cleanup = await connect_request_mcp(servers)

    assert len(tools) == 1
    assert tools[0].name == "req_tool"

    await cleanup()
    mock_client.__aexit__.assert_called_once()


async def test_tool_metadata_injection(monkeypatch):
    monkeypatch.setattr("ai.mcp.Config.MCP_GATED_TOOLS", ["gated_tool"])
    monkeypatch.setattr("ai.mcp.Config.MCP_TOOL_PERMISSIONS", json.dumps({"perm_tool": ["db:read"]}))

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock()

    gated = _mock_tool("gated_tool")
    perm = _mock_tool("perm_tool")
    plain = _mock_tool("plain_tool")
    mock_client.get_tools = MagicMock(return_value=[gated, perm, plain])

    server_json = json.dumps([{"name": "test", "transport": "stdio", "command": "echo"}])
    monkeypatch.setattr("ai.mcp.Config.MCP_SERVERS", server_json)

    with patch("ai.mcp.MultiServerMCPClient", return_value=mock_client):
        from ai.mcp import init_mcp, get_mcp_tools
        await init_mcp()

    tools = get_mcp_tools()
    tool_map = {t.name: t for t in tools}

    assert tool_map["gated_tool"].metadata["gated"] is True
    assert tool_map["plain_tool"].metadata["gated"] is False
    assert tool_map["perm_tool"].metadata["permissions"] == ["db:read"]
    assert tool_map["plain_tool"].metadata["permissions"] == []


async def test_mcp_connection_error(monkeypatch):
    server_json = json.dumps([{"name": "broken", "transport": "sse", "url": "http://localhost:1/sse"}])
    monkeypatch.setattr("ai.mcp.Config.MCP_SERVERS", server_json)
    monkeypatch.setattr("ai.mcp.Config.MCP_GATED_TOOLS", [])
    monkeypatch.setattr("ai.mcp.Config.MCP_TOOL_PERMISSIONS", "{}")

    mock_client = MagicMock()
    mock_client.__aenter__ = AsyncMock(side_effect=ConnectionError("refused"))
    mock_client.__aexit__ = AsyncMock()

    with patch("ai.mcp.MultiServerMCPClient", return_value=mock_client):
        from ai.mcp import init_mcp

        with pytest.raises(McpConnectionError, match="connection failed"):
            await init_mcp()


async def test_close_mcp_idempotent():
    from ai import mcp
    from ai.mcp import close_mcp

    mock_client = MagicMock()
    mock_client.__aexit__ = AsyncMock()
    mcp._static_client = mock_client
    mcp._static_tools = [_mock_tool("x")]

    await close_mcp()
    assert mcp._static_client is None
    assert mcp._static_tools == []
    mock_client.__aexit__.assert_called_once()

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
