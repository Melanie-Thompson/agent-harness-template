from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import Config
from errors import McpConnectionError
from schemas.mcp import McpServerConfig, McpToolPolicy, McpTransport

logger = logging.getLogger(__name__)

_static_client: Any | None = None
_static_tools: list[BaseTool] = []


def _stdio_config(server: McpServerConfig) -> dict:
    return {
        "transport": "stdio",
        "command": server.command,
        "args": server.args,
        "env": server.env or None,
    }


def _sse_config(server: McpServerConfig) -> dict:
    return {
        "transport": "sse",
        "url": server.url,
        "headers": server.headers or None,
    }


def _streamable_config(server: McpServerConfig) -> dict:
    return {
        "transport": "streamable_http",
        "url": server.url,
        "headers": server.headers or None,
    }


_TRANSPORT_CONFIG: dict[McpTransport, Callable[[McpServerConfig], dict]] = {
    McpTransport.STDIO: _stdio_config,
    McpTransport.SSE: _sse_config,
    McpTransport.STREAMABLE_HTTP: _streamable_config,
}


def _build_server_params(servers: list[McpServerConfig]) -> dict[str, dict]:
    params: dict[str, dict] = {}
    for server in servers:
        builder = _TRANSPORT_CONFIG[server.transport]
        params[server.name] = builder(server)
    return params


def _load_tool_policy() -> McpToolPolicy:
    raw_perms = json.loads(Config.MCP_TOOL_PERMISSIONS)
    return McpToolPolicy(
        gated_tools=set(Config.MCP_GATED_TOOLS),
        tool_permissions=raw_perms,
    )


def _inject_metadata(tools: list[BaseTool], policy: McpToolPolicy) -> list[BaseTool]:
    for tool in tools:
        if not hasattr(tool, "metadata") or tool.metadata is None:
            tool.metadata = {}
        tool.metadata["gated"] = tool.name in policy.gated_tools
        tool.metadata["permissions"] = policy.tool_permissions.get(tool.name, [])
    return tools


async def init_mcp() -> None:
    global _static_client, _static_tools

    server_configs = [McpServerConfig(**s) for s in json.loads(Config.MCP_SERVERS)]
    if not server_configs:
        _static_client = None
        _static_tools = []
        return

    params = _build_server_params(server_configs)
    client = MultiServerMCPClient(params)
    try:
        await client.__aenter__()
    except Exception as exc:
        raise McpConnectionError("static", str(exc)) from exc

    policy = _load_tool_policy()
    _static_client = client
    _static_tools = _inject_metadata(client.get_tools(), policy)
    logger.info("MCP init: %d tools from %d servers", len(_static_tools), len(server_configs))


def get_mcp_tools() -> list[BaseTool]:
    return list(_static_tools)


async def connect_request_mcp(
    servers: list[McpServerConfig],
) -> tuple[list[BaseTool], Callable]:
    params = _build_server_params(servers)
    client = MultiServerMCPClient(params)

    try:
        await client.__aenter__()
    except Exception as exc:
        server_names = ", ".join(s.name for s in servers)
        raise McpConnectionError(server_names, str(exc)) from exc

    policy = _load_tool_policy()
    tools = _inject_metadata(client.get_tools(), policy)

    async def cleanup() -> None:
        await client.__aexit__(None, None, None)

    return tools, cleanup


async def close_mcp() -> None:
    global _static_client, _static_tools
    if _static_client is not None:
        await _static_client.__aexit__(None, None, None)
        _static_client = None
    _static_tools = []
