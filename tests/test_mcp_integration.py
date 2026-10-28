"""Integration tests for MCP server connections.

These tests connect to real MCP servers via stdio and verify that tool
discovery, metadata injection, and tool invocation work end-to-end through
the harness's MCP layer.

Requires: uvx (mcp-server-git), npx (@modelcontextprotocol/server-memory,
@modelcontextprotocol/server-filesystem).

Run with:  uv run pytest tests/test_mcp_integration.py -v
Skip with: uv run pytest tests/ -v --ignore=tests/test_mcp_integration.py
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from typing import Any

import pytest


def _text_from_result(result: Any) -> str:
    """Extract text from an MCP tool result (str or list of content blocks)."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return " ".join(block.get("text", "") for block in result if isinstance(block, dict))
    return str(result)

from ai.mcp import (
    _build_server_params,
    close_mcp,
    connect_request_mcp,
    get_mcp_tools,
    init_mcp,
)
import ai.mcp as mcp_module
from schemas.mcp import McpServerConfig, McpTransport

_HAS_UVX = shutil.which("uvx") is not None
_HAS_NPX = shutil.which("npx") is not None

pytestmark = pytest.mark.skipif(
    not (_HAS_UVX and _HAS_NPX),
    reason="uvx and npx required for MCP integration tests",
)


@pytest.fixture(autouse=True)
def _reset_mcp_state():
    orig_client = mcp_module._static_client
    mcp_module._static_client = None
    yield
    mcp_module._static_client = orig_client


@pytest.fixture
def repo_path():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="mcp_fs_test_")
    with open(os.path.join(d, "hello.txt"), "w") as f:
        f.write("hello from integration test")
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# mcp-server-git
# ---------------------------------------------------------------------------


class TestMcpServerGit:
    async def test_connect_and_discover_tools(self, repo_path):
        servers = [
            McpServerConfig(
                name="git",
                transport=McpTransport.STDIO,
                command="uvx",
                args=["mcp-server-git", "--repository", repo_path],
            ),
        ]
        tools = await connect_request_mcp(servers)
        assert len(tools) > 0
        tool_names = [t.name for t in tools]
        assert "git_status" in tool_names or "git_log" in tool_names, \
            f"Expected git tools, got: {tool_names}"

    async def test_invoke_git_status(self, repo_path):
        servers = [
            McpServerConfig(
                name="git",
                transport=McpTransport.STDIO,
                command="uvx",
                args=["mcp-server-git", "--repository", repo_path],
            ),
        ]
        tools = await connect_request_mcp(servers)
        tool_map = {t.name: t for t in tools}
        assert "git_status" in tool_map, f"Available tools: {list(tool_map.keys())}"

        result = await tool_map["git_status"].ainvoke({"repo_path": repo_path})
        text = _text_from_result(result)
        assert "branch" in text.lower() or "clean" in text.lower() or "main" in text.lower()

    async def test_invoke_git_log(self, repo_path):
        servers = [
            McpServerConfig(
                name="git",
                transport=McpTransport.STDIO,
                command="uvx",
                args=["mcp-server-git", "--repository", repo_path],
            ),
        ]
        tools = await connect_request_mcp(servers)
        tool_map = {t.name: t for t in tools}
        assert "git_log" in tool_map, f"Available tools: {list(tool_map.keys())}"

        result = await tool_map["git_log"].ainvoke({"repo_path": repo_path, "max_count": 3})
        text = _text_from_result(result)
        assert len(text) > 0

    async def test_metadata_injection(self, repo_path, monkeypatch):
        monkeypatch.setattr("ai.mcp.Config.MCP_GATED_TOOLS", ["git_status"])
        monkeypatch.setattr(
            "ai.mcp.Config.MCP_TOOL_PERMISSIONS",
            json.dumps({"git_log": ["git:read"]}),
        )

        servers = [
            McpServerConfig(
                name="git",
                transport=McpTransport.STDIO,
                command="uvx",
                args=["mcp-server-git", "--repository", repo_path],
            ),
        ]
        tools = await connect_request_mcp(servers)
        tool_map = {t.name: t for t in tools}

        if "git_status" in tool_map:
            assert tool_map["git_status"].metadata["gated"] is True
        if "git_log" in tool_map:
            assert tool_map["git_log"].metadata["permissions"] == ["git:read"]


# ---------------------------------------------------------------------------
# @modelcontextprotocol/server-memory
# ---------------------------------------------------------------------------


class TestMcpServerMemory:
    async def test_connect_and_discover_tools(self):
        servers = [
            McpServerConfig(
                name="memory",
                transport=McpTransport.STDIO,
                command="npx",
                args=["-y", "@modelcontextprotocol/server-memory"],
            ),
        ]
        tools = await connect_request_mcp(servers)
        assert len(tools) > 0
        tool_names = [t.name for t in tools]
        assert any(
            "entity" in n.lower() or "relation" in n.lower() or "create" in n.lower()
            for n in tool_names
        ), f"Expected memory graph tools, got: {tool_names}"

    async def test_create_and_read_entities(self):
        servers = [
            McpServerConfig(
                name="memory",
                transport=McpTransport.STDIO,
                command="npx",
                args=["-y", "@modelcontextprotocol/server-memory"],
            ),
        ]
        tools = await connect_request_mcp(servers)
        tool_map = {t.name: t for t in tools}

        if "create_entities" in tool_map:
            result = await tool_map["create_entities"].ainvoke({
                "entities": [
                    {"name": "TestProject", "entityType": "project", "observations": ["A test project"]},
                ],
            })
            assert len(_text_from_result(result)) > 0

        if "read_graph" in tool_map:
            graph = await tool_map["read_graph"].ainvoke({})
            assert len(_text_from_result(graph)) >= 0


# ---------------------------------------------------------------------------
# @modelcontextprotocol/server-filesystem
# ---------------------------------------------------------------------------


class TestMcpServerFilesystem:
    async def test_connect_and_discover_tools(self, tmp_dir):
        servers = [
            McpServerConfig(
                name="filesystem",
                transport=McpTransport.STDIO,
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", tmp_dir],
            ),
        ]
        tools = await connect_request_mcp(servers)
        assert len(tools) > 0
        tool_names = [t.name for t in tools]
        assert any(
            "read" in n.lower() or "list" in n.lower()
            for n in tool_names
        ), f"Expected file tools, got: {tool_names}"

    async def test_read_file(self, tmp_dir):
        servers = [
            McpServerConfig(
                name="filesystem",
                transport=McpTransport.STDIO,
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", tmp_dir],
            ),
        ]
        tools = await connect_request_mcp(servers)
        tool_map = {t.name: t for t in tools}

        if "read_file" in tool_map:
            result = await tool_map["read_file"].ainvoke({"path": os.path.join(tmp_dir, "hello.txt")})
            assert "hello from integration test" in _text_from_result(result)

    async def test_list_directory(self, tmp_dir):
        servers = [
            McpServerConfig(
                name="filesystem",
                transport=McpTransport.STDIO,
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", tmp_dir],
            ),
        ]
        tools = await connect_request_mcp(servers)
        tool_map = {t.name: t for t in tools}

        if "list_directory" in tool_map:
            result = await tool_map["list_directory"].ainvoke({"path": tmp_dir})
            assert "hello.txt" in _text_from_result(result)


# ---------------------------------------------------------------------------
# Multi-server: all three simultaneously
# ---------------------------------------------------------------------------


class TestMultiServer:
    async def test_connect_all_three(self, repo_path, tmp_dir):
        servers = [
            McpServerConfig(
                name="git",
                transport=McpTransport.STDIO,
                command="uvx",
                args=["mcp-server-git", "--repository", repo_path],
            ),
            McpServerConfig(
                name="memory",
                transport=McpTransport.STDIO,
                command="npx",
                args=["-y", "@modelcontextprotocol/server-memory"],
            ),
            McpServerConfig(
                name="filesystem",
                transport=McpTransport.STDIO,
                command="npx",
                args=["-y", "@modelcontextprotocol/server-filesystem", tmp_dir],
            ),
        ]
        tools = await connect_request_mcp(servers)
        tool_names = [t.name for t in tools]
        assert len(tools) >= 5, f"Expected tools from 3 servers, got {len(tools)}: {tool_names}"

        has_git = any("git" in n for n in tool_names)
        has_memory = any("entity" in n.lower() or "relation" in n.lower() or "graph" in n.lower() for n in tool_names)
        has_fs = any("file" in n.lower() or "directory" in n.lower() or "list" in n.lower() for n in tool_names)

        assert has_git, f"Missing git tools in {tool_names}"
        assert has_memory, f"Missing memory tools in {tool_names}"
        assert has_fs, f"Missing filesystem tools in {tool_names}"

    async def test_init_mcp_static_servers(self, repo_path, monkeypatch):
        server_json = json.dumps([
            {"name": "git", "transport": "stdio", "command": "uvx", "args": ["mcp-server-git", "--repository", repo_path]},
        ])
        monkeypatch.setattr("ai.mcp.Config.MCP_SERVERS", server_json)
        monkeypatch.setattr("ai.mcp.Config.MCP_GATED_TOOLS", [])
        monkeypatch.setattr("ai.mcp.Config.MCP_TOOL_PERMISSIONS", "{}")

        await init_mcp()
        try:
            tools = await get_mcp_tools()
            assert len(tools) > 0
            assert any("git" in t.name for t in tools)
        finally:
            await close_mcp()
