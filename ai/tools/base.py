from __future__ import annotations

import asyncio
from typing import Any, Callable

from langchain_core.tools import BaseTool, StructuredTool

_registry: dict[str, BaseTool] = {}


def harness_tool(
    name: str,
    *,
    description: str,
    permissions: list[str] | None = None,
    gated: bool = False,
) -> Callable:
    def decorator(fn: Callable) -> BaseTool:
        is_async = asyncio.iscoroutinefunction(fn)
        tool = StructuredTool.from_function(
            func=None if is_async else fn,
            coroutine=fn if is_async else None,
            name=name,
            description=description,
        )
        tool.metadata = {
            "permissions": permissions or [],
            "gated": gated,
        }
        _registry[name] = tool
        return tool

    return decorator


def get_tools(*, permissions: list[str] | None = None) -> list[BaseTool]:
    if permissions is None:
        return list(_registry.values())

    perm_set = set(permissions)
    return [t for t in _registry.values() if perm_set.issuperset(t.metadata.get("permissions", []))]


def get_tool(name: str) -> BaseTool | None:
    return _registry.get(name)


def clear_registry() -> None:
    _registry.clear()
