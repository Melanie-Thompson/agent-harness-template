from __future__ import annotations

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
        if not callable(fn):
            raise TypeError(f"harness_tool expects a callable, got {type(fn)}")

        tool = StructuredTool.from_function(
            func=fn if not _is_coroutine(fn) else None,
            coroutine=fn if _is_coroutine(fn) else None,
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

    result = []
    for tool in _registry.values():
        required = tool.metadata.get("permissions", [])
        if all(p in permissions for p in required):
            result.append(tool)
    return result


def get_tool(name: str) -> BaseTool | None:
    return _registry.get(name)


def clear_registry() -> None:
    _registry.clear()


def _is_coroutine(fn: Callable) -> bool:
    import asyncio
    return asyncio.iscoroutinefunction(fn)
