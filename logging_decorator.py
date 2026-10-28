from __future__ import annotations

import functools
import inspect
import re
import time
from collections.abc import AsyncGenerator, Callable
from typing import Any

import structlog

_REDACT_PATTERN = re.compile(r"password|token|secret|api_key|code|otp", re.IGNORECASE)
_MAX_STR_LEN = 200


def _safe_repr(value: Any, key: str = "") -> Any:
    if _REDACT_PATTERN.search(key):
        return "***"
    if isinstance(value, str):
        return value[:_MAX_STR_LEN] + "..." if len(value) > _MAX_STR_LEN else value
    if isinstance(value, dict):
        return {k: _safe_repr(v, k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 10:
            return [_safe_repr(v) for v in value[:10]] + [f"... ({len(value)} total)"]
        return [_safe_repr(v) for v in value]
    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"
    return value


def _bind_params(func: Callable, args: tuple, kwargs: dict) -> dict[str, Any]:
    try:
        sig = inspect.signature(func)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        params = {}
        for k, v in bound.arguments.items():
            if k == "self":
                continue
            params[k] = _safe_repr(v, k)
        return params
    except Exception:
        return {"args_count": len(args), "kwargs_keys": list(kwargs.keys())}


def log_call(func: Callable | None = None, *, level: str = "info") -> Callable:
    def decorator(fn: Callable) -> Callable:
        log = structlog.get_logger(fn.__module__)
        log_method = level.lower()

        if inspect.isasyncgenfunction(fn):
            @functools.wraps(fn)
            async def async_gen_wrapper(*args: Any, **kwargs: Any) -> AsyncGenerator:
                params = _bind_params(fn, args, kwargs)
                getattr(log, log_method)(f"{fn.__name__}.start", function=fn.__name__, params=params)
                t0 = time.perf_counter()
                try:
                    async for item in fn(*args, **kwargs):
                        yield item
                except Exception as exc:
                    log.error(
                        f"{fn.__name__}.error",
                        function=fn.__name__,
                        error=str(exc),
                        duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                    )
                    raise
                finally:
                    duration = round((time.perf_counter() - t0) * 1000, 2)
                    getattr(log, log_method)(f"{fn.__name__}.end", function=fn.__name__, duration_ms=duration)
            return async_gen_wrapper

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                params = _bind_params(fn, args, kwargs)
                getattr(log, log_method)(f"{fn.__name__}.start", function=fn.__name__, params=params)
                t0 = time.perf_counter()
                try:
                    result = await fn(*args, **kwargs)
                    duration = round((time.perf_counter() - t0) * 1000, 2)
                    getattr(log, log_method)(f"{fn.__name__}.end", function=fn.__name__, duration_ms=duration)
                    return result
                except Exception as exc:
                    log.error(
                        f"{fn.__name__}.error",
                        function=fn.__name__,
                        error=str(exc),
                        duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                    )
                    raise
            return async_wrapper

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            params = _bind_params(fn, args, kwargs)
            getattr(log, log_method)(f"{fn.__name__}.start", function=fn.__name__, params=params)
            t0 = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                duration = round((time.perf_counter() - t0) * 1000, 2)
                getattr(log, log_method)(f"{fn.__name__}.end", function=fn.__name__, duration_ms=duration)
                return result
            except Exception as exc:
                log.error(
                    f"{fn.__name__}.error",
                    function=fn.__name__,
                    error=str(exc),
                    duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                )
                raise
        return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator
