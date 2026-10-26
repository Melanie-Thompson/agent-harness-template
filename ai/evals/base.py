from __future__ import annotations

from collections.abc import Callable

from schemas.ai import EvalResult, EvalStage

_eval_registry: dict[str, dict] = {}


def evaluator(
    name: str,
    *,
    stages: list[EvalStage],
    description: str = "",
    max_retries: int = 1,
) -> Callable:
    def decorator(fn: Callable) -> Callable:
        _eval_registry[name] = {
            "fn": fn,
            "stages": stages,
            "description": description,
            "max_retries": max_retries,
        }
        return fn

    return decorator


def get_evaluators_for_stage(stage: EvalStage) -> list[tuple[str, dict]]:
    return [
        (name, entry)
        for name, entry in _eval_registry.items()
        if stage in entry["stages"]
    ]


def get_evaluator(name: str) -> dict | None:
    return _eval_registry.get(name)


def clear_eval_registry() -> None:
    _eval_registry.clear()
