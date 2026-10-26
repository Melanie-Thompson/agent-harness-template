from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from ai.evals.base import (
    _eval_registry,
    clear_eval_registry,
    evaluator,
    get_evaluator,
    get_evaluators_for_stage,
)
from ai.evals.pipeline import run_eval_stage
from schemas.ai import (
    EvalContext,
    EvalResult,
    EvalStage,
    EvalStageConfig,
    EvalVerdict,
)


@pytest.fixture(autouse=True)
def _isolate_registry():
    saved = dict(_eval_registry)
    _eval_registry.clear()
    yield
    _eval_registry.clear()
    _eval_registry.update(saved)


# --- Registry tests ---


def test_evaluator_decorator_registers():
    @evaluator("test_eval", stages=[EvalStage.INPUT], description="test")
    async def test_eval(ctx: EvalContext) -> EvalResult:
        return EvalResult(verdict=EvalVerdict.PASS)

    assert "test_eval" in _eval_registry
    entry = _eval_registry["test_eval"]
    assert entry["stages"] == [EvalStage.INPUT]
    assert entry["description"] == "test"


def test_get_evaluators_for_stage_filters():
    @evaluator("input_only", stages=[EvalStage.INPUT])
    async def input_only(ctx): ...

    @evaluator("output_only", stages=[EvalStage.OUTPUT])
    async def output_only(ctx): ...

    @evaluator("both_stages", stages=[EvalStage.INPUT, EvalStage.OUTPUT])
    async def both_stages(ctx): ...

    input_evals = get_evaluators_for_stage(EvalStage.INPUT)
    input_names = [name for name, _ in input_evals]
    assert "input_only" in input_names
    assert "both_stages" in input_names
    assert "output_only" not in input_names


def test_get_evaluator_returns_entry():
    @evaluator("findable", stages=[EvalStage.INPUT])
    async def findable(ctx): ...

    assert get_evaluator("findable") is not None
    assert get_evaluator("missing") is None


def test_clear_eval_registry():
    @evaluator("temp", stages=[EvalStage.INPUT])
    async def temp(ctx): ...

    clear_eval_registry()
    assert len(_eval_registry) == 0


# --- Pipeline tests ---


async def test_run_eval_stage_all_pass():
    @evaluator("pass1", stages=[EvalStage.INPUT])
    async def pass1(ctx):
        return EvalResult(verdict=EvalVerdict.PASS)

    @evaluator("pass2", stages=[EvalStage.INPUT])
    async def pass2(ctx):
        return EvalResult(verdict=EvalVerdict.PASS)

    ctx = EvalContext(user_id="u1", stage=EvalStage.INPUT, prompt="hi")
    result = await run_eval_stage(EvalStage.INPUT, ctx)
    assert result.verdict == EvalVerdict.PASS


async def test_run_eval_stage_first_fail_stops():
    call_order = []

    @evaluator("fail_first", stages=[EvalStage.INPUT])
    async def fail_first(ctx):
        call_order.append("fail_first")
        return EvalResult(verdict=EvalVerdict.FAIL, reason="blocked")

    @evaluator("never_reached", stages=[EvalStage.INPUT])
    async def never_reached(ctx):
        call_order.append("never_reached")
        return EvalResult(verdict=EvalVerdict.PASS)

    ctx = EvalContext(user_id="u1", stage=EvalStage.INPUT, prompt="bad")
    result = await run_eval_stage(EvalStage.INPUT, ctx)
    assert result.verdict == EvalVerdict.FAIL
    assert result.reason == "blocked"
    assert "never_reached" not in call_order


async def test_run_eval_stage_retry_then_pass():
    attempt = 0

    @evaluator("retry_once", stages=[EvalStage.TOOL_RESULT], max_retries=2)
    async def retry_once(ctx):
        nonlocal attempt
        attempt += 1
        if attempt == 1:
            return EvalResult(verdict=EvalVerdict.RETRY, reason="try again")
        return EvalResult(verdict=EvalVerdict.PASS)

    ctx = EvalContext(user_id="u1", stage=EvalStage.TOOL_RESULT, tool_result="data")
    result = await run_eval_stage(EvalStage.TOOL_RESULT, ctx)
    assert result.verdict == EvalVerdict.PASS
    assert attempt == 2


async def test_run_eval_stage_retry_exhausted():
    @evaluator("always_retry", stages=[EvalStage.TOOL_RESULT], max_retries=2)
    async def always_retry(ctx):
        return EvalResult(verdict=EvalVerdict.RETRY, reason="keep trying")

    ctx = EvalContext(user_id="u1", stage=EvalStage.TOOL_RESULT, tool_result="data")
    result = await run_eval_stage(EvalStage.TOOL_RESULT, ctx)
    assert result.verdict == EvalVerdict.RETRY
    assert result.reason == "keep trying"


async def test_run_eval_stage_disabled():
    @evaluator("should_skip", stages=[EvalStage.INPUT])
    async def should_skip(ctx):
        return EvalResult(verdict=EvalVerdict.FAIL, reason="should not run")

    from ai.guardrails import get_guardrail_config
    cfg = get_guardrail_config()
    cfg.eval_pipeline.input.enabled = False
    try:
        ctx = EvalContext(user_id="u1", stage=EvalStage.INPUT, prompt="hi")
        result = await run_eval_stage(EvalStage.INPUT, ctx)
        assert result.verdict == EvalVerdict.PASS
    finally:
        cfg.eval_pipeline.input.enabled = True


async def test_run_eval_stage_explicit_evaluator_list():
    @evaluator("selected", stages=[EvalStage.OUTPUT])
    async def selected(ctx):
        return EvalResult(verdict=EvalVerdict.PASS)

    @evaluator("not_selected", stages=[EvalStage.OUTPUT])
    async def not_selected(ctx):
        return EvalResult(verdict=EvalVerdict.FAIL, reason="should not run")

    from ai.guardrails import get_guardrail_config
    cfg = get_guardrail_config()
    cfg.eval_pipeline.output.evaluators = ["selected"]
    try:
        ctx = EvalContext(user_id="u1", stage=EvalStage.OUTPUT, output="safe")
        result = await run_eval_stage(EvalStage.OUTPUT, ctx)
        assert result.verdict == EvalVerdict.PASS
    finally:
        cfg.eval_pipeline.output.evaluators = []


async def test_run_eval_stage_unknown_evaluator_skipped():
    from ai.guardrails import get_guardrail_config
    cfg = get_guardrail_config()
    cfg.eval_pipeline.output.evaluators = ["nonexistent_evaluator"]
    try:
        ctx = EvalContext(user_id="u1", stage=EvalStage.OUTPUT, output="safe")
        result = await run_eval_stage(EvalStage.OUTPUT, ctx)
        assert result.verdict == EvalVerdict.PASS
    finally:
        cfg.eval_pipeline.output.evaluators = []


async def test_run_eval_stage_no_evaluators_for_stage():
    ctx = EvalContext(user_id="u1", stage=EvalStage.TOOL_RESULT, tool_result="data")
    result = await run_eval_stage(EvalStage.TOOL_RESULT, ctx)
    assert result.verdict == EvalVerdict.PASS


# --- Built-in evaluator tests ---


async def test_guardrail_input_eval_safe():
    from ai.guardrails import GuardrailResult
    from ai.evals.builtin import guardrail_input_eval

    ctx = EvalContext(user_id="u1", stage=EvalStage.INPUT, prompt="hello")
    with patch("ai.evals.builtin.check_input", AsyncMock(return_value=GuardrailResult(safe=True))):
        result = await guardrail_input_eval(ctx)
    assert result.verdict == EvalVerdict.PASS


async def test_guardrail_input_eval_unsafe():
    from ai.guardrails import GuardrailResult
    from ai.evals.builtin import guardrail_input_eval

    ctx = EvalContext(user_id="u1", stage=EvalStage.INPUT, prompt="bad stuff")
    with patch("ai.evals.builtin.check_input", AsyncMock(return_value=GuardrailResult(safe=False, reason="unsafe"))):
        result = await guardrail_input_eval(ctx)
    assert result.verdict == EvalVerdict.FAIL
    assert result.reason == "unsafe"


async def test_guardrail_output_eval_safe():
    from ai.guardrails import GuardrailResult
    from ai.evals.builtin import guardrail_output_eval

    ctx = EvalContext(user_id="u1", stage=EvalStage.OUTPUT, output="safe response")
    with patch("ai.evals.builtin.check_output", AsyncMock(return_value=GuardrailResult(safe=True))):
        result = await guardrail_output_eval(ctx)
    assert result.verdict == EvalVerdict.PASS


async def test_guardrail_output_eval_unsafe():
    from ai.guardrails import GuardrailResult
    from ai.evals.builtin import guardrail_output_eval

    ctx = EvalContext(user_id="u1", stage=EvalStage.OUTPUT, output="harmful stuff")
    with patch("ai.evals.builtin.check_output", AsyncMock(return_value=GuardrailResult(safe=False, reason="harmful"))):
        result = await guardrail_output_eval(ctx)
    assert result.verdict == EvalVerdict.FAIL
    assert result.reason == "harmful"
