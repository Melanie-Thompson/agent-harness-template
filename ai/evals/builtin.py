from ai.evals.base import evaluator
from ai.guardrails import check_input, check_output
from schemas.ai import EvalContext, EvalResult, EvalStage, EvalVerdict


@evaluator(
    "guardrail_input",
    stages=[EvalStage.INPUT],
    description="LLM-based input safety check",
)
async def guardrail_input_eval(ctx: EvalContext) -> EvalResult:
    result = await check_input(ctx.prompt or "")
    if result.safe:
        return EvalResult(verdict=EvalVerdict.PASS)
    return EvalResult(verdict=EvalVerdict.FAIL, reason=result.reason)


@evaluator(
    "guardrail_output",
    stages=[EvalStage.OUTPUT],
    description="LLM-based output safety check",
)
async def guardrail_output_eval(ctx: EvalContext) -> EvalResult:
    result = await check_output(ctx.output or "")
    if result.safe:
        return EvalResult(verdict=EvalVerdict.PASS)
    return EvalResult(verdict=EvalVerdict.FAIL, reason=result.reason)
