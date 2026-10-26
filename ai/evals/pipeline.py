from __future__ import annotations

from ai.evals.base import get_evaluator, get_evaluators_for_stage
from ai.guardrails import get_guardrail_config
from schemas.ai import EvalContext, EvalResult, EvalStage, EvalStageConfig, EvalVerdict

_STAGE_ATTR: dict[EvalStage, str] = {
    EvalStage.INPUT: "input",
    EvalStage.TOOL_RESULT: "tool_result",
    EvalStage.OUTPUT: "output",
}


def _get_stage_config(stage: EvalStage) -> EvalStageConfig:
    cfg = get_guardrail_config()
    return getattr(cfg.eval_pipeline, _STAGE_ATTR[stage])


async def run_eval_stage(stage: EvalStage, ctx: EvalContext) -> EvalResult:
    stage_cfg = _get_stage_config(stage)
    if not stage_cfg.enabled:
        return EvalResult(verdict=EvalVerdict.PASS)

    if stage_cfg.evaluators:
        evaluator_list = [
            (name, entry)
            for name in stage_cfg.evaluators
            if (entry := get_evaluator(name)) is not None
        ]
    else:
        evaluator_list = get_evaluators_for_stage(stage)

    for _name, entry in evaluator_list:
        fn = entry["fn"]
        max_retries = min(entry["max_retries"], stage_cfg.max_retries)

        for attempt in range(max_retries + 1):
            result = await fn(ctx)
            if result.verdict == EvalVerdict.PASS:
                break
            if result.verdict == EvalVerdict.FAIL:
                return result
            if attempt == max_retries:
                return result

    return EvalResult(verdict=EvalVerdict.PASS)
