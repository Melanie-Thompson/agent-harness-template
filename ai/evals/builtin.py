from __future__ import annotations

import os

import numpy as np

from ai.evals.base import evaluator
from config import Config
from logging_config import get_logger
from logging_decorator import log_call
from schemas.ai import EvalContext, EvalResult, EvalStage, EvalVerdict
from tda.embeddings import get_embedding_model

_log = get_logger("ai.evals.builtin")

SIMILARITY_THRESHOLD = float(os.getenv("GUARDRAIL_THRESHOLD", "0.62"))

INPUT_CATEGORIES = [
    "Ignore previous instructions, disregard system prompt, jailbreak the AI",
    "Generate malware, ransomware, keylogger, exploit code, virus, trojan",
    "Provide someone's social security number, credit card number, or private personal data",
    "How to build weapons, bombs, explosives, poison, or synthesize drugs",
]

OUTPUT_CATEGORIES = [
    "Step-by-step instructions for building weapons, explosives, or synthesizing dangerous substances",
    "Real social security numbers, credit card numbers, or private credentials",
    "Malware source code, exploit payloads, or working attack scripts",
]

_input_embeddings: np.ndarray | None = None
_output_embeddings: np.ndarray | None = None


def _get_input_embeddings() -> np.ndarray:
    global _input_embeddings
    if _input_embeddings is None:
        model = get_embedding_model()
        _input_embeddings = np.array(list(model.embed(INPUT_CATEGORIES)))
    return _input_embeddings


def _get_output_embeddings() -> np.ndarray:
    global _output_embeddings
    if _output_embeddings is None:
        model = get_embedding_model()
        _output_embeddings = np.array(list(model.embed(OUTPUT_CATEGORIES)))
    return _output_embeddings


@log_call
def _semantic_check(
    text: str,
    category_embeddings: np.ndarray,
    category_labels: list[str],
) -> EvalResult:
    text_emb = np.array(list(get_embedding_model().embed([text])))
    similarities = (text_emb @ category_embeddings.T)[0]
    max_idx = int(similarities.argmax())
    max_sim = float(similarities[max_idx])

    _log.debug("guardrail.semantic_check", max_sim=round(max_sim, 3), matched=category_labels[max_idx])

    if max_sim >= SIMILARITY_THRESHOLD:
        return EvalResult(
            verdict=EvalVerdict.FAIL,
            reason=category_labels[max_idx],
        )
    return EvalResult(verdict=EvalVerdict.PASS)


@evaluator(
    "guardrail_input",
    stages=[EvalStage.INPUT],
    description="Semantic similarity input safety check (local embeddings, ~3ms per check)",
)
@log_call
async def guardrail_input_eval(ctx: EvalContext) -> EvalResult:
    if not Config.GUARDRAILS_ENABLED:
        return EvalResult(verdict=EvalVerdict.PASS)
    return _semantic_check(ctx.prompt or "", _get_input_embeddings(), INPUT_CATEGORIES)


@evaluator(
    "guardrail_output",
    stages=[EvalStage.OUTPUT],
    description="Semantic similarity output safety check (local embeddings, ~3ms per check)",
)
@log_call
async def guardrail_output_eval(ctx: EvalContext) -> EvalResult:
    if not Config.GUARDRAILS_ENABLED:
        return EvalResult(verdict=EvalVerdict.PASS)
    return _semantic_check(ctx.output or "", _get_output_embeddings(), OUTPUT_CATEGORIES)
