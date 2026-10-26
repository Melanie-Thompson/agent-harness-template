from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class GuardrailRulesConfig(BaseModel):
    enabled: bool = True
    categories: list[str] = Field(default_factory=list)
    custom_rules: list[str] = Field(default_factory=list)


class EvalStage(str, Enum):
    INPUT = "input"
    TOOL_RESULT = "tool_result"
    OUTPUT = "output"


class EvalVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    RETRY = "retry"


class EvalResult(BaseModel):
    verdict: EvalVerdict
    reason: str = ""
    corrective_context: str | None = None


class EvalContext(BaseModel):
    user_id: str
    stage: EvalStage
    prompt: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    tool_result: str | None = None
    output: str | None = None


class EvalStageConfig(BaseModel):
    enabled: bool = True
    evaluators: list[str] = Field(default_factory=list)
    max_retries: int = 1


class EvalPipelineConfig(BaseModel):
    input: EvalStageConfig = Field(default_factory=EvalStageConfig)
    tool_result: EvalStageConfig = Field(default_factory=EvalStageConfig)
    output: EvalStageConfig = Field(default_factory=EvalStageConfig)


class GuardrailConfig(BaseModel):
    input: GuardrailRulesConfig = Field(default_factory=GuardrailRulesConfig)
    output: GuardrailRulesConfig = Field(default_factory=GuardrailRulesConfig)
    eval_pipeline: EvalPipelineConfig = Field(default_factory=EvalPipelineConfig)
