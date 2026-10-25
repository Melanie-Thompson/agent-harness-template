from pydantic import BaseModel, Field


class GuardrailRulesConfig(BaseModel):
    enabled: bool = True
    categories: list[str] = Field(default_factory=list)
    custom_rules: list[str] = Field(default_factory=list)


class GuardrailConfig(BaseModel):
    input: GuardrailRulesConfig = Field(default_factory=GuardrailRulesConfig)
    output: GuardrailRulesConfig = Field(default_factory=GuardrailRulesConfig)
