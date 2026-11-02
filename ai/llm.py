import os
from typing import Any, Callable, TypeVar

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from config import Config
from errors import UnsupportedProviderError
from logging_decorator import log_call

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash",
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
    "openrouter": "nvidia/nemotron-3.5-super-vl",
}

_MODEL_REGISTRY: dict[str, dict] | None = None


def _get_model_registry() -> dict[str, dict]:
    global _MODEL_REGISTRY
    if _MODEL_REGISTRY is None:
        import json
        raw = os.getenv("HARNESS_LLM_MODELS", "[]")
        try:
            models = json.loads(raw)
            _MODEL_REGISTRY = {m["id"]: m for m in models}
        except (json.JSONDecodeError, KeyError):
            _MODEL_REGISTRY = {}
    return _MODEL_REGISTRY


def _make_gemini(model: str, temperature: float) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=Config.GOOGLE_API_KEY)


def _make_anthropic(model: str, temperature: float) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=model, temperature=temperature, api_key=Config.ANTHROPIC_API_KEY)


def _make_openai(model: str, temperature: float) -> BaseChatModel:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, temperature=temperature, api_key=Config.OPENAI_API_KEY)


def _make_openrouter(model: str, temperature: float) -> BaseChatModel:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=Config.OPENROUTER_API_KEY,
        base_url=Config.OPENROUTER_BASE_URL,
    )


_PROVIDERS: dict[str, Callable[[str, float], BaseChatModel]] = {
    "gemini": _make_gemini,
    "anthropic": _make_anthropic,
    "openai": _make_openai,
    "openrouter": _make_openrouter,
}


@log_call
def get_llm(model_override: str | None = None) -> BaseChatModel:
    provider = Config.LLM_PROVIDER
    model_id = model_override or Config.LLM_MODEL or ""
    temperature = Config.LLM_TEMPERATURE

    registry = _get_model_registry()
    entry = registry.get(model_id)
    if entry:
        provider = entry.get("provider", provider)
        model = entry.get("model", model_id)
        base_url = entry.get("base_url")
        api_key_env = entry.get("api_key_env")
        if base_url and api_key_env:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=os.getenv(api_key_env, ""),
                base_url=base_url,
            )
    else:
        model = model_id or DEFAULT_MODELS.get(provider, "")

    factory = _PROVIDERS.get(provider)
    if not factory:
        raise UnsupportedProviderError(provider)
    return factory(model, temperature)


@log_call
async def invoke_structured(
    prompt: str,
    output_schema: type[T],
    *,
    model: BaseChatModel | None = None,
    conversation_id: str | None = None,
) -> T:
    llm = model or get_llm()
    structured_llm = llm.with_structured_output(output_schema)

    if conversation_id:
        from ai.context import build_context
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        role_map = {"user": HumanMessage, "assistant": AIMessage, "system": SystemMessage}
        history = await build_context(conversation_id)
        messages = [role_map.get(m["role"], HumanMessage)(content=m["content"]) for m in history]
        messages.append(HumanMessage(content=prompt))
        return await structured_llm.ainvoke(messages)

    return await structured_llm.ainvoke(prompt)
