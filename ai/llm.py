from typing import TypeVar

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from config import Config

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "anthropic": "claude-sonnet-4-20250514",
    "openai": "gpt-4o",
}


def get_llm(model_override: str | None = None) -> BaseChatModel:
    provider = Config.LLM_PROVIDER
    model = model_override or Config.LLM_MODEL or DEFAULT_MODELS.get(provider, "")
    temperature = Config.LLM_TEMPERATURE

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            google_api_key=Config.GOOGLE_API_KEY,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=Config.ANTHROPIC_API_KEY,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=Config.OPENAI_API_KEY,
        )

    raise ValueError(f"Unsupported LLM provider: {provider}")


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
