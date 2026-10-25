from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from ai.llm import get_llm, invoke_structured, DEFAULT_MODELS
from config import Config


class Sentiment(BaseModel):
    label: str
    score: float


class Summary(BaseModel):
    title: str
    points: list[str]


@pytest.fixture(autouse=True)
def _reset_llm_config():
    orig = {
        "LLM_PROVIDER": Config.LLM_PROVIDER,
        "LLM_MODEL": Config.LLM_MODEL,
        "LLM_TEMPERATURE": Config.LLM_TEMPERATURE,
        "ANTHROPIC_API_KEY": Config.ANTHROPIC_API_KEY,
        "OPENAI_API_KEY": Config.OPENAI_API_KEY,
        "GOOGLE_API_KEY": Config.GOOGLE_API_KEY,
    }
    yield
    for k, v in orig.items():
        setattr(Config, k, v)


# --- get_llm ---


def test_get_llm_gemini():
    Config.LLM_PROVIDER = "gemini"
    Config.LLM_MODEL = ""
    Config.GOOGLE_API_KEY = "fake"
    llm = get_llm()
    assert type(llm).__name__ == "ChatGoogleGenerativeAI"
    assert "gemini-2.5-flash" in llm.model


def test_get_llm_anthropic():
    Config.LLM_PROVIDER = "anthropic"
    Config.LLM_MODEL = ""
    Config.ANTHROPIC_API_KEY = "fake"
    llm = get_llm()
    assert type(llm).__name__ == "ChatAnthropic"
    assert llm.model == DEFAULT_MODELS["anthropic"]


def test_get_llm_openai():
    Config.LLM_PROVIDER = "openai"
    Config.LLM_MODEL = ""
    Config.OPENAI_API_KEY = "fake"
    llm = get_llm()
    assert type(llm).__name__ == "ChatOpenAI"


def test_get_llm_unsupported():
    Config.LLM_PROVIDER = "llama"
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_llm()


# --- invoke_structured ---


def _mock_llm(return_value):
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=return_value)
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_chain
    return mock_llm, mock_chain


async def test_invoke_structured_returns_pydantic():
    mock_llm, mock_chain = _mock_llm(Sentiment(label="positive", score=0.95))

    result = await invoke_structured("test prompt", Sentiment, model=mock_llm)

    mock_llm.with_structured_output.assert_called_once_with(Sentiment)
    mock_chain.ainvoke.assert_awaited_once_with("test prompt")
    assert isinstance(result, Sentiment)
    assert result.label == "positive"
    assert result.score == 0.95


async def test_invoke_structured_complex_schema():
    expected = Summary(title="Test", points=["a", "b"])
    mock_llm, _ = _mock_llm(expected)

    result = await invoke_structured("summarise this", Summary, model=mock_llm)

    assert isinstance(result, Summary)
    assert result.title == "Test"
    assert result.points == ["a", "b"]


async def test_invoke_structured_uses_default_llm():
    mock_llm, _ = _mock_llm(Sentiment(label="neutral", score=0.5))

    with patch("ai.llm.get_llm", return_value=mock_llm):
        result = await invoke_structured("hello", Sentiment)

        assert result.label == "neutral"


# --- model_override ---


def test_get_llm_model_override():
    Config.LLM_PROVIDER = "anthropic"
    Config.LLM_MODEL = "claude-sonnet-4-20250514"
    Config.ANTHROPIC_API_KEY = "fake"

    llm = get_llm(model_override="claude-haiku-4-20250414")
    assert llm.model == "claude-haiku-4-20250414"
