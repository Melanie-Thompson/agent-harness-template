from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai.context import build_context, _estimate_tokens
from db.conversations import add_message, create_conversation
from schemas.conversation import ConversationMessage, Role


def _msg(conversation_id: str, role: Role, content: str, ts_offset: int = 0, **kw) -> ConversationMessage:
    return ConversationMessage(
        conversation_id=conversation_id,
        message_id=kw.pop("message_id", f"m-{ts_offset}"),
        session_id="sess-1",
        user_id="user-1",
        role=role,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=ts_offset),
        content=content,
    )


def test_estimate_tokens():
    assert _estimate_tokens("") == 0
    assert _estimate_tokens("abcd") == 1
    assert _estimate_tokens("a" * 100) == 25


async def test_build_context_returns_messages():
    cid = await create_conversation("user-1", "sess-1")
    await add_message(_msg(cid, Role.USER, "hello", ts_offset=0))
    await add_message(_msg(cid, Role.ASSISTANT, "hi there", ts_offset=1))

    ctx = await build_context(cid)
    assert len(ctx) == 2
    assert ctx[0] == {"role": "user", "content": "hello"}
    assert ctx[1] == {"role": "assistant", "content": "hi there"}


async def test_build_context_respects_max_messages():
    cid = await create_conversation("user-1", "sess-1")
    for i in range(10):
        await add_message(_msg(cid, Role.USER, f"msg-{i}", ts_offset=i))

    ctx = await build_context(cid, max_messages=5, max_tokens=999999)
    assert len(ctx) <= 5


async def test_build_context_respects_max_tokens():
    cid = await create_conversation("user-1", "sess-1")
    await add_message(_msg(cid, Role.USER, "a" * 100, ts_offset=0))
    await add_message(_msg(cid, Role.USER, "b" * 100, ts_offset=1))
    await add_message(_msg(cid, Role.USER, "c" * 100, ts_offset=2))

    ctx = await build_context(cid, max_tokens=50, max_messages=100)
    assert len(ctx) == 2
    assert ctx[0]["content"] == "b" * 100
    assert ctx[1]["content"] == "c" * 100


async def test_build_context_empty_conversation():
    cid = await create_conversation("user-1", "sess-1")
    ctx = await build_context(cid)
    assert ctx == []


async def test_build_context_trims_oldest_first():
    cid = await create_conversation("user-1", "sess-1")
    await add_message(_msg(cid, Role.USER, "x" * 40, ts_offset=0))
    await add_message(_msg(cid, Role.ASSISTANT, "y" * 40, ts_offset=1))
    await add_message(_msg(cid, Role.USER, "z" * 40, ts_offset=2))

    ctx = await build_context(cid, max_tokens=20, max_messages=100)
    assert len(ctx) == 2
    assert ctx[0]["content"] == "y" * 40
    assert ctx[1]["content"] == "z" * 40


async def test_invoke_structured_with_conversation_id():
    cid = await create_conversation("user-1", "sess-1")
    await add_message(_msg(cid, Role.USER, "earlier question", ts_offset=0))

    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(return_value=MagicMock(label="positive", score=0.9))

    mock_llm = MagicMock()
    mock_llm.with_structured_output = MagicMock(return_value=mock_chain)

    with patch("ai.llm.get_llm", return_value=mock_llm):
        from ai.llm import invoke_structured
        from pydantic import BaseModel

        class Sentiment(BaseModel):
            label: str
            score: float

        await invoke_structured("new question", Sentiment, conversation_id=cid)

    call_args = mock_chain.ainvoke.call_args[0][0]
    assert len(call_args) == 2
    assert call_args[0].content == "earlier question"
    assert call_args[1].content == "new question"
