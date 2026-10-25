from datetime import datetime, timezone, timedelta

import pytest

from db.conversations import (
    add_message,
    create_conversation,
    delete_conversation,
    get_conversation_meta,
    get_conversation_store,
    get_messages,
    list_conversations,
)
from schemas.conversation import ConversationMessage, Role


def _msg(conversation_id: str, role: Role = Role.USER, content: str = "hello", **kw) -> ConversationMessage:
    return ConversationMessage(
        conversation_id=conversation_id,
        message_id=kw.pop("message_id", "0000000000000-aaaaaaaaaaaa"),
        session_id=kw.pop("session_id", "sess-1"),
        user_id=kw.pop("user_id", "user-1"),
        role=role,
        timestamp=kw.pop("timestamp", datetime.now(timezone.utc)),
        content=content,
        **kw,
    )


async def test_create_conversation_returns_id():
    cid = await create_conversation("user-1", "sess-1")
    assert isinstance(cid, str)
    assert len(cid) > 10


async def test_create_conversation_stores_metadata():
    cid = await create_conversation("user-1", "sess-1")
    meta = await get_conversation_meta(cid)
    assert meta is not None
    assert meta["user_id"] == "user-1"
    assert meta["session_id"] == "sess-1"
    assert "created_at" in meta


async def test_add_and_get_messages():
    cid = await create_conversation("user-1", "sess-1")
    await add_message(_msg(cid, Role.USER, "hi"))
    await add_message(_msg(cid, Role.ASSISTANT, "hello back"))

    msgs = await get_messages(cid)
    assert len(msgs) == 2
    assert msgs[0].content == "hi"
    assert msgs[1].content == "hello back"


async def test_messages_ordered_by_timestamp():
    cid = await create_conversation("user-1", "sess-1")
    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 2, tzinfo=timezone.utc)
    t3 = datetime(2026, 1, 3, tzinfo=timezone.utc)

    await add_message(_msg(cid, Role.ASSISTANT, "second", timestamp=t2, message_id="m2"))
    await add_message(_msg(cid, Role.USER, "first", timestamp=t1, message_id="m1"))
    await add_message(_msg(cid, Role.USER, "third", timestamp=t3, message_id="m3"))

    msgs = await get_messages(cid)
    assert [m.content for m in msgs] == ["first", "second", "third"]


async def test_get_messages_time_range():
    cid = await create_conversation("user-1", "sess-1")
    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 1, 2, tzinfo=timezone.utc)
    t3 = datetime(2026, 1, 3, tzinfo=timezone.utc)

    await add_message(_msg(cid, content="a", timestamp=t1, message_id="m1"))
    await add_message(_msg(cid, content="b", timestamp=t2, message_id="m2"))
    await add_message(_msg(cid, content="c", timestamp=t3, message_id="m3"))

    # after/before are inclusive (ZRANGEBYSCORE default), so t1..t3 returns all 3
    msgs = await get_messages(cid, after=t1, before=t3)
    assert len(msgs) == 3

    # narrow the range to only include the middle message
    msgs = await get_messages(cid, after=t2, before=t2)
    assert len(msgs) == 1
    assert msgs[0].content == "b"


async def test_get_messages_limit():
    cid = await create_conversation("user-1", "sess-1")
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(5):
        await add_message(_msg(
            cid, content=f"msg-{i}",
            timestamp=base + timedelta(seconds=i),
            message_id=f"m{i}",
        ))

    msgs = await get_messages(cid, limit=3)
    assert len(msgs) == 3
    assert msgs[0].content == "msg-0"


async def test_list_conversations_for_user():
    cid1 = await create_conversation("user-1", "sess-1")
    cid2 = await create_conversation("user-1", "sess-1")
    await create_conversation("user-2", "sess-2")

    convs = await list_conversations("user-1")
    ids = [c.conversation_id for c in convs]
    assert cid1 in ids
    assert cid2 in ids
    assert len(ids) == 2


async def test_list_conversations_includes_message_count():
    cid = await create_conversation("user-1", "sess-1")
    await add_message(_msg(cid, content="a", message_id="m1"))
    await add_message(_msg(cid, content="b", message_id="m2"))

    convs = await list_conversations("user-1")
    assert convs[0].message_count == 2


async def test_delete_conversation():
    cid = await create_conversation("user-1", "sess-1")
    await add_message(_msg(cid, content="hi"))

    deleted = await delete_conversation(cid, "user-1")
    assert deleted is True

    msgs = await get_messages(cid)
    assert msgs == []

    meta = await get_conversation_meta(cid)
    assert meta is None

    convs = await list_conversations("user-1")
    assert len(convs) == 0


async def test_get_messages_empty_conversation():
    msgs = await get_messages("nonexistent-conv-id")
    assert msgs == []


async def test_get_conversation_meta_nonexistent():
    meta = await get_conversation_meta("nonexistent-conv-id")
    assert meta is None


async def test_ttl_is_set_on_keys():
    cid = await create_conversation("user-1", "sess-1")
    await add_message(_msg(cid, content="hi"))

    r = get_conversation_store()
    meta_ttl = await r.ttl(f"conv_meta:{cid}")
    conv_ttl = await r.ttl(f"conv:{cid}")
    user_ttl = await r.ttl(f"user_convs:user-1")

    assert meta_ttl > 0
    assert conv_ttl > 0
    assert user_ttl > 0
