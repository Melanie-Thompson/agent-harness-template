from __future__ import annotations

import pytest

from ai.pending import (
    _KEY_PREFIX,
    _pending_futures,
    cleanup_pending,
    create_pending,
    get_pending_meta,
    resolve_pending,
)
from config import Config
from db.sessions import get_dragonfly
from errors import PendingActionNotFoundError


@pytest.fixture(autouse=True)
def _clean_futures():
    _pending_futures.clear()
    yield
    _pending_futures.clear()


async def test_create_pending_stores_in_dragonfly():
    await create_pending("p1", tool_name="db_write", tool_args={"x": 1}, user_id="u1")
    r = get_dragonfly()
    raw = await r.get(f"{_KEY_PREFIX}p1")
    assert raw is not None
    assert "db_write" in raw


async def test_create_pending_sets_ttl():
    await create_pending("p2", tool_name="db_write", tool_args={}, user_id="u1")
    r = get_dragonfly()
    ttl = await r.ttl(f"{_KEY_PREFIX}p2")
    assert 0 < ttl <= Config.AGENT_PENDING_TTL


async def test_create_pending_returns_future():
    future = await create_pending("p3", tool_name="t", tool_args={}, user_id="u1")
    assert not future.done()
    assert "p3" in _pending_futures


async def test_resolve_pending_signals_future():
    future = await create_pending("p4", tool_name="t", tool_args={}, user_id="u1")
    action = await resolve_pending("p4", True)
    assert action == "approved"
    assert future.result() is True


async def test_resolve_pending_denied():
    future = await create_pending("p5", tool_name="t", tool_args={}, user_id="u1")
    action = await resolve_pending("p5", False)
    assert action == "denied"
    assert future.result() is False


async def test_resolve_pending_deletes_dragonfly_key():
    await create_pending("p6", tool_name="t", tool_args={}, user_id="u1")
    await resolve_pending("p6", True)
    r = get_dragonfly()
    assert await r.get(f"{_KEY_PREFIX}p6") is None


async def test_resolve_pending_idempotent():
    await create_pending("p7", tool_name="t", tool_args={}, user_id="u1")
    await resolve_pending("p7", True)
    with pytest.raises(PendingActionNotFoundError):
        await resolve_pending("p7", True)


async def test_resolve_pending_expired():
    r = get_dragonfly()
    await r.delete(f"{_KEY_PREFIX}nonexistent")
    with pytest.raises(PendingActionNotFoundError):
        await resolve_pending("nonexistent", True)


async def test_cleanup_pending_removes_all():
    await create_pending("p8", tool_name="t", tool_args={}, user_id="u1")
    await cleanup_pending("p8")
    assert "p8" not in _pending_futures
    r = get_dragonfly()
    assert await r.get(f"{_KEY_PREFIX}p8") is None


async def test_get_pending_meta_round_trip():
    await create_pending("p9", tool_name="db_write", tool_args={"a": 1}, user_id="u1")
    meta = await get_pending_meta("p9")
    assert meta is not None
    assert meta.pending_id == "p9"
    assert meta.tool_name == "db_write"
    assert meta.tool_args == {"a": 1}
    assert meta.user_id == "u1"


async def test_get_pending_meta_missing():
    meta = await get_pending_meta("missing")
    assert meta is None
