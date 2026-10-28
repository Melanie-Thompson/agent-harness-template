import asyncio

from logging_broadcast import LogBroadcast, close_log_broadcast, get_log_broadcast, init_log_broadcast
from logging_decorator import _safe_repr, log_call


# --- _safe_repr ---


class TestSafeRepr:
    def test_redacts_sensitive_keys(self):
        assert _safe_repr("hunter2", "password") == "***"
        assert _safe_repr("abc123", "api_key") == "***"
        assert _safe_repr("xyz", "otp") == "***"
        assert _safe_repr("tok", "secret_token") == "***"

    def test_truncates_long_strings(self):
        long = "x" * 300
        result = _safe_repr(long)
        assert result.endswith("...")
        assert len(result) == 203

    def test_short_string_unchanged(self):
        assert _safe_repr("hello") == "hello"

    def test_truncates_long_lists(self):
        long_list = list(range(20))
        result = _safe_repr(long_list)
        assert len(result) == 11
        assert "20 total" in result[-1]

    def test_bytes_shown_as_placeholder(self):
        result = _safe_repr(b"binary stuff")
        assert result == "<bytes len=12>"

    def test_nested_dict_redaction(self):
        data = {"config": {"token": "secret123", "name": "ok"}}
        result = _safe_repr(data)
        assert result["config"]["token"] == "***"
        assert result["config"]["name"] == "ok"


# --- log_call decorator ---


class TestLogCallSync:
    def test_wraps_sync_function(self):
        @log_call
        def add(a, b):
            return a + b

        assert add(1, 2) == 3
        assert add.__name__ == "add"

    def test_sync_error_reraises(self):
        @log_call
        def fail():
            raise ValueError("boom")

        try:
            fail()
        except ValueError as e:
            assert str(e) == "boom"
        else:
            assert False, "Should have raised"


class TestLogCallAsync:
    async def test_wraps_async_function(self):
        @log_call
        async def add_async(a, b):
            return a + b

        assert await add_async(1, 2) == 3
        assert add_async.__name__ == "add_async"

    async def test_async_error_reraises(self):
        @log_call
        async def fail_async():
            raise RuntimeError("async boom")

        try:
            await fail_async()
        except RuntimeError as e:
            assert str(e) == "async boom"
        else:
            assert False, "Should have raised"


class TestLogCallAsyncGen:
    async def test_wraps_async_generator(self):
        @log_call
        async def gen_items():
            yield 1
            yield 2
            yield 3

        items = [item async for item in gen_items()]
        assert items == [1, 2, 3]
        assert gen_items.__name__ == "gen_items"

    async def test_async_gen_error_reraises(self):
        @log_call
        async def fail_gen():
            yield 1
            raise ValueError("gen boom")

        items = []
        try:
            async for item in fail_gen():
                items.append(item)
        except ValueError as e:
            assert str(e) == "gen boom"
        else:
            assert False, "Should have raised"
        assert items == [1]


class TestLogCallWithLevel:
    def test_custom_level(self):
        @log_call(level="debug")
        def quiet_func():
            return "ok"

        assert quiet_func() == "ok"


# --- LogBroadcast ---


class TestLogBroadcast:
    def test_publish_and_subscribe(self):
        bc = LogBroadcast(buffer_size=100)
        queue = bc.subscribe(catchup=0)
        bc.publish({"event": "hello"})
        assert queue.get_nowait() == {"event": "hello"}

    def test_catchup_from_buffer(self):
        bc = LogBroadcast(buffer_size=100)
        for i in range(5):
            bc.publish({"event": f"msg-{i}"})
        queue = bc.subscribe(catchup=3)
        items = []
        while not queue.empty():
            items.append(queue.get_nowait())
        assert len(items) == 3
        assert items[0]["event"] == "msg-2"

    def test_unsubscribe(self):
        bc = LogBroadcast(buffer_size=100)
        queue = bc.subscribe(catchup=0)
        bc.unsubscribe(queue)
        bc.publish({"event": "missed"})
        assert queue.empty()

    def test_full_queue_drops_subscriber(self):
        bc = LogBroadcast(buffer_size=1000)
        queue = bc.subscribe(catchup=0)
        for i in range(501):
            bc.publish({"event": f"spam-{i}"})
        assert queue not in bc._subscribers

    def test_buffer_size_limit(self):
        bc = LogBroadcast(buffer_size=5)
        for i in range(10):
            bc.publish({"event": f"msg-{i}"})
        assert len(bc._buffer) == 5
        assert bc._buffer[0]["event"] == "msg-5"


# --- Singleton lifecycle ---


class TestBroadcastSingleton:
    def test_init_and_get(self):
        close_log_broadcast()
        assert get_log_broadcast() is None

        init_log_broadcast()
        bc = get_log_broadcast()
        assert bc is not None
        assert isinstance(bc, LogBroadcast)

        close_log_broadcast()
        assert get_log_broadcast() is None
