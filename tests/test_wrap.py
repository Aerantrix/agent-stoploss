from types import SimpleNamespace

import pytest

from agent_stoploss import Stopped, StreamingNotSupported


class Messages:
    def __init__(self):
        self.calls = 0

    def create(self, **kw):
        self.calls += 1
        return SimpleNamespace(model=kw["model"], content=[],
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=100,
                                                     cache_read_input_tokens=0,
                                                     cache_creation_input_tokens=0))

    def stream(self, **kw):
        return "streaming"


class FakeAnthropic:
    def __init__(self):
        self.messages = Messages()
        self.api_key = "k"

    def with_options(self, **kw):
        return self


class AsyncCompletions:
    async def create(self, **kw):
        return SimpleNamespace(model="gpt-test", usage={"prompt_tokens": 50, "completion_tokens": 5})


class FakeAsyncOpenAI:
    def __init__(self):
        self.chat = SimpleNamespace(completions=AsyncCompletions())


def test_wrapped_client_records_usage(make_guard, sink):
    guard = make_guard()
    client = guard.wrap(FakeAnthropic())
    with guard.run():
        client.messages.create(model="claude-haiku-5-5", messages=[])
        client.with_options(timeout=5).messages.create(model="claude-haiku-5-5", messages=[])
    calls = sink.of("model_call")
    assert len(calls) == 2 and calls[0]["input_tokens"] == 1000
    assert calls[0]["cost_usd"] == pytest.approx((1000 * 0.10 + 100 * 0.50) / 1e6)
    assert [e["step"] for e in sink.of("step")] == [1, 2]
    assert client.api_key == "k"


def test_outside_a_run_the_client_is_untouched(make_guard, sink):
    guard = make_guard()
    client = guard.wrap(FakeAnthropic())
    client.messages.create(model="m", messages=[])
    assert client.messages.stream(model="m") == "streaming"
    assert len(sink) == 0


def test_streaming_is_refused_inside_a_run(make_guard):
    guard = make_guard()
    client = guard.wrap(FakeAnthropic())
    with guard.run():
        with pytest.raises(StreamingNotSupported):
            client.messages.create(model="m", messages=[], stream=True)
        with pytest.raises(StreamingNotSupported):
            client.messages.stream(model="m")


def test_budget_stop_happens_before_the_call(make_guard):
    guard = make_guard("enforce", run_budget={"max_tokens": 2000})
    raw = FakeAnthropic()
    client = guard.wrap(raw)
    with pytest.raises(Stopped):
        with guard.run():
            for _ in range(5):
                client.messages.create(model="claude-haiku-5-5", messages=[])
    assert raw.messages.calls == 2  # the third call was refused, not paid for


async def test_async_client(make_guard, sink):
    guard = make_guard()
    client = guard.wrap(FakeAsyncOpenAI())
    async with guard.run():
        resp = await client.chat.completions.create(model="gpt-test", messages=[])
    assert resp.model == "gpt-test"
    assert sink.of("model_call")[0]["input_tokens"] == 50
