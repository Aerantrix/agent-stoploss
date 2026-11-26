"""Wrapping model clients so token usage is recorded without extra code.

The wrapper is a transparent proxy. It intercepts only the methods that make
a model call, and only while a run is active:

    Anthropic   messages.create, beta.messages.create
    OpenAI      chat.completions.create, responses.create

Streaming is refused inside a run (StreamingNotSupported) rather than counted
as zero. Everything else passes straight through to the real client.
"""

from __future__ import annotations

import inspect
from typing import Any

from .run import StreamingNotSupported, current_run

CREATE = {
    ("messages", "create"),
    ("beta", "messages", "create"),
    ("chat", "completions", "create"),
    ("responses", "create"),
}
STREAM = {
    ("messages", "stream"),
    ("beta", "messages", "stream"),
    ("chat", "completions", "stream"),
    ("responses", "stream"),
}
_PREFIXES = {p[:i] for p in CREATE | STREAM for i in range(1, len(p))}


def wrap_client(client: Any) -> Any:
    return _Proxy(client, ())


class _Proxy:
    __slots__ = ("_target", "_path")

    def __init__(self, target: Any, path: tuple[str, ...]):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_path", path)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        path = self._path + (name,)
        if path in CREATE and callable(attr):
            return _wrap_create(attr)
        if path in STREAM and callable(attr):
            return _wrap_stream(attr, ".".join(path))
        if not self._path and name in ("with_options", "copy") and callable(attr):
            return lambda *a, **kw: _Proxy(attr(*a, **kw), ())
        if path in _PREFIXES:
            return _Proxy(attr, path)
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._target, name, value)

    def __repr__(self) -> str:
        return f"<agent-stoploss wrapped {self._target!r}>"


def _record(response: Any, kwargs: dict) -> None:
    run = current_run()
    usage = getattr(response, "usage", None)
    if run is None or usage is None:
        return
    model = getattr(response, "model", None) or kwargs.get("model")
    run.record_usage(usage, model=model)


def _wrap_create(method: Any) -> Any:
    def create(*args: Any, **kwargs: Any) -> Any:
        run = current_run()
        if run is None:
            return method(*args, **kwargs)
        if kwargs.get("stream"):
            raise StreamingNotSupported(
                "agent-stoploss cannot count tokens of a streamed response yet; "
                "call without stream=True inside a run, or record usage yourself "
                "with run.record_usage()")
        run.before_model_call()
        response = method(*args, **kwargs)
        if inspect.isawaitable(response):
            async def finish() -> Any:
                result = await response
                _record(result, kwargs)
                return result

            return finish()
        _record(response, kwargs)
        return response

    return create


def _wrap_stream(method: Any, name: str) -> Any:
    def stream(*args: Any, **kwargs: Any) -> Any:
        if current_run() is None:
            return method(*args, **kwargs)
        raise StreamingNotSupported(
            f"agent-stoploss cannot count tokens of {name}() yet; use the non-streaming "
            "call inside a run, or record usage yourself with run.record_usage()")

    return stream
