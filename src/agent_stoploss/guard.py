"""StopLoss: the object a developer creates once per process."""

from __future__ import annotations

import functools
import inspect
import os
import threading
import time
import warnings
from typing import Any, Callable, Mapping

from .config import Config
from .journal import JournalWriter, NullWriter
from .run import Run, StopEvent, Stopped, current_run


class StopLoss:
    """Spend control for agent runs.

        guard = StopLoss.from_file("stoploss.yaml")   # or StopLoss(mode="shadow")

        @guard.tool
        def search(query): ...

        client = guard.wrap(Anthropic())

        with guard.run(agent="weekly-report") as run:
            ...

    One StopLoss serves any number of runs, concurrently; each run keeps its
    own state.
    """

    def __init__(self, config: Config | Mapping[str, Any] | None = None, *,
                 mode: str | None = None,
                 on_stop: Callable[[StopEvent], None] | None = None,
                 on_verdict: Callable[[StopEvent], None] | None = None,
                 journal: JournalWriter | None = None):
        if not isinstance(config, Config):
            config = Config.from_dict(config)
        if mode is not None:
            config = Config.from_dict({**config.source, "mode": mode})
        self.config = config
        self.on_stop = on_stop
        self.on_verdict = on_verdict
        if journal is not None:
            self._writer = journal
        elif config.log_dir is not None:
            self._writer = JournalWriter(config.log_dir)
        else:
            self._writer = NullWriter()
        self._warned: set[str] = set()
        self._warn_lock = threading.Lock()

    @classmethod
    def from_file(cls, path: str | os.PathLike, **kwargs: Any) -> "StopLoss":
        return cls(Config.from_file(path), **kwargs)

    # -- runs -------------------------------------------------------------

    def run(self, agent: str | None = None, run_id: str | None = None,
            metadata: Mapping[str, Any] | None = None) -> Run:
        """A new run. Use as a context manager (``with`` or ``async with``)."""
        return Run(self, agent=agent, run_id=run_id, metadata=dict(metadata or {}))

    # -- tools ------------------------------------------------------------

    def tool(self, fn: Callable | None = None, *, name: str | None = None):
        """Decorator: every call of the tool inside a run is checked and
        recorded. Works on plain and async functions. Outside a run the tool
        runs as usual, with a one-time warning."""

        def decorate(f: Callable) -> Callable:
            tool_name = name or getattr(f, "__name__", "tool")
            try:
                sig = inspect.signature(f)
            except (TypeError, ValueError):
                sig = None

            def arguments(a: tuple, kw: dict) -> dict:
                # What the agent passed, by parameter name; defaults it did
                # not pass stay out, so they do not change the call print.
                if sig is not None:
                    try:
                        return dict(sig.bind(*a, **kw).arguments)
                    except TypeError:
                        pass
                return {"args": list(a), **kw} if a else dict(kw)

            if inspect.iscoroutinefunction(f):
                @functools.wraps(f)
                async def async_wrapper(*a, **kw):
                    run = current_run()
                    if run is None:
                        self._warn_once(f"no-run:{tool_name}", f"agent-stoploss: tool {tool_name!r} "
                                        "called outside guard.run(); not checked")
                        return await f(*a, **kw)
                    args = arguments(a, kw)
                    run.before_tool(tool_name, args)
                    t0 = time.perf_counter()
                    try:
                        result = await f(*a, **kw)
                    except Stopped:
                        raise
                    except Exception as exc:
                        run.after_tool(tool_name, args, error=exc,
                                       latency_ms=(time.perf_counter() - t0) * 1000)
                        raise
                    run.after_tool(tool_name, args, result=result,
                                   latency_ms=(time.perf_counter() - t0) * 1000)
                    return result

                return async_wrapper

            @functools.wraps(f)
            def wrapper(*a, **kw):
                run = current_run()
                if run is None:
                    self._warn_once(f"no-run:{tool_name}", f"agent-stoploss: tool {tool_name!r} "
                                    "called outside guard.run(); not checked")
                    return f(*a, **kw)
                args = arguments(a, kw)
                run.before_tool(tool_name, args)
                t0 = time.perf_counter()
                try:
                    result = f(*a, **kw)
                except Stopped:
                    raise
                except Exception as exc:
                    run.after_tool(tool_name, args, error=exc,
                                   latency_ms=(time.perf_counter() - t0) * 1000)
                    raise
                run.after_tool(tool_name, args, result=result,
                               latency_ms=(time.perf_counter() - t0) * 1000)
                return result

            return wrapper

        return decorate(fn) if fn is not None else decorate

    # -- model clients ----------------------------------------------------

    def wrap(self, client: Any) -> Any:
        """Wrap an Anthropic or OpenAI client (sync or async) so every model
        call inside a run records its tokens and cost by itself."""
        from .wrap import wrap_client

        return wrap_client(client)

    # -- internals --------------------------------------------------------

    def _callback(self, name: str, event: StopEvent) -> None:
        fn = getattr(self, name)
        if fn is None:
            return
        try:
            fn(event)
        except Exception as exc:  # a broken alert must not break the agent
            self._warn_once(f"cb:{name}:{type(exc).__name__}",
                            f"agent-stoploss: {name} callback raised {exc!r}")

    def _warn_unpriced(self, model: str | None) -> None:
        self._warn_once(f"price:{model}", f"agent-stoploss: no price for model {model!r}; "
                        "its cost is counted as $0. Add it under prices: in the config, "
                        "or set strict_pricing: true to stop such runs.")

    def _warn_once(self, key: str, message: str) -> None:
        with self._warn_lock:
            if key in self._warned:
                return
            self._warned.add(key)
        warnings.warn(message, RuntimeWarning, stacklevel=4)
