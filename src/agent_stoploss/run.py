"""One agent run: the live side of agent-stoploss.

A Run turns what the agent does into journal events, feeds the facts to the
Engine, and acts on its verdicts according to the mode:

    shadow   record everything, change nothing; the agent never notices
    enforce  hand warnings to the agent as notices, raise Stopped on a stop
"""

from __future__ import annotations

import contextvars
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from . import __version__
from .engine import ALLOW, Decision, Engine, Verdict, strongest
from .errors import status_of
from .fingerprint import call_print, canonical, result_print, shape, to_plain
from .pricing import cost_of, find_price, parse_usage

if TYPE_CHECKING:
    from .guard import StopLoss

OUTCOMES = ("ok", "failed", "unknown")
RAW_LIMIT = 10_000

_current: contextvars.ContextVar["Run | None"] = contextvars.ContextVar(
    "agent_stoploss_run", default=None)


def current_run() -> "Run | None":
    """The run active in this context (thread or asyncio task), if any."""
    return _current.get()


class Stopped(Exception):
    """Raised in enforce mode when agent-stoploss stops a run.

    Catch it at the top of the agent loop to finish cleanly: report what was
    done and why the run ended.
    """

    def __init__(self, verdict: Verdict, run: "Run"):
        self.mechanism = verdict.mechanism
        self.reason = verdict.reason
        self.run_id = run.run_id
        self.summary = run.summary()
        super().__init__(f"agent-stoploss stopped run {run.run_id}: {verdict.reason} "
                         f"[{verdict.mechanism}]")


class ToolFailed(Exception):
    """Pass as ``error=`` to mark a tool call failed without an exception of
    your own, e.g. a result flagged is_error. ``status`` is the HTTP status
    if there was one."""

    def __init__(self, message: str = "tool failed", status: int | None = None):
        super().__init__(message)
        self.status = status


class StreamingNotSupported(RuntimeError):
    """agent-stoploss cannot count tokens of a streamed response yet. Raised
    instead of counting it as zero."""


@dataclass(frozen=True)
class StopEvent:
    """What on_stop and on_verdict callbacks receive."""

    run_id: str
    agent: str | None
    decision: str
    mechanism: str | None
    reason: str
    enforced: bool
    step: int
    summary: dict[str, Any] = field(default_factory=dict)


class Run:
    def __init__(self, guard: "StopLoss", agent: str | None = None, run_id: str | None = None,
                 metadata: dict[str, Any] | None = None):
        self.guard = guard
        self.config = guard.config
        self.agent = agent
        self.run_id = run_id or f"r-{uuid.uuid4().hex[:12]}"
        self.metadata = dict(metadata or {})
        self.engine = Engine(self.config)
        self.first_stop: Verdict | None = None
        self.stopped: Verdict | None = None
        self.warn_count = 0
        self._lock = threading.RLock()
        self._step_open = False
        self._notices: list[str] = []
        self._outcome: str | None = None
        self._outcome_source: str | None = None
        self._started = False
        self._ended = False
        self._tokens: list[contextvars.Token] = []

    # -- properties -------------------------------------------------------

    @property
    def enforce(self) -> bool:
        return self.config.mode == "enforce"

    @property
    def step(self) -> int:
        return self.engine.step

    @property
    def tokens(self) -> int:
        return self.engine.tokens

    @property
    def cost_usd(self) -> float:
        return self.engine.cost

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "agent": self.agent,
            "mode": self.config.mode,
            "steps": self.engine.step,
            "tokens": self.engine.tokens,
            "cost_usd": round(self.engine.cost, 6),
            "would_stop": self.first_stop.mechanism if self.first_stop else None,
            "stopped": self.stopped.mechanism if self.stopped else None,
        }

    # -- lifecycle --------------------------------------------------------

    def start(self) -> "Run":
        with self._lock:
            if not self._started:
                self._started = True
                self._emit("run_start", agent=self.agent, mode=self.config.mode,
                           config_hash=self.config.hash, version=__version__,
                           metadata=to_plain(self.metadata) or None)
        return self

    def end(self, status: str = "completed") -> None:
        """Close the run. Called by the context manager; idempotent."""
        with self._lock:
            if self._ended:
                return
            self._ended = True
            if self.stopped is not None:
                status = "stopped"
            outcome, source = self._outcome, self._outcome_source
            if outcome is None:
                # A guess, and marked as one: finished without an error.
                source = "heuristic"
                outcome = {"completed": "ok", "error": "failed"}.get(status, "unknown")
            self._emit("run_end", status=status, outcome=outcome, outcome_source=source,
                       steps=self.engine.step, tokens=self.engine.tokens,
                       cost_usd=round(self.engine.cost, 6),
                       would_stop=self.first_stop.mechanism if self.first_stop else None,
                       warnings=self.warn_count)

    def __enter__(self) -> "Run":
        self.start()
        self._tokens.append(_current.set(self))
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._tokens:
            _current.reset(self._tokens.pop())
        if exc_type is None:
            self.end("completed")
        elif issubclass(exc_type, Stopped):
            self.end("stopped")
        else:
            self.end("error")
        return False

    async def __aenter__(self) -> "Run":
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return self.__exit__(exc_type, exc, tb)

    def outcome(self, value: str) -> None:
        """Label the run: ok, failed or unknown. Replay uses the label to tell
        a stop that saved money from one that broke good work."""
        if value not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {value!r}")
        with self._lock:
            self._outcome, self._outcome_source = value, "developer"

    # -- the agent loop ---------------------------------------------------

    def check_step(self) -> Verdict:
        """Call at the top of each agent turn, before the model call."""
        with self._lock:
            self._raise_if_stopped()
            verdict = self._open_step()
            self._handle(verdict)
            return verdict

    def record_usage(self, usage: Any, model: str | None = None) -> float | None:
        """Record a model call from an SDK usage object. Returns its cost in
        USD, or None when the model has no known price."""
        u = parse_usage(usage)
        return self.record_model_call(u.input_tokens, u.output_tokens, model=model,
                                      cache_read_tokens=u.cache_read_tokens,
                                      cache_write_tokens=u.cache_write_tokens)

    def record_model_call(self, input_tokens: int, output_tokens: int, model: str | None = None,
                          cost_usd: float | None = None, cache_read_tokens: int = 0,
                          cache_write_tokens: int = 0, same_step: bool = False) -> float | None:
        """Record one model call. With ``same_step=True`` the tokens are added
        to the current step instead of opening a new one: for an SDK that
        reports one model response in parts."""
        from .pricing import Usage

        u = Usage(int(input_tokens), int(output_tokens), int(cache_read_tokens),
                  int(cache_write_tokens))
        price_known = True
        if cost_usd is None:
            price = find_price(model, self.config.prices)
            if price is None:
                price_known = False
                if not self.config.strict_pricing:
                    self.guard._warn_unpriced(model)
            else:
                cost_usd = cost_of(u, price)
        with self._lock:
            # A model call with no check_step before it opens its step now.
            # The step's budget check still sees the spend before this call.
            step_verdict = ALLOW if (self._step_open or same_step) else self._open_step()
            verdict = self.engine.model_call(u.total, cost_usd, model, price_known)
            self._step_open = False
            self._emit("model_call", step=self.engine.step, model=model,
                       input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                       cache_read_tokens=u.cache_read_tokens,
                       cache_write_tokens=u.cache_write_tokens,
                       cost_usd=None if cost_usd is None else round(cost_usd, 8),
                       price_known=price_known)
            self._handle(strongest([step_verdict, verdict]))
        return cost_usd

    def before_model_call(self) -> None:
        """Open a step if the agent did not call check_step. Used by wrap()."""
        with self._lock:
            self._raise_if_stopped()
            if not self._step_open:
                self._handle(self._open_step())

    def before_tool(self, tool: str, args: Any = None) -> Verdict:
        """Call before running a tool. In enforce mode raises Stopped instead
        of letting a call through to a dependency that is down."""
        with self._lock:
            self._raise_if_stopped()
            fields: dict[str, Any] = {"step": self.engine.step, "tool": tool,
                                      "call_fp": call_print(tool, args),
                                      "args_shape": shape(args if args is not None else {})}
            if self.config.raw_text:
                fields["args"] = to_plain(args)
            self._emit("tool_call", **fields)
            verdict = self.engine.before_tool(tool, now=time.time())
            self._handle(verdict)
            return verdict

    def after_tool(self, tool: str, args: Any = None, result: Any = None,
                   error: BaseException | None = None, latency_ms: float | None = None) -> Verdict:
        """Call after a tool returns (result=) or fails (error=)."""
        ok = error is None
        call_fp = call_print(tool, args)
        fields: dict[str, Any] = {"tool": tool, "call_fp": call_fp, "ok": ok}
        if ok:
            fields["result_fp"] = result_print(tool, args, result,
                                               self.config.result_volatile_fields)
            fields["result_shape"] = shape(result)
            if self.config.raw_text:
                fields["result"] = canonical(to_plain(result))[:RAW_LIMIT]
        else:
            fields["error_status"] = status_of(error)
            fields["error_type"] = type(error).__name__
            if self.config.raw_text:
                fields["error"] = str(error)[:RAW_LIMIT]
        if latency_ms is not None:
            fields["latency_ms"] = round(latency_ms, 1)
        with self._lock:
            self._emit("tool_result", step=self.engine.step, **fields)
            verdict = self.engine.after_tool(tool, call_fp, ok, fields.get("result_fp"),
                                             fields.get("error_status"), fields.get("error_type"),
                                             now=time.time())
            self._handle(verdict)
            return verdict

    def notices(self) -> list[dict[str, str]]:
        """Pending notices as user messages, to append before the next model
        call. Always empty in shadow mode."""
        return [{"role": "user", "content": t} for t in self.notice_texts()]

    def notice_texts(self) -> list[str]:
        with self._lock:
            out, self._notices = self._notices, []
            for text in out:
                self._emit("notice", step=self.engine.step, text=text)
            return out

    # -- internals --------------------------------------------------------

    def _open_step(self) -> Verdict:
        verdict = self.engine.start_step(now=time.time())
        self._step_open = True
        self._emit("step", step=self.engine.step)
        return verdict

    def _emit(self, ev: str, **fields: Any) -> None:
        if not self._started and ev != "run_start":
            self.start()
        self.guard._writer.write({"schema": 1, "run_id": self.run_id, "ev": ev,
                                  "ts": round(time.time(), 3), **fields})

    def _raise_if_stopped(self) -> None:
        if self.stopped is not None and self.enforce:
            raise Stopped(self.stopped, self)

    def _handle(self, verdict: Verdict) -> None:
        if verdict.decision is Decision.ALLOW:
            return
        self._emit("verdict", step=self.engine.step, decision=verdict.decision.value,
                   mechanism=verdict.mechanism, reason=verdict.reason, enforced=self.enforce)
        if verdict.warns:
            self.warn_count += 1
            if self.enforce:
                text = self.config.notice.format(reason=verdict.reason)
                if text not in self._notices:
                    self._notices.append(text)
        elif self.first_stop is None:
            self.first_stop = verdict
        if verdict.stops and self.enforce:
            self.stopped = verdict
        event = StopEvent(self.run_id, self.agent, verdict.decision.value, verdict.mechanism,
                          verdict.reason, self.enforce, self.engine.step, self.summary())
        self.guard._callback("on_verdict", event)
        if verdict.stops and self.enforce:
            self.guard._callback("on_stop", event)
            raise Stopped(verdict, self)
