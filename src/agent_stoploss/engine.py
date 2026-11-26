"""The decision engine: facts in, verdicts out.

The engine never sees a tool, an exception or a model response, only the
facts the journal records about them: prints, statuses, token counts. That is
what lets ``agent-stoploss replay`` re-run a recorded run under different
thresholds and get exactly the verdicts a live run would have produced.

Mechanisms:

    run_budget       a token or cost ceiling for one run (off by default)
    loop_call        the same call repeated
    loop_result      the same answer returned, whatever the call
    circuit_breaker  too many failures among a tool's recent calls
    terminal_error   an error that retrying cannot fix (401, 404, ...)
    pricing          a model with no known price, under strict_pricing
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

from .config import Config
from .errors import classify


class Decision(str, Enum):
    ALLOW = "allow"
    WARN = "warn"
    STOP = "stop"


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    mechanism: str | None = None
    reason: str = ""

    @property
    def stops(self) -> bool:
        return self.decision is Decision.STOP

    @property
    def warns(self) -> bool:
        return self.decision is Decision.WARN


ALLOW = Verdict(Decision.ALLOW)


def strongest(verdicts) -> Verdict:
    """Stop over warn over allow; the first of equals wins."""
    best = ALLOW
    for v in verdicts:
        if v.stops:
            return v
        if v.warns and not best.warns:
            best = v
    return best


class Engine:
    """State of one run. Not thread-safe; the Run holding it takes a lock."""

    def __init__(self, config: Config):
        self.config = config
        self.step = 0
        self.tokens = 0
        self.cost = 0.0
        self._calls: dict[str, int] = {}
        self._results: dict[str, int] = {}
        self._history: dict[str, deque] = {}
        self._opened_at: dict[str, float] = {}

    # -- run budget -------------------------------------------------------

    def start_step(self, now: float = 0.0) -> Verdict:
        self.step += 1
        c = self.config
        if c.max_tokens is not None:
            if self.tokens >= c.max_tokens:
                return Verdict(Decision.STOP, "run_budget",
                               f"token ceiling reached: {self.tokens:,} of {c.max_tokens:,}")
        if c.max_cost_usd is not None:
            if self.cost >= c.max_cost_usd:
                return Verdict(Decision.STOP, "run_budget",
                               f"cost ceiling reached: ${self.cost:.2f} of ${c.max_cost_usd:.2f}")
        if c.max_tokens is not None and self.tokens >= c.budget_warn_at * c.max_tokens:
            return Verdict(Decision.WARN, "run_budget",
                           f"{self.tokens:,} of {c.max_tokens:,} tokens used")
        if c.max_cost_usd is not None and self.cost >= c.budget_warn_at * c.max_cost_usd:
            return Verdict(Decision.WARN, "run_budget",
                           f"${self.cost:.2f} of ${c.max_cost_usd:.2f} spent")
        return ALLOW

    def model_call(self, tokens: int, cost: float | None, model: str | None = None,
                   price_known: bool = True) -> Verdict:
        self.tokens += tokens
        self.cost += cost or 0.0
        if not price_known and self.config.strict_pricing:
            return Verdict(Decision.STOP, "pricing",
                           f"no price for model {model!r} and strict_pricing is on")
        return ALLOW

    # -- tools ------------------------------------------------------------

    def _window(self, tool: str, size: int) -> deque:
        w = self._history.get(tool)
        if w is None or w.maxlen != size:
            w = deque(w or (), maxlen=size)
            self._history[tool] = w
        return w

    def before_tool(self, tool: str, now: float = 0.0) -> Verdict:
        rules = self.config.rules_for(tool)
        if not rules.breaker_enabled:
            return ALLOW
        opened = self._opened_at.get(tool)
        if opened is not None:
            if now - opened < self.config.breaker_cooldown_seconds:
                return Verdict(Decision.STOP, "circuit_breaker",
                               f"{tool} is not answering; calls to it are paused")
            # Cooldown over: half-open, let one call through as a probe.
            del self._opened_at[tool]
            self._window(tool, rules.breaker_window).clear()
        window = self._window(tool, rules.breaker_window)
        failures = sum(window)
        if failures >= rules.breaker_max_failures:
            self._opened_at[tool] = now
            return Verdict(Decision.STOP, "circuit_breaker",
                           f"{tool} failed in {failures} of its last {len(window)} calls")
        if failures == rules.breaker_max_failures - 1:
            return Verdict(Decision.WARN, "circuit_breaker",
                           f"{tool} failed in {failures} of its last {len(window)} calls")
        return ALLOW

    def after_tool(self, tool: str, call_fp: str, ok: bool, result_fp: str | None = None,
                   error_status: int | None = None, error_type: str | None = None,
                   now: float = 0.0) -> Verdict:
        rules = self.config.rules_for(tool)
        c = self.config
        checks = []

        if rules.breaker_enabled:
            self._window(tool, rules.breaker_window).append(not ok)

        if not ok and rules.classifier_enabled:
            kind = classify(error_status, error_type, c.terminal_statuses,
                            c.transient_statuses, c.terminal_exceptions)
            if kind == "terminal":
                what = f"HTTP {error_status}" if error_status else (error_type or "error")
                checks.append(Verdict(Decision.STOP, "terminal_error",
                                      f"{tool} returned {what}; the same call cannot succeed"))

        if rules.loop_enabled:
            checks.append(self._observe(tool, call_fp, result_fp if ok else None,
                                        rules.loop_threshold))
        return strongest(checks)

    def _observe(self, tool: str, call_fp: str, result_fp: str | None, threshold: int) -> Verdict:
        # Errors count towards the call print only. A failing dependency is the
        # circuit breaker's evidence; printing every error text as a result
        # would stop a flaky service whose retries do succeed.
        calls = self._calls[call_fp] = self._calls.get(call_fp, 0) + 1
        results = 0
        if result_fp is not None:
            results = self._results[result_fp] = self._results.get(result_fp, 0) + 1

        if calls >= threshold:
            return Verdict(Decision.STOP, "loop_call",
                           f"the same {tool} call was made {calls} times")
        if results >= threshold:
            return Verdict(Decision.STOP, "loop_result",
                           f"{tool} returned the same result {results} times")
        if calls == threshold - 1:
            return Verdict(Decision.WARN, "loop_call",
                           f"the same {tool} call was made {calls} times")
        if results == threshold - 1:
            return Verdict(Decision.WARN, "loop_result",
                           f"{tool} returned the same result {results} times")
        return ALLOW
