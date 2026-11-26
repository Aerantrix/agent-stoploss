"""Replay: what a config would have done to runs already recorded.

The same Engine that judges a live run is fed the recorded facts in order.
The first stop is where the run would have ended; the spend up to that point
is what it would have cost, the rest is the saving.

The replay cannot know how the agent would have reacted to a warning, so it
assumes the run went on as recorded until the first stop.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from .config import Config
from .engine import Engine, Verdict
from .journal import read_runs


@dataclass
class RunResult:
    run_id: str
    agent: str | None
    outcome: str
    outcome_source: str
    steps: int
    tokens: int
    cost_usd: float
    minimum_tokens: int
    stopped: bool = False
    mechanism: str | None = None
    reason: str = ""
    stop_step: int | None = None
    steps_with: int = 0
    tokens_with: int = 0
    cost_with: float = 0.0
    warnings: int = 0

    @property
    def excess(self) -> int:
        return max(0, self.tokens - self.minimum_tokens)

    @property
    def saved_tokens(self) -> int:
        return self.tokens - self.tokens_with

    @property
    def saved_cost(self) -> float:
        return self.cost_usd - self.cost_with

    @property
    def broke_good_run(self) -> bool:
        return self.stopped and self.outcome == "ok"

    @property
    def missed(self) -> bool:
        """Not ok, spent above the minimum, and not stopped."""
        return not self.stopped and self.outcome != "ok" and self.excess > 0


def replay_run(events: list[dict[str, Any]], config: Config,
               trace: list[dict[str, Any]] | None = None) -> RunResult:
    """Replay one run. With ``trace``, append one entry per warn or stop:
    the index of the event it came at, and the verdict."""
    engine = Engine(config)

    def note(i: int, v: Verdict) -> None:
        if trace is not None and v.decision.value != "allow":
            trace.append({"i": i, "decision": v.decision.value, "mechanism": v.mechanism,
                          "reason": v.reason, "step": engine.step})

    first = events[0]
    start = next((e for e in events if e["ev"] == "run_start"), {})
    end = next((e for e in reversed(events) if e["ev"] == "run_end"), None)

    # Totals as recorded, from the model calls themselves.
    tokens = cost = 0
    steps = 0
    minimum = 0
    for e in events:
        if e["ev"] == "model_call":
            t = _tokens(e)
            tokens += t
            cost += e.get("cost_usd") or 0.0
            if e.get("step", 0) <= config.minimum_steps:
                minimum += t
        if e["ev"] in ("step", "model_call"):
            steps = max(steps, e.get("step", 0))

    if end is not None:
        outcome = end.get("outcome") or "unknown"
        source = end.get("outcome_source") or "heuristic"
    else:
        outcome, source = "unknown", "missing"

    res = RunResult(first["run_id"], start.get("agent"), outcome, source, steps, tokens,
                    round(cost, 8), minimum, steps_with=steps, tokens_with=tokens,
                    cost_with=round(cost, 8))

    spent_tokens = 0
    spent_cost = 0.0
    for i, e in enumerate(events):
        ev = e["ev"]
        now = e.get("ts", 0.0)
        verdict: Verdict | None = None
        if ev == "step" or (ev == "model_call" and e.get("step", 0) > engine.step):
            stop_before = engine.step
            verdict = engine.start_step(now)
            note(i, verdict)
            if verdict.stops:
                _stop(res, verdict, engine.step, stop_before, spent_tokens, spent_cost)
                return res
            if verdict.warns:
                res.warnings += 1
        if ev == "model_call":
            t = _tokens(e)
            spent_tokens += t
            spent_cost += e.get("cost_usd") or 0.0
            verdict = engine.model_call(t, e.get("cost_usd"), e.get("model"),
                                        e.get("price_known", True))
        elif ev == "tool_call":
            verdict = engine.before_tool(e["tool"], now)
        elif ev == "tool_result":
            verdict = engine.after_tool(e["tool"], e["call_fp"], bool(e.get("ok")),
                                        e.get("result_fp"), e.get("error_status"),
                                        e.get("error_type"), now)
        else:
            continue
        note(i, verdict)
        if verdict.stops:
            _stop(res, verdict, engine.step, engine.step, spent_tokens, spent_cost)
            return res
        if verdict.warns:
            res.warnings += 1
    return res


def _tokens(e: dict[str, Any]) -> int:
    return sum(int(e.get(k) or 0) for k in
               ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"))


def _stop(res: RunResult, v: Verdict, step: int, steps_done: int, tokens: int, cost: float) -> None:
    res.stopped = True
    res.mechanism = v.mechanism
    res.reason = v.reason
    res.stop_step = step
    res.steps_with = steps_done
    res.tokens_with = tokens
    res.cost_with = round(cost, 8)


@dataclass
class Report:
    config_hash: str
    runs: list[RunResult] = field(default_factory=list)

    @property
    def stopped(self) -> list[RunResult]:
        return [r for r in self.runs if r.stopped]

    @property
    def broke(self) -> list[RunResult]:
        return [r for r in self.runs if r.broke_good_run]

    @property
    def missed(self) -> list[RunResult]:
        return [r for r in self.runs if r.missed]

    def by_mechanism(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.stopped:
            out[r.mechanism or "?"] = out.get(r.mechanism or "?", 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def totals(self) -> dict[str, Any]:
        tokens = sum(r.tokens for r in self.runs)
        saved = sum(r.saved_tokens for r in self.runs)
        excess = sum(r.excess for r in self.runs if r.outcome != "ok")
        saved_in_excess = sum(max(0, r.tokens - max(r.tokens_with, r.minimum_tokens))
                              for r in self.runs if r.outcome != "ok")
        return {
            "runs": len(self.runs),
            "labelled": sum(1 for r in self.runs if r.outcome_source == "developer"),
            "would_stop": len(self.stopped),
            "tokens": tokens,
            "cost_usd": round(sum(r.cost_usd for r in self.runs), 6),
            "saved_tokens": saved,
            "saved_cost_usd": round(sum(r.saved_cost for r in self.runs), 6),
            "excess_tokens_not_ok": excess,
            "saved_in_excess_tokens": saved_in_excess,
            "broke_ok_runs": len(self.broke),
            "missed_runs": len(self.missed),
            "by_mechanism": self.by_mechanism(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {"config_hash": self.config_hash, "totals": self.totals(),
                "runs": [{**asdict(r), "excess": r.excess, "saved_tokens": r.saved_tokens}
                         for r in self.runs]}


def replay(paths: Iterable[str | os.PathLike], config: Config) -> Report:
    return replay_runs(read_runs(paths), config)


def replay_runs(runs: Mapping[str, list[dict[str, Any]]], config: Config) -> Report:
    """Replay runs already read with journal.read_runs()."""
    report = Report(config.hash)
    for events in runs.values():
        report.runs.append(replay_run(events, config))
    return report


def suggest_max_tokens(report: Report) -> int | None:
    """A run-budget ceiling that no run labelled ok would have hit: 1.5 x the
    most expensive ok run, rounded up to a thousand. None without ok runs."""
    ok = [r.tokens for r in report.runs if r.outcome == "ok"]
    if not ok:
        return None
    return int(-(-max(ok) * 1.5 // 1000) * 1000)
