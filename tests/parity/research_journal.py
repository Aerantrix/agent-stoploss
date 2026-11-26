"""Turn the recorded runs of the Agent Fuel study into agent-stoploss journals.

The study recorded each run as a trace of model replies and token counts.
Its analysis replayed the controller over those traces by parsing each reply
and calling the stub tool again (evaluation/analyse.py, replay()). This module
does the same walk and writes what a live agent-stoploss run would have
journalled, so the library's replay can be checked against results/replay.csv.

Needs the study repository; see RESEARCH_REPO below.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from agent_stoploss.errors import status_of
from agent_stoploss.fingerprint import call_print, result_print

# A checkout of the Agent Fuel study; the parity tests skip without one.
RESEARCH_REPO = Path(os.environ.get("AGENT_FUEL_REPO", "")) if os.environ.get("AGENT_FUEL_REPO") else None

# Mechanism names in the study and in the library.
MECHANISMS = {
    "run_budget": "run_budget",
    "loop_detector_call_fingerprint": "loop_call",
    "loop_detector_result_fingerprint": "loop_result",
    "circuit_breaker": "circuit_breaker",
    "circuit_breaker_sliding_window": "circuit_breaker",
    "error_classifier": "terminal_error",
}

# The two study configurations that matter for the library:
#   G  Agent Fuel: the three mechanisms plus a 50,000-token run budget
#   M  the three mechanisms alone, no run budget (the library default)
CONFIGS = {
    "G": {"run_budget": {"max_tokens": 50_000}},
    "M": {},
}


def available() -> bool:
    return RESEARCH_REPO is not None and \
        (RESEARCH_REPO / "console" / "data" / "runs.jsonl").exists() and \
        (RESEARCH_REPO / "results" / "replay.csv").exists()


def _research_imports():
    root = str(RESEARCH_REPO)
    if root not in sys.path:
        sys.path.insert(0, root)
    from agent import real_prompts  # noqa: E402
    from agent.runner import UnknownTool, parse_action  # noqa: E402
    from sandbox.fake_tools import build_tools  # noqa: E402

    return build_tools, parse_action, UnknownTool, real_prompts


def load_runs() -> list[dict]:
    """The runs the study analysed: evaluation/analyse.py, load()."""
    scenarios = json.loads((RESEARCH_REPO / "evaluation" / "scenarios.json").read_text())
    by_id = {s["id"]: s for s in scenarios["scenarios"]}
    latest: dict[str, dict] = {}
    path = RESEARCH_REPO / "console" / "data" / "runs.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r or r.get("arm") != "A" or r.get("scenario_id") not in by_id:
            continue
        latest[r["id"]] = r
    for r in latest.values():
        r["_scenario"] = by_id[r["scenario_id"]]
    return list(latest.values())


def journal_of(run: dict) -> tuple[list[dict], bool]:
    """The journal a shadow-mode run would have written for this trace, and
    whether the study's own replay gave up on the trace early.

    The study's replay stops reading a trace at the first reply its parser
    cannot read and reports the run as not stopped, without looking at the
    remaining steps (five runs of scenario 13 whose first reply holds several
    tool calls in a format the adapter does not read). For those runs the
    study's numbers are not a replay result, so they cannot be compared.
    """
    build_tools, parse_action, UnknownTool, real_prompts = _research_imports()
    tools = build_tools(run["_scenario"])
    cond = str(run.get("condition", ""))
    adapter = real_prompts.ADAPTERS[cond[:3]]() if cond[:3] in real_prompts.ADAPTERS else None
    read = adapter.parse if adapter else parse_action
    rid = run["id"]

    def ev(name, **fields):
        return {"schema": 1, "run_id": rid, "ev": name, "ts": 0.0, **fields}

    events = [ev("run_start", agent=f"scenario-{run['scenario_id']}", mode="shadow")]
    steps = 0
    truncated = False
    for i, step in enumerate(run["trace"]):
        n = i + 1
        steps = n
        events.append(ev("step", step=n))
        events.append(ev("model_call", step=n, model=run["model"], input_tokens=step["in"],
                         output_tokens=step["out"], cost_usd=0.0, price_known=True))
        action = read(step["reply"])
        if action is None:
            truncated = n < len(run["trace"])
            break
        name, args = action
        events.append(ev("tool_call", step=n, tool=name, call_fp=call_print(name, args)))
        tool = tools.get(name)
        res = {"step": n, "tool": name, "call_fp": call_print(name, args)}
        if tool is None:
            exc = UnknownTool(name)
            res.update(ok=False, error_status=status_of(exc), error_type=type(exc).__name__)
        else:
            try:
                result = tool.get_metrics(**args)
            except Exception as exc:  # the stub raises on purpose; so does a bad argument
                res.update(ok=False, error_status=status_of(exc), error_type=type(exc).__name__)
            else:
                res.update(ok=True, result_fp=result_print(name, args, result,
                                                           ("timestamp", "request_id", "trace_id")))
        events.append(ev("tool_result", **res))
    group = run["_scenario"]["group"]
    # The study's healthy scenarios (control and reference) are the runs a
    # stop must not touch; label them ok. Refusal scenarios cannot succeed.
    outcome = "ok" if group in ("C", "R") else "failed"
    events.append(ev("run_end", status="completed", outcome=outcome, outcome_source="developer",
                     steps=steps, tokens=run["tokens"]))
    return events, truncated


def expected_rows() -> dict[tuple[str, str], dict]:
    """results/replay.csv rows for configurations G and M, by (run_id, code)."""
    import csv

    out = {}
    with open(RESEARCH_REPO / "results" / "replay.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["code"] in CONFIGS:
                out[(row["run_id"], row["code"])] = row
    return out


def journal_from_tool_calls(run: dict) -> list[dict]:
    """A journal for the runs the study's parser could not read, from the
    structured replies themselves: {"text": ..., "tool_calls": [...]}. The live
    runner executed only the first call of a reply, and so does this."""
    build_tools, _, _, _ = _research_imports()
    tools = build_tools(run["_scenario"])
    rid = run["id"]
    events = [{"schema": 1, "run_id": rid, "ev": "run_start", "ts": 0.0}]
    for n, step in enumerate(run["trace"], start=1):
        events.append({"schema": 1, "run_id": rid, "ev": "model_call", "step": n, "ts": 0.0,
                       "input_tokens": step["in"], "output_tokens": step["out"]})
        try:
            calls = json.loads(step["reply"]).get("tool_calls") or []
        except (ValueError, AttributeError):
            calls = []
        if not calls:
            continue
        name, args = calls[0]["name"], calls[0].get("args") or {}
        fp = call_print(name, args)
        events.append({"schema": 1, "run_id": rid, "ev": "tool_call", "tool": name, "call_fp": fp,
                       "ts": 0.0})
        res = {"schema": 1, "run_id": rid, "ev": "tool_result", "tool": name, "call_fp": fp, "ts": 0.0}
        try:
            result = tools[name].get_metrics(**args)
        except Exception as exc:
            res.update(ok=False, error_status=status_of(exc), error_type=type(exc).__name__)
        else:
            res.update(ok=True, result_fp=result_print(name, args, result))
        events.append(res)
    return events
