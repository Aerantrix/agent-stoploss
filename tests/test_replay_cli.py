import json

import pytest

from agent_stoploss import Config, StopLoss, Stopped
from agent_stoploss.cli import main
from agent_stoploss.journal import read_runs
from agent_stoploss.replay import replay, replay_run

from .conftest import HTTPError, Unavailable


def record(tmp_path, mode="shadow", **settings):
    """Three runs written to real journal files: a 401, an outage, a healthy one."""
    guard = StopLoss({**settings, "mode": mode, "log": {"dir": str(tmp_path / "logs")}})

    @guard.tool
    def analytics(domain, kind="ok"):
        if kind == "auth":
            raise HTTPError(401)
        if kind == "down":
            raise Unavailable()
        return {"domain": domain, "traffic": int(domain[4:-4]) * 1000 + 7}

    for kind, steps, outcome in (("auth", 6, "failed"), ("down", 8, "failed"), ("ok", 10, "ok")):
        try:
            with guard.run(agent=kind) as run:
                run.outcome(outcome)
                for i in range(steps):
                    run.check_step()
                    run.record_model_call(1000, 100, model="claude-haiku-5-5")
                    try:
                        analytics(f"site{i}.com", kind)
                    except (HTTPError, Unavailable):
                        pass
        except Stopped:
            pass
    return tmp_path / "logs"


def test_replay_reproduces_the_live_shadow_verdicts(tmp_path):
    logs = record(tmp_path)
    runs = read_runs([logs])
    assert len(runs) == 3
    report = replay([logs], Config.from_dict({}))
    by_agent = {r.agent: r for r in report.runs}
    for events in runs.values():
        live = next(e for e in events if e["ev"] == "run_end")["would_stop"]
        agent = events[0]["agent"]
        assert by_agent[agent].mechanism == live
    assert by_agent["auth"].mechanism == "terminal_error" and by_agent["auth"].stop_step == 1
    assert by_agent["down"].mechanism == "circuit_breaker" and by_agent["down"].stop_step == 5
    assert not by_agent["ok"].stopped
    assert by_agent["auth"].tokens == 6600 and by_agent["auth"].tokens_with == 1100
    assert report.broke == [] and report.missed == []


def test_replay_with_other_thresholds(tmp_path):
    logs = record(tmp_path)
    loose = Config.from_dict({"circuit_breaker": {"window": 10, "max_failures": 10},
                              "error_classifier": False, "run_budget": {"max_tokens": 5000}})
    by_agent = {r.agent: r for r in replay([logs], loose).runs}
    assert by_agent["auth"].mechanism == "run_budget"
    assert by_agent["ok"].stopped  # a tight budget breaks the healthy run
    assert replay([logs], loose).broke[0].agent == "ok"


def test_enforced_run_replays_to_the_same_stop(tmp_path):
    logs = record(tmp_path, mode="enforce")
    report = replay([logs], Config.from_dict({}))
    assert {r.agent: r.mechanism for r in report.runs} == {
        "auth": "terminal_error", "down": "circuit_breaker", "ok": None}


def test_cli_replay_text_and_json(tmp_path, capsys):
    logs = record(tmp_path)
    assert main(["replay", str(logs)]) == 0
    out = capsys.readouterr().out
    assert "Would have stopped               2" in out
    assert "error retrying cannot fix" in out and "Suggested run_budget.max_tokens: 17,000" in out
    assert main(["replay", str(logs), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["totals"]["runs"] == 3 and data["totals"]["broke_ok_runs"] == 0


def test_cli_check_config(tmp_path, capsys):
    good = tmp_path / "good.yaml"
    good.write_text("mode: enforce\nloop_detector: {threshold: 4}\n")
    bad = tmp_path / "bad.yaml"
    bad.write_text("mode: enforce\nloop_detecter: {threshold: 4}\n")
    assert main(["check-config", str(good)]) == 0
    assert "OK" in capsys.readouterr().out
    assert main(["check-config", str(bad)]) == 2
    assert "loop_detecter: unknown setting" in capsys.readouterr().err


def test_bad_lines_are_skipped(tmp_path):
    f = tmp_path / "x.jsonl"
    good = {"schema": 1, "run_id": "r1", "ev": "run_start", "ts": 0}
    f.write_text("not json\n" + json.dumps(good) + "\n" + json.dumps({**good, "schema": 9}) + "\n")
    with pytest.warns(UserWarning):
        runs = read_runs([f])
    assert list(runs) == ["r1"] and len(runs["r1"]) == 1


def test_run_without_end_is_unknown():
    events = [{"schema": 1, "run_id": "r", "ev": "run_start", "ts": 0},
              {"schema": 1, "run_id": "r", "ev": "model_call", "step": 1, "input_tokens": 5,
               "output_tokens": 5, "ts": 0}]
    res = replay_run(events, Config.from_dict({}))
    assert (res.outcome, res.outcome_source, res.tokens) == ("unknown", "missing", 10)
