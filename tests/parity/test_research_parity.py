"""The library's replay reproduces the Agent Fuel study's replay.

For every analysed run of the study and for configurations G (mechanisms plus
a 50,000-token budget) and M (mechanisms alone), the library must stop the
same runs, by the same mechanism, at the same step and spend as
results/replay.csv. Skipped when the study repository is not present.
"""

import json
from pathlib import Path

import pytest

from agent_stoploss import Config
from agent_stoploss.replay import replay_run

from . import research_journal as rj

pytestmark = pytest.mark.skipif(not rj.available(), reason="Agent Fuel study repository not found")

OUT = Path(__file__).resolve().parents[2] / "test-results"


def test_replay_matches_study():
    runs = rj.load_runs()
    expected = rj.expected_rows()
    compared, skipped, mismatches = 0, [], []
    per_mechanism: dict[str, int] = {}

    for run in runs:
        journal, truncated = rj.journal_of(run)
        if truncated:
            skipped.append(run["id"])
            continue
        for code, settings in rj.CONFIGS.items():
            row = expected[(run["id"], code)]
            res = replay_run(journal, Config.from_dict(settings))
            want = {
                "stopped": row["halted"] == "True",
                "mechanism": rj.MECHANISMS.get(row["mechanism"]) if row["mechanism"] else None,
                "tokens_with": int(row["tokens_with"]),
                "steps_with": int(row["steps_with"]),
            }
            got = {"stopped": res.stopped, "mechanism": res.mechanism,
                   "tokens_with": res.tokens_with, "steps_with": res.steps_with}
            compared += 1
            if res.stopped:
                key = f"{code}:{res.mechanism}"
                per_mechanism[key] = per_mechanism.get(key, 0) + 1
            if want != got:
                mismatches.append({"run_id": run["id"], "config": code, "study": want, "library": got})

    OUT.mkdir(exist_ok=True)
    summary = {
        "runs": len(runs),
        "comparisons": compared,
        "matching": compared - len(mismatches),
        "mismatches": mismatches,
        "skipped_runs": skipped,
        "skipped_reason": "the study's replay stopped reading these traces at an unparsed "
                          "first reply, so its row is not a replay result",
        "stops_by_config_and_mechanism": dict(sorted(per_mechanism.items())),
    }
    (OUT / "parity.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    assert len(runs) == 660
    assert len(skipped) <= 5, skipped
    assert not mismatches, f"{len(mismatches)} of {compared} differ; see test-results/parity.json"


def test_runs_the_study_could_not_replay_are_left_alone():
    """The five healthy 40-site runs the study's parser skipped: read from
    their structured replies, the default config does not stop them."""
    runs = {r["id"]: r for r in rj.load_runs()}
    skipped = [rid for rid, r in runs.items() if rj.journal_of(r)[1]]
    assert skipped
    for rid in skipped:
        res = replay_run(rj.journal_from_tool_calls(runs[rid]), Config.from_dict({}))
        assert runs[rid]["_scenario"]["group"] == "C"
        assert not res.stopped, (rid, res.mechanism, res.reason)
