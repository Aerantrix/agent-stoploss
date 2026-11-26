"""Write the Agent Fuel study's runs as one agent-stoploss journal file."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "parity"))
import research_journal as rj  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else "research_logs")
out.mkdir(parents=True, exist_ok=True)
with open(out / "stoploss-research.jsonl", "w", encoding="utf-8") as f:
    for i, run in enumerate(rj.load_runs()):
        events, truncated = rj.journal_of(run)
        if truncated:
            events = rj.journal_from_tool_calls(run)
            group = run["_scenario"]["group"]
            events.append({"run_id": run["id"], "ev": "run_end", "status": "completed",
                           "outcome": "ok" if group in ("C", "R") else "failed",
                           "outcome_source": "developer"})
        for e in events:
            e["schema"] = 1
            e["ts"] = 1763900000 + i * 60
        events[0]["agent"] = "scenario-%02d" % run["scenario_id"]
        for e in events:
            f.write(json.dumps(e) + "\n")
print(out)
