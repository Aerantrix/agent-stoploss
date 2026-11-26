"""The whole workflow on a scripted agent: no API key, no network.

    python examples/offline_demo.py [log_dir]

1. Four agent runs in shadow mode: one healthy, three stuck in the usual ways.
2. agent-stoploss replay on the journal they wrote.
3. The same four runs in enforce mode.

The "model" is a script that behaves like a stubborn agent: it retries until
it has tried twelve times, rewording or varying its call as real agents do.
"""

from __future__ import annotations

import sys
import tempfile

from agent_stoploss import StopLoss, Stopped
from agent_stoploss.cli import main as cli


class HTTPError(Exception):
    def __init__(self, status: int):
        super().__init__(f"HTTP {status}")
        self.status_code = status


class ServiceDown(Exception):
    pass


def make_tools(guard: StopLoss, mode: str):
    @guard.tool
    def analytics(domain: str, period: str = "30d"):
        if mode == "auth":
            raise HTTPError(401)
        if mode == "down":
            raise ServiceDown("connection refused")
        if mode == "empty":
            return {"domain": domain, "period": period, "results": []}
        n = int("".join(c for c in domain if c.isdigit()) or 0)
        return {"domain": domain, "period": period, "traffic": 20_000 + 7_919 * n % 90_000}

    return analytics


def agent(guard: StopLoss, mode: str) -> str:
    """One run. Each step: a model call (scripted tokens), then a tool call."""
    analytics = make_tools(guard, mode)
    plan = {
        "healthy": [f"site{i}.com" for i in range(1, 11)],     # ten sites, done
        "auth": ["example.com"] * 12,                            # same call after a 401
        "down": [f"example.com?try={i}" for i in range(12)],     # varies the call
        "empty": [f"exampl{'e' * i}.com" for i in range(1, 13)], # rewords the query
    }[mode]
    with guard.run(agent=f"report-{mode}") as run:
        try:
            for domain in plan:
                run.check_step()
                run.record_model_call(2_400 + 150 * run.step, 180, model="claude-haiku-5-5")
                try:
                    analytics(domain)
                except (HTTPError, ServiceDown):
                    pass  # the agent reads the error and tries again
                run.notices()  # would be appended to the next model call
            run.outcome("ok" if mode == "healthy" else "failed")
            return f"finished after {run.step} steps, {run.tokens:,} tokens"
        except Stopped as stop:
            run.outcome("failed")
            return f"STOPPED at step {run.step}: {stop.reason}"


def main(log_dir: str) -> None:
    modes = ("healthy", "auth", "down", "empty")

    print("== 1. shadow mode: nothing is stopped, everything is recorded\n")
    shadow = StopLoss({"mode": "shadow", "log": {"dir": log_dir}})
    for m in modes:
        print(f"  {m:<8} {agent(shadow, m)}")

    print(f"\n== 2. agent-stoploss replay {log_dir}\n")
    cli(["replay", log_dir])

    print("\n== 3. enforce mode\n")
    stops = []
    enforce = StopLoss({"mode": "enforce", "log": {"dir": None}}, on_stop=stops.append)
    for m in modes:
        print(f"  {m:<8} {agent(enforce, m)}")
    print(f"\n  on_stop was called {len(stops)} times")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="stoploss-demo-"))
