"""agent-stoploss command line.

    agent-stoploss replay LOGS... [--config stoploss.yaml] [--json]
    agent-stoploss dashboard LOGS... [--config stoploss.yaml] [--port 8740]
    agent-stoploss check-config stoploss.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import Config, ConfigError
from .replay import Report, replay, suggest_max_tokens

NAMES = {
    "run_budget": "run budget",
    "loop_call": "same call repeated",
    "loop_result": "same result repeated",
    "circuit_breaker": "tool keeps failing",
    "terminal_error": "error retrying cannot fix",
    "pricing": "unpriced model (strict_pricing)",
}


def _load_config(path: str | None) -> Config:
    if path:
        return Config.from_file(path)
    default = Path("stoploss.yaml")
    return Config.from_file(default) if default.exists() else Config.from_dict({})


def _money(x: float) -> str:
    return f"${x:,.2f}" if x >= 0.01 or x == 0 else f"${x:,.4f}"


def format_report(report: Report, limit: int = 20) -> str:
    t = report.totals()
    lines = []
    w = lines.append
    w(f"agent-stoploss replay  (config {report.config_hash})")
    w("")
    w(f"Runs                      {t['runs']:>8}   "
      f"({t['labelled']} labelled by the developer, the rest by heuristic)")
    w(f"Recorded spend            {t['tokens']:>8,} tokens   {_money(t['cost_usd'])}")
    w(f"Would have stopped        {t['would_stop']:>8}")
    for mech, n in t["by_mechanism"].items():
        w(f"  {NAMES.get(mech, mech):<24}{n:>6}")
    w(f"Would have saved          {t['saved_tokens']:>8,} tokens   {_money(t['saved_cost_usd'])}")
    if t["excess_tokens_not_ok"]:
        share = 100 * t["saved_in_excess_tokens"] / t["excess_tokens_not_ok"]
        w(f"  of the excess in runs not ok: {share:.0f}%")
    w("")

    broke = report.broke
    missed = sorted(report.missed, key=lambda r: -r.excess)
    width = max([len(r.run_id) for r in broke[:limit] + missed[:limit]] + [10])
    w(f"Runs labelled ok that would have been stopped: {len(broke)}")
    for r in broke[:limit]:
        w(f"  {r.run_id:<{width}}  step {r.stop_step:<4} {NAMES.get(r.mechanism, r.mechanism)}: {r.reason}")
    if len(broke) > limit:
        w(f"  ... and {len(broke) - limit} more (use --json)")
    w("")

    w(f"Runs not ok, above the minimum, not stopped by anything: {len(missed)}")
    for r in missed[:limit]:
        w(f"  {r.run_id:<{width}}  {r.tokens:>9,} tokens, {r.excess:>9,} above the minimum  [{r.outcome}]")
    if len(missed) > limit:
        w(f"  ... and {len(missed) - limit} more (use --json)")

    suggestion = suggest_max_tokens(report)
    if suggestion:
        w("")
        w(f"Suggested run_budget.max_tokens: {suggestion:,} "
          "(1.5 x the most expensive run labelled ok)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-stoploss",
                                     description="Stop AI agents from burning tokens on broken tools.")
    parser.add_argument("--version", action="version", version=f"agent-stoploss {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("replay", help="what a config would have done to recorded runs")
    p.add_argument("logs", nargs="+", help="journal files or directories")
    p.add_argument("--config", help="config file (default: ./stoploss.yaml if present)")
    p.add_argument("--json", action="store_true", help="full result as JSON")
    p.add_argument("--limit", type=int, default=20, help="rows per list in the text report")

    d = sub.add_parser("dashboard", help="local page to explore runs and tune thresholds")
    d.add_argument("logs", nargs="+", help="journal files or directories")
    d.add_argument("--config", help="config to start from (default: ./stoploss.yaml if present)")
    d.add_argument("--port", type=int, default=8740)
    d.add_argument("--no-browser", action="store_true", help="do not open a browser")

    c = sub.add_parser("check-config", help="validate a config file")
    c.add_argument("path")

    args = parser.parse_args(argv)
    try:
        if args.command == "check-config":
            cfg = Config.from_file(args.path)
            print(f"{args.path}: OK  (mode {cfg.mode}, config {cfg.hash})")
            if cfg.mode == "enforce" and cfg.max_tokens is None and cfg.max_cost_usd is None:
                print("note: enforce mode with no run budget; only the mechanisms can stop a run")
            return 0

        cfg = _load_config(args.config)
        if args.command == "dashboard":
            from .dashboard import serve

            path = args.config or ("stoploss.yaml" if Path("stoploss.yaml").exists() else None)
            serve(args.logs, cfg, path, port=args.port, open_browser=not args.no_browser)
            return 0
        report = replay(args.logs, cfg)
        if args.json:
            json.dump(report.to_dict(), sys.stdout, indent=2, default=str)
            print()
        else:
            print(format_report(report, args.limit))
        return 0
    except ConfigError as e:
        print(str(e), file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"not found: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"agent-stoploss: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
