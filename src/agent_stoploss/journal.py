"""The event journal: one JSON line per event, schema 1.

Events:

    run_start    run_id, agent, mode, config_hash, metadata
    step         step                       (a new agent turn begins)
    model_call   step, model, input_tokens, output_tokens, cache_read_tokens,
                 cache_write_tokens, cost_usd, price_known
    tool_call    step, tool, call_fp, args_shape   (+ args with raw_text)
    tool_result  step, tool, call_fp, ok, result_fp, error_status, error_type,
                 latency_ms, result_shape   (+ result / error with raw_text)
    verdict      step, decision, mechanism, reason, enforced   (warn and stop only)
    notice       step, text                  (a notice handed to the agent)
    run_end      status, outcome, outcome_source, steps, tokens, cost_usd

Every line carries ``schema``, ``run_id``, ``ev`` and ``ts`` (Unix seconds).
Raw arguments and results are left out unless ``log.raw_text`` is on: logs
end up in places prompts and customer data should not.
"""

from __future__ import annotations

import json
import os
import threading
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

SCHEMA = 1

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(path, threading.Lock())


class JournalWriter:
    """Appends events to ``<dir>/stoploss-YYYY-MM-DD.jsonl`` (UTC date).

    Each event is a single write of one line, so several processes can share
    a directory. ``sink`` replaces the file, for tests and custom transports.
    """

    def __init__(self, directory: str | os.PathLike | None = None,
                 sink: Callable[[dict[str, Any]], None] | None = None):
        self.directory = Path(directory) if directory is not None else None
        self.sink = sink
        if self.directory is not None and sink is None:
            self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, ts: float) -> Path | None:
        if self.directory is None:
            return None
        day = time.strftime("%Y-%m-%d", time.gmtime(ts))
        return self.directory / f"stoploss-{day}.jsonl"

    def write(self, event: dict[str, Any]) -> None:
        if self.sink is not None:
            self.sink(event)
            return
        path = self.path_for(event["ts"])
        if path is None:
            return
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
        with _lock_for(str(path)):
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)


class NullWriter(JournalWriter):
    def __init__(self):
        super().__init__(None)

    def write(self, event: dict[str, Any]) -> None:
        pass


def _files(paths: Iterable[str | os.PathLike]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            out += sorted(p.rglob("*.jsonl"))
        elif p.exists():
            out.append(p)
        else:
            raise FileNotFoundError(p)
    return out


def read_events(paths: Iterable[str | os.PathLike]) -> Iterator[dict[str, Any]]:
    """Every well-formed event in the given files and directories, in file order."""
    skipped = 0
    other_schema = 0
    for path in _files(paths):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    skipped += 1
                    continue
                if not isinstance(ev, dict) or "run_id" not in ev or "ev" not in ev:
                    skipped += 1
                    continue
                if ev.get("schema") != SCHEMA:
                    other_schema += 1
                    continue
                yield ev
    if skipped:
        warnings.warn(f"agent-stoploss: skipped {skipped} unreadable journal lines", stacklevel=2)
    if other_schema:
        warnings.warn(f"agent-stoploss: skipped {other_schema} events with an unknown schema",
                      stacklevel=2)


def read_runs(paths: Iterable[str | os.PathLike]) -> "OrderedDict[str, list[dict[str, Any]]]":
    """Events grouped by run, runs in the order they first appear.

    Within a run, events keep the order they were written in. Runs that
    span midnight are joined across the two daily files.
    """
    runs: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for ev in read_events(paths):
        runs.setdefault(ev["run_id"], []).append(ev)
    return runs
