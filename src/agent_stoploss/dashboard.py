"""The local dashboard: a "what if" sandbox over recorded runs.

    agent-stoploss dashboard ./stoploss_logs [--config stoploss.yaml]

A standard-library HTTP server on 127.0.0.1 and one static page. It reads the
journal, replays it under whatever thresholds the page sends, and hands back
a config file to download. It never controls a live agent, never writes to
disk and never talks to anything but the browser on this machine.

The numbers come from the same replay function as ``agent-stoploss replay``,
so the terminal and the page always agree.
"""

from __future__ import annotations

import json
import os
import threading
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from typing import Any, Iterable

import yaml

from . import __version__
from .config import DEFAULTS, Config, ConfigError
from .journal import read_runs
from .replay import replay_run, replay_runs, suggest_max_tokens

DEFAULT_PORT = 8740
MAX_BODY = 1_000_000


class State:
    """The logs in memory, and the config the dashboard was started with."""

    def __init__(self, paths: Iterable[str | os.PathLike], config: Config, config_path: str | None):
        self.paths = [str(p) for p in paths]
        self.config = config
        self.config_path = config_path
        self.lock = threading.Lock()
        self.runs: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self.reload()

    def reload(self) -> None:
        runs = read_runs(self.paths)
        with self.lock:
            self.runs = runs

    def meta(self) -> dict[str, Any]:
        with self.lock:
            runs = self.runs
        tools: dict[str, int] = {}
        agents: dict[str, int] = {}
        for events in runs.values():
            for e in events:
                if e["ev"] == "tool_call":
                    tools[e["tool"]] = tools.get(e["tool"], 0) + 1
                elif e["ev"] == "run_start" and e.get("agent"):
                    agents[e["agent"]] = agents.get(e["agent"], 0) + 1
        return {
            "version": __version__,
            "logs": self.paths,
            "config_path": self.config_path,
            # tools in their long form ({"loop_detector": {"enabled": false}}),
            # which is what the page edits
            "config": _jsonable({**self.config.source, "tools": self.config.tools}),
            "defaults": _jsonable(DEFAULTS),
            "runs": len(runs),
            "tools": dict(sorted(tools.items(), key=lambda kv: -kv[1])),
            "agents": dict(sorted(agents.items())),
        }

    def report(self, settings: dict[str, Any]) -> dict[str, Any]:
        config = Config.from_dict(settings)
        with self.lock:
            runs = self.runs
        report = replay_runs(runs, config)
        out = report.to_dict()
        starts = {rid: next((e.get("ts") for e in ev if e["ev"] == "run_start"), ev[0].get("ts"))
                  for rid, ev in runs.items()}
        for row in out["runs"]:
            row["ts"] = starts.get(row["run_id"])
        out["suggested_max_tokens"] = suggest_max_tokens(report)
        return out

    def run_detail(self, run_id: str, settings: dict[str, Any]) -> dict[str, Any] | None:
        config = Config.from_dict(settings)
        with self.lock:
            events = self.runs.get(run_id)
        if events is None:
            return None
        trace: list[dict[str, Any]] = []
        res = replay_run(events, config, trace=trace)
        return {"events": events, "verdicts": trace,
                "result": {**res.__dict__, "excess": res.excess, "saved_tokens": res.saved_tokens}}


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=sorted))


def to_yaml(settings: dict[str, Any]) -> str:
    """The config as a file, keeping only what differs from the defaults."""
    config = Config.from_dict(settings)  # validate first
    diff = _diff(config.source, DEFAULTS)
    diff["mode"] = config.mode
    head = (f"# agent-stoploss config, exported from the dashboard (config {config.hash}).\n"
            "# Settings not listed here keep their defaults.\n"
            "# Check before deploying:  agent-stoploss check-config stoploss.yaml\n\n")
    ordered = {"mode": diff.pop("mode"), **diff}
    return head + yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True, default_flow_style=False)


def _diff(value: dict[str, Any], default: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, v in value.items():
        d = default.get(key)
        if key in ("tools", "prices"):
            if v:
                out[key] = v
        elif isinstance(v, dict) and isinstance(d, dict):
            sub = _diff(v, d)
            if sub:
                out[key] = sub
        elif v != d:
            out[key] = v
    return out


class Handler(BaseHTTPRequestHandler):
    state: State
    port: int

    server_version = f"agent-stoploss/{__version__}"

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet
        pass

    # Only a browser on this machine, at this address, may talk to us: a page
    # elsewhere cannot reach 127.0.0.1 under another host name (DNS rebinding)
    # and cannot send JSON without a preflight we never answer.
    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        return host in (f"127.0.0.1:{self.port}", f"localhost:{self.port}")

    def _send(self, status: int, body: bytes, ctype: str, extra: dict[str, str] | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'none'; script-src 'self' 'unsafe-inline'; style-src 'unsafe-inline'; "
                         "connect-src 'self'; img-src 'self' data:; base-uri 'none'; form-action 'none'; "
                         "frame-ancestors 'none'")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data: Any) -> None:
        self._send(status, json.dumps(data, default=str).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        if not self._host_ok():
            return self._json(403, {"error": "forbidden host"})
        if self.path in ("/", "/index.html"):
            page = resources.files("agent_stoploss").joinpath("static/dashboard.html").read_bytes()
            return self._send(200, page, "text/html; charset=utf-8")
        if self.path == "/api/meta":
            return self._json(200, self.state.meta())
        return self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if not self._host_ok():
            return self._json(403, {"error": "forbidden host"})
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return self._json(415, {"error": "JSON only"})
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self._json(413, {"error": "too large"})
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except ValueError:
            return self._json(400, {"error": "bad JSON"})
        settings = body.get("config") or {}
        try:
            if self.path == "/api/replay":
                return self._json(200, self.state.report(settings))
            if self.path == "/api/run":
                detail = self.state.run_detail(str(body.get("run_id")), settings)
                return self._json(200, detail) if detail else self._json(404, {"error": "no such run"})
            if self.path == "/api/export":
                return self._json(200, {"yaml": to_yaml(settings)})
            if self.path == "/api/reload":
                self.state.reload()
                return self._json(200, self.state.meta())
        except ConfigError as e:
            return self._json(422, {"problems": e.problems})
        return self._json(404, {"error": "not found"})


def serve(paths: Iterable[str | os.PathLike], config: Config, config_path: str | None = None,
          port: int = DEFAULT_PORT, open_browser: bool = True) -> None:
    state = State(paths, config, config_path)
    handler = type("BoundHandler", (Handler,), {"state": state, "port": port})
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"agent-stoploss dashboard: {url}  ({len(state.runs)} runs; Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
