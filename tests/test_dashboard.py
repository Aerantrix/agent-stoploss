import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import pytest
import yaml

from agent_stoploss import Config
from agent_stoploss.dashboard import Handler, State, to_yaml

from .test_replay_cli import record


@pytest.fixture
def server(tmp_path):
    logs = record(tmp_path)
    state = State([logs], Config.from_dict({"tools": {"analytics": {"circuit_breaker": False}}}), "stoploss.yaml")
    srv = ThreadingHTTPServer(("127.0.0.1", 0), type("H", (Handler,), {"state": state, "port": 0}))
    port = srv.server_address[1]
    srv.RequestHandlerClass.port = port
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield port
    srv.shutdown()
    srv.server_close()


def call(port, method, path, body=None, host=None, ctype="application/json"):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    headers = {"Host": host or f"127.0.0.1:{port}"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = ctype
    conn.request(method, path, body=data, headers=headers)
    r = conn.getresponse()
    raw = r.read()
    return r.status, dict(r.getheaders()), raw


def test_page_and_meta(server):
    status, headers, page = call(server, "GET", "/")
    assert status == 200 and b"agent-stoploss" in page
    assert "default-src 'none'" in headers["Content-Security-Policy"]
    assert b"https://" not in page and b"http://" not in page.replace(b"http://www.w3.org/2000/svg", b"")
    status, _, raw = call(server, "GET", "/api/meta")
    meta = json.loads(raw)
    assert meta["runs"] == 3 and "analytics" in meta["tools"]
    assert meta["config"]["tools"] == {"analytics": {"circuit_breaker": {"enabled": False}}}
    assert set(meta["agents"]) == {"auth", "down", "ok"}


def test_replay_matches_cli_numbers(server, tmp_path):
    from agent_stoploss.replay import replay

    status, _, raw = call(server, "POST", "/api/replay", {"config": {}})
    data = json.loads(raw)
    expected = replay([tmp_path / "logs"], Config.from_dict({})).totals()
    assert status == 200 and data["totals"] == json.loads(json.dumps(expected))
    assert all(r["ts"] for r in data["runs"])
    assert data["suggested_max_tokens"] == 17000


def test_invalid_config_lists_problems(server):
    status, _, raw = call(server, "POST", "/api/replay", {"config": {"loop_detector": {"threshold": 1}}})
    assert status == 422 and "loop_detector.threshold" in json.loads(raw)["problems"][0]


def test_run_detail(server):
    _, _, raw = call(server, "POST", "/api/replay", {"config": {}})
    auth = next(r for r in json.loads(raw)["runs"] if r["agent"] == "auth")
    status, _, raw = call(server, "POST", "/api/run", {"run_id": auth["run_id"], "config": {}})
    d = json.loads(raw)
    assert status == 200 and d["result"]["mechanism"] == "terminal_error"
    stop = [v for v in d["verdicts"] if v["decision"] == "stop"]
    assert len(stop) == 1 and d["events"][stop[0]["i"]]["ev"] == "tool_result"
    assert call(server, "POST", "/api/run", {"run_id": "nope", "config": {}})[0] == 404


def test_export(server):
    status, _, raw = call(server, "POST", "/api/export",
                          {"config": {"mode": "enforce", "loop_detector": {"threshold": 4}}})
    text = json.loads(raw)["yaml"]
    assert status == 200 and text.startswith("# agent-stoploss config")
    data = yaml.safe_load(text)
    assert data == {"mode": "enforce", "loop_detector": {"threshold": 4}}
    assert Config.from_dict(data).loop_threshold == 4


def test_guards(server):
    assert call(server, "GET", "/api/meta", host="evil.example:80")[0] == 403
    assert call(server, "POST", "/api/replay", {"config": {}}, ctype="text/plain")[0] == 415
    assert call(server, "GET", "/../../etc/passwd")[0] == 404
    assert call(server, "POST", "/api/replay", {"config": {}}, host=f"localhost:{server}")[0] == 200


def test_to_yaml_keeps_tools_and_round_trips():
    text = to_yaml({"tools": {"job": {"loop_detector": {"enabled": False}}}, "run_budget": {"max_tokens": 9000}})
    data = yaml.safe_load(text)
    cfg = Config.from_dict(data)
    assert not cfg.rules_for("job").loop_enabled and cfg.max_tokens == 9000
    assert data["mode"] == "shadow"
