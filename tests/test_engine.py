"""The mechanisms on the failure patterns they exist for, and on the honest
work they must leave alone."""

from agent_stoploss import Config
from agent_stoploss.engine import Engine
from agent_stoploss.fingerprint import call_print, result_print


def engine(**settings):
    return Engine(Config.from_dict(settings))


def ok(e, tool, args, result, now=0.0):
    e.before_tool(tool, now)
    return e.after_tool(tool, call_print(tool, args), True, result_print(tool, args, result), now=now)


def fail(e, tool, args, status=None, etype="Error", now=0.0):
    v = e.before_tool(tool, now)
    if v.stops:
        return v
    return e.after_tool(tool, call_print(tool, args), False, None, status, etype, now=now)


def test_terminal_error_stops_at_first_401():
    v = fail(engine(), "analytics", {"domain": "a"}, status=401)
    assert v.stops and v.mechanism == "terminal_error"


def test_transient_error_does_not_stop_at_once():
    v = fail(engine(), "analytics", {"domain": "a"}, status=503)
    assert not v.stops


def test_reworded_query_same_empty_answer():
    e = engine()
    verdicts = [ok(e, "search", {"q": q}, {"q": q, "results": []}) for q in ("a", "b", "c")]
    assert verdicts[1].warns and verdicts[1].mechanism == "loop_result"
    assert verdicts[2].stops and verdicts[2].mechanism == "loop_result"


def test_same_call_repeated():
    e = engine()
    vs = [fail(e, "analytics", {"domain": "a"}) for _ in range(3)]
    assert vs[2].stops and vs[2].mechanism == "loop_call"


def test_service_down_with_varied_calls_trips_the_breaker():
    e = engine()
    vs = [fail(e, "analytics", {"domain": f"d{i}"}) for i in range(5)]
    assert [v.stops for v in vs] == [False, False, False, False, True]
    assert vs[4].mechanism == "circuit_breaker"


def test_breaker_warns_before_the_call_that_would_trip_it():
    e = engine()
    for i in range(3):
        fail(e, "analytics", {"domain": f"d{i}"})
    v = e.before_tool("analytics")
    assert v.warns and v.mechanism == "circuit_breaker"


def test_flapping_service_is_left_alone():
    e = engine()
    for i in range(30):
        if i % 2:
            v = fail(e, "analytics", {"domain": f"d{i}"})
        else:
            v = ok(e, "analytics", {"domain": f"d{i}"}, {"domain": f"d{i}", "traffic": i})
        assert not v.stops, i


def test_forty_domains_and_twenty_pages_are_left_alone():
    e = engine()
    for i in range(1, 41):
        assert not ok(e, "analytics", {"domain": f"site{i}.com"}, {"domain": f"site{i}.com", "traffic": i * 7}).stops
    for p in range(1, 21):
        assert not ok(e, "list", {"page": p}, {"page": p, "rows": [p * 10 + k for k in range(5)]}).stops


def test_breaker_cooldown_lets_a_probe_through():
    e = engine(circuit_breaker={"cooldown_seconds": 60})
    for i in range(4):
        fail(e, "t", {"i": i}, now=0)
    assert e.before_tool("t", now=1).stops
    assert e.before_tool("t", now=30).stops
    assert not e.before_tool("t", now=61).stops


def test_per_tool_overrides():
    e = engine(tools={"job_status": {"loop_detector": False}})
    for _ in range(10):
        assert not ok(e, "job_status", {"id": 7}, {"state": "running"}).stops
    e2 = engine(tools={"flaky": {"circuit_breaker": {"window": 10, "max_failures": 8}}})
    for i in range(7):
        assert not fail(e2, "flaky", {"i": i}).stops


def test_budget_off_by_default():
    e = engine()
    e.model_call(10_000_000, 100.0)
    assert not e.start_step().stops


def test_budget_warns_then_stops_at_step_start():
    e = engine(run_budget={"max_tokens": 1000})
    assert not e.start_step().warns
    e.model_call(850, 0)
    assert e.start_step().warns
    e.model_call(200, 0)
    v = e.start_step()
    assert v.stops and v.mechanism == "run_budget"


def test_cost_budget():
    e = engine(run_budget={"max_cost_usd": 1.0})
    e.model_call(10, 1.2)
    assert e.start_step().stops


def test_strict_pricing():
    assert engine(strict_pricing=True).model_call(10, None, "x", price_known=False).stops
    assert not engine().model_call(10, None, "x", price_known=False).stops


def test_terminal_beats_loop_in_the_same_call():
    e = engine()
    fail(e, "t", {"a": 1}, status=503)
    fail(e, "t", {"a": 1}, status=503)
    v = fail(e, "t", {"a": 1}, status=404)
    assert v.mechanism == "terminal_error"
