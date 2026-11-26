import asyncio
import threading
import warnings

import pytest

from agent_stoploss import Stopped, ToolFailed, current_run

from .conftest import HTTPError, Unavailable


def test_shadow_changes_nothing_and_records_everything(make_guard, sink):
    guard = make_guard("shadow")

    @guard.tool
    def analytics(domain, period="30d"):
        raise HTTPError(401)

    with guard.run(agent="a") as run:
        for _ in range(3):
            run.check_step()
            run.record_model_call(100, 20, model="claude-haiku-5-5")
            with pytest.raises(HTTPError):
                analytics("x.com")
        assert run.notices() == []

    verdicts = sink.of("verdict")
    assert verdicts and all(not v["enforced"] for v in verdicts)
    assert verdicts[0]["mechanism"] == "terminal_error"
    end = sink.of("run_end")[0]
    assert end["status"] == "completed" and end["would_stop"] == "terminal_error"
    assert end["outcome"] == "ok" and end["outcome_source"] == "heuristic"
    assert [e["ev"] for e in sink[:4]] == ["run_start", "step", "model_call", "tool_call"]
    assert all(e["schema"] == 1 and e["run_id"] == run.run_id for e in sink)


def test_enforce_raises_stopped_and_calls_on_stop(make_guard, sink):
    stops = []
    guard = make_guard("enforce")
    guard.on_stop = stops.append
    calls = []

    @guard.tool
    def analytics(domain):
        calls.append(domain)
        raise HTTPError(401)

    with pytest.raises(Stopped) as e:
        with guard.run(agent="a") as run:
            run.check_step()
            analytics("x.com")
    assert e.value.mechanism == "terminal_error"
    assert e.value.summary["steps"] == 1
    assert isinstance(e.value.__context__, HTTPError)
    assert len(stops) == 1 and stops[0].enforced
    assert sink.of("run_end")[0]["status"] == "stopped"
    assert sink.of("run_end")[0]["outcome"] == "unknown"
    assert calls == ["x.com"]


def test_stop_is_sticky_and_open_breaker_spares_the_call(make_guard):
    guard = make_guard("enforce")
    calls = []

    @guard.tool
    def svc(i):
        calls.append(i)
        raise Unavailable()

    with guard.run() as run:
        for i in range(4):
            with pytest.raises(Unavailable):
                svc(i)
        with pytest.raises(Stopped):
            svc(4)
        assert calls == [0, 1, 2, 3]  # the fifth call never reached the service
        with pytest.raises(Stopped):
            run.check_step()


def test_warnings_become_notices_in_enforce(make_guard, sink):
    guard = make_guard("enforce", notice="note: {reason}.")

    @guard.tool
    def search(q):
        return {"q": q, "results": []}

    with guard.run() as run:
        search("a")
        search("b")
        notes = run.notices()
        assert notes == [{"role": "user", "content": "note: search returned the same result 2 times."}]
        assert run.notices() == []
    assert sink.of("notice")[0]["text"].startswith("note:")


def test_outcome_label(make_guard, sink):
    guard = make_guard()
    with guard.run() as run:
        run.outcome("failed")
    end = sink.of("run_end")[0]
    assert (end["outcome"], end["outcome_source"]) == ("failed", "developer")
    with pytest.raises(ValueError):
        run.outcome("great")


def test_exception_in_run_is_a_failed_run(make_guard, sink):
    guard = make_guard()
    with pytest.raises(KeyError):
        with guard.run():
            raise KeyError("x")
    end = sink.of("run_end")[0]
    assert (end["status"], end["outcome"]) == ("error", "failed")


def test_usage_objects_and_pricing(make_guard, sink):
    guard = make_guard()
    with guard.run() as run:
        cost = run.record_usage({"input_tokens": 1_000_000, "output_tokens": 0,
                                 "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
                                model="claude-opus-5-5")
        assert cost == pytest.approx(4.0)
        run.record_usage({"prompt_tokens": 100, "completion_tokens": 10,
                          "prompt_tokens_details": {"cached_tokens": 40}}, model="gpt-x")
    calls = sink.of("model_call")
    assert calls[1]["input_tokens"] == 60 and calls[1]["cache_read_tokens"] == 40
    assert calls[1]["price_known"] is False and calls[1]["cost_usd"] is None
    assert [e["step"] for e in sink.of("step")] == [1, 2]


def test_dated_and_bedrock_model_ids_are_priced(make_guard):
    guard = make_guard()
    with guard.run() as run:
        assert run.record_model_call(1_000_000, 0, model="claude-sonnet-5-5-20261001") == pytest.approx(2.0)
        assert run.record_model_call(1_000_000, 0, model="us.anthropic.claude-haiku-5-5") == pytest.approx(0.1)


def test_config_prices_override_builtin(make_guard):
    guard = make_guard(prices={"claude-opus-5-5": {"input": 1, "output": 1}})
    with guard.run() as run:
        assert run.record_model_call(1_000_000, 0, model="claude-opus-5-5") == pytest.approx(1.0)


def test_unknown_model_warns_once(make_guard):
    guard = make_guard()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with guard.run() as run:
            run.record_model_call(1, 1, model="mystery")
            run.record_model_call(1, 1, model="mystery")
    assert sum("mystery" in str(x.message) for x in w) == 1


def test_strict_pricing_stops_in_enforce(make_guard):
    guard = make_guard("enforce", strict_pricing=True)
    with pytest.raises(Stopped) as e:
        with guard.run() as run:
            run.record_model_call(1, 1, model="mystery")
    assert e.value.mechanism == "pricing"


def test_budget_in_enforce(make_guard):
    guard = make_guard("enforce", run_budget={"max_tokens": 1000})
    with pytest.raises(Stopped) as e:
        with guard.run() as run:
            for _ in range(20):
                run.check_step()
                run.record_model_call(300, 0, cost_usd=0)
    assert e.value.mechanism == "run_budget"
    assert e.value.summary["steps"] == 5 and e.value.summary["tokens"] == 1200


def test_tool_outside_a_run_passes_through(make_guard):
    guard = make_guard("enforce")

    @guard.tool
    def f(x):
        return x * 2

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert f(2) == 4
        assert f(3) == 6
    assert len(w) == 1


def test_tool_failed_marker_and_manual_hooks(make_guard, sink):
    guard = make_guard("enforce")
    with pytest.raises(Stopped):
        with guard.run() as run:
            run.before_tool("mcp_tool", {"a": 1})
            run.after_tool("mcp_tool", {"a": 1}, error=ToolFailed("denied", status=403))
    res = sink.of("tool_result")[0]
    assert (res["ok"], res["error_status"], res["error_type"]) == (False, 403, "ToolFailed")


def test_raw_text_off_by_default_and_on_by_flag(make_guard, sink):
    guard = make_guard()

    @guard.tool
    def lookup(email):
        return {"name": "Jane Customer"}

    with guard.run():
        lookup("jane@example.com")
    assert "jane" not in str(sink).lower()

    sink.clear()
    guard2 = make_guard(log={"dir": None, "raw_text": True})
    guard2._writer = guard._writer

    @guard2.tool
    def lookup2(email):
        return {"name": "Jane Customer"}

    with guard2.run():
        lookup2("jane@example.com")
    assert "jane@example.com" in str(sink) and "Jane Customer" in str(sink)


def test_callback_errors_do_not_break_the_agent(make_guard):
    guard = make_guard("shadow")
    guard.on_verdict = lambda ev: 1 / 0

    @guard.tool
    def t():
        raise HTTPError(404)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        with guard.run():
            with pytest.raises(HTTPError):
                t()


async def test_async_tools_and_concurrent_runs_do_not_mix(make_guard, sink):
    guard = make_guard("enforce")

    @guard.tool
    async def fetch(i):
        await asyncio.sleep(0)
        return {"i": i, "rows": [i]}

    async def agent(name, n):
        async with guard.run(agent=name) as run:
            for i in range(n):
                run.check_step()
                await fetch(i)
            assert current_run() is run
            return run.run_id, run.step

    results = await asyncio.gather(agent("a", 5), agent("b", 7))
    assert [r[1] for r in results] == [5, 7]
    ends = {e["run_id"]: e for e in sink.of("run_end")}
    assert {ends[rid]["steps"] for rid, _ in results} == {5, 7}
    assert current_run() is None


def test_threads_each_have_their_own_run(make_guard, sink):
    guard = make_guard()

    @guard.tool
    def work(i):
        return {"i": i}

    def agent(n):
        with guard.run() as run:
            for i in range(n):
                run.check_step()
                work(i)

    threads = [threading.Thread(target=agent, args=(n,)) for n in (3, 4, 5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(e["steps"] for e in sink.of("run_end")) == [3, 4, 5]
