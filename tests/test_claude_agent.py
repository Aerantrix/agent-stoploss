"""The Claude Agent SDK integration, driven the way the SDK drives it: hook
callbacks with the SDK's input dicts, and the messages query() yields.

The SDK itself is not needed: the messages are stand-ins with the SDK's class
names and fields, and a fake query() calls the hooks from the options it is
given, as the CLI would."""

from dataclasses import dataclass, field
from typing import Any

import pytest

from agent_stoploss import Stopped
from agent_stoploss.claude_agent import StopLossHooks, guarded_query


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict


@dataclass
class TextBlock:
    text: str


@dataclass
class AssistantMessage:
    content: list
    model: str = "claude-haiku-5-5"
    usage: dict | None = None
    message_id: str | None = None
    parent_tool_use_id: str | None = None


@dataclass
class ResultMessage:
    subtype: str = "success"
    is_error: bool = False
    usage: dict | None = None
    model_usage: dict | None = None


@dataclass
class Options:
    """Stands in for ClaudeAgentOptions: only the hooks matter here."""
    hooks: dict[str, Any] = field(default_factory=dict)


def usage(inp, out, read=0, write=0):
    return {"input_tokens": inp, "output_tokens": out,
            "cache_read_input_tokens": read, "cache_creation_input_tokens": write}


def pre(tool, args, tid):
    return {"hook_event_name": "PreToolUse", "session_id": "s", "transcript_path": "", "cwd": "",
            "tool_name": tool, "tool_input": args, "tool_use_id": tid}


def post(tool, args, tid, response):
    return {"hook_event_name": "PostToolUse", "session_id": "s", "transcript_path": "", "cwd": "",
            "tool_name": tool, "tool_input": args, "tool_use_id": tid, "tool_response": response,
            "duration_ms": 12}


def failed(tool, args, tid, error, interrupt=False):
    return {"hook_event_name": "PostToolUseFailure", "session_id": "s", "transcript_path": "", "cwd": "",
            "tool_name": tool, "tool_input": args, "tool_use_id": tid, "error": error,
            "is_interrupt": interrupt}


def turn(i, tool="mcp__analytics__report", args=None, tokens=(1000, 50)):
    tid = f"tu{i}"
    msg = AssistantMessage([TextBlock("checking"), ToolUseBlock(tid, tool, args or {"domain": "example.com"})],
                           usage=usage(*tokens), message_id=f"msg{i}")
    return msg, tid


class Agent:
    """A scripted SDK session: per turn, yields the model's message, then
    runs the tool hooks; stops early when a hook says continue: false."""

    def __init__(self, turns):
        # (message, tool, args, tool_use_id, outcome); outcome is ("ok", response) or ("fail", text)
        self.turns = turns
        self.answers = []

    def query(self, *, prompt, options):
        hooks = options.hooks

        async def run_hooks(event, data, tid):
            out = {}
            for cb in hooks.get(event, []):
                out = await cb(data, tid, {"signal": None})
                self.answers.append((event, out))
                if out:
                    return out
            return out

        async def gen():
            for msg, tool, args, tid, (kind, value) in self.turns:
                yield msg
                out = await run_hooks("PreToolUse", pre(tool, args, tid), tid)
                if out.get("continue_") is False:
                    break
                if kind == "ok":
                    out = await run_hooks("PostToolUse", post(tool, args, tid, value), tid)
                else:
                    out = await run_hooks("PostToolUseFailure", failed(tool, args, tid, value), tid)
                if out.get("continue_") is False:
                    break
            yield ResultMessage()

        return gen()


def script(n, outcome, tool="mcp__analytics__report", args=None):
    rows = []
    for i in range(n):
        a = args(i) if callable(args) else (args or {"domain": "example.com"})
        msg, tid = turn(i, tool, a)
        rows.append((msg, tool, a, tid, outcome(i) if callable(outcome) else outcome))
    return rows


@pytest.fixture
def options(monkeypatch):
    # options() builds HookMatchers from the SDK; without the SDK, hand the
    # raw callbacks to the fake agent instead.
    def raw_options(self, options=None):
        return Options(hooks={event: [cb] for event, cb in self.callbacks().items()})

    monkeypatch.setattr(StopLossHooks, "options", raw_options)


async def collect(gen):
    return [m async for m in gen]


async def test_enforce_stops_on_terminal_error(make_guard, sink, options):
    guard = make_guard("enforce")
    agent = Agent(script(10, ("fail", "Error: HTTP 401 Unauthorized")))
    seen = []
    with pytest.raises(Stopped) as info:
        async for m in guarded_query(guard, prompt="report", agent="report", query=agent.query):
            seen.append(m)
    assert info.value.mechanism == "terminal_error"
    # The first failed call is enough; the agent never takes a second turn.
    assert len([m for m in seen if isinstance(m, AssistantMessage)]) == 1
    event, out = agent.answers[-1]
    assert event == "PostToolUseFailure" and out["continue_"] is False
    assert out["stopReason"].startswith("agent-stoploss stopped this run")
    end = sink.of("run_end")[0]
    assert end["status"] == "stopped" and end["steps"] == 1
    assert sink.of("tool_result")[0]["error_status"] == 401


async def test_enforce_stops_a_loop_and_denies_the_next_call(make_guard, sink, options):
    guard = make_guard("enforce", notice="note: {reason}.")
    agent = Agent(script(10, ("ok", {"domain": "example.com", "rows": []})))
    with pytest.raises(Stopped) as info:
        await collect(guarded_query(guard, prompt="report", query=agent.query))
    assert info.value.mechanism in ("loop_call", "loop_result")
    # A warning reached the model first, as additional context.
    contexts = [out["hookSpecificOutput"].get("additionalContext") for _, out in agent.answers
                if out.get("hookSpecificOutput", {}).get("additionalContext")]
    assert contexts and contexts[0].startswith("note:")
    stop_answer = [out for _, out in agent.answers if out.get("continue_") is False][0]
    assert stop_answer["stopReason"].startswith("agent-stoploss stopped this run")
    assert len(sink.of("notice")) >= 1


async def test_pre_tool_denies_once_stopped(make_guard, options):
    guard = make_guard("enforce")
    async with guard.run() as run:
        sl = StopLossHooks(run)
        cb = sl.callbacks()
        msg, tid = turn(0)
        sl.observe(msg)
        await cb["PreToolUse"](pre("t", {}, tid), tid, None)
        out = await cb["PostToolUseFailure"](failed("t", {}, tid, "403 Forbidden"), tid, None)
        assert out["continue_"] is False
        out = await cb["PreToolUse"](pre("t", {}, "tu9"), "tu9", None)
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
        with pytest.raises(Stopped):
            sl.finish()


async def test_shadow_records_and_changes_nothing(make_guard, sink, options):
    guard = make_guard("shadow")
    agent = Agent(script(6, ("fail", "Error: HTTP 401 Unauthorized")))
    msgs = await collect(guarded_query(guard, prompt="report", query=agent.query))
    assert len(msgs) == 7                                   # all six turns and the result
    assert all(out == {} for _, out in agent.answers)       # the agent never notices
    stops = [v for v in sink.of("verdict") if v["decision"] == "stop"]
    assert stops and stops[0]["mechanism"] == "terminal_error" and stops[0]["enforced"] is False
    end = sink.of("run_end")[0]
    assert (end["status"], end["would_stop"], end["steps"]) == ("completed", "terminal_error", 6)


async def test_usage_counted_once_per_response(make_guard, sink, options):
    guard = make_guard("shadow")
    async with guard.run() as run:
        sl = StopLossHooks(run)
        # One response reported in two parts with the same message id, then
        # a second response.
        sl.observe(AssistantMessage([TextBlock("a")], usage=usage(1000, 10, read=500), message_id="m1"))
        sl.observe(AssistantMessage([ToolUseBlock("t1", "x", {})], usage=usage(1000, 80, read=500), message_id="m1"))
        sl.observe(AssistantMessage([TextBlock("b")], usage=usage(1200, 40), message_id="m2"))
        sl.observe(ResultMessage(usage=usage(2200, 120)))
        assert run.step == 2
        assert run.tokens == 1000 + 80 + 500 + 1200 + 40
    calls = sink.of("model_call")
    assert [c["step"] for c in calls] == [1, 1, 2]
    assert sum(c["output_tokens"] for c in calls) == 120


async def test_hook_before_its_message_shares_the_step(make_guard, sink, options):
    guard = make_guard("shadow")
    async with guard.run() as run:
        sl = StopLossHooks(run)
        cb = sl.callbacks()
        # The CLI may run the hook before the message reaches the reader.
        await cb["PreToolUse"](pre("x", {"a": 1}, "t1"), "t1", None)
        await cb["PreToolUse"](pre("y", {"a": 2}, "t2"), "t2", None)    # same response, second call
        sl.observe(AssistantMessage([ToolUseBlock("t1", "x", {"a": 1}), ToolUseBlock("t2", "y", {"a": 2})],
                                    usage=usage(500, 30), message_id="m1"))
        await cb["PostToolUse"](post("x", {"a": 1}, "t1", {"ok": 1}), "t1", None)
        # Next response: message first, then its hook.
        sl.observe(AssistantMessage([ToolUseBlock("t3", "x", {"a": 3})], usage=usage(600, 30), message_id="m2"))
        await cb["PreToolUse"](pre("x", {"a": 3}, "t3"), "t3", None)
        assert run.step == 2
    assert [e["step"] for e in sink.of("tool_call")] == [1, 1, 2]
    assert [e["step"] for e in sink.of("model_call")] == [1, 2]


async def test_mcp_error_result_is_a_failure(make_guard, sink, options):
    guard = make_guard("shadow")
    async with guard.run() as run:
        sl = StopLossHooks(run)
        cb = sl.callbacks()
        resp = {"content": [{"type": "text", "text": "upstream said 503 Service Unavailable"}], "is_error": True}
        await cb["PostToolUse"](post("mcp__a__b", {}, "t1", resp), "t1", None)
        fine = {"content": [{"type": "text", "text": "fine"}]}
        await cb["PostToolUse"](post("mcp__a__b", {}, "t2", fine), "t2", None)
    r = sink.of("tool_result")
    assert (r[0]["ok"], r[0]["error_status"], r[0]["error_type"]) == (False, 503, "ToolFailed")
    assert r[1]["ok"] is True


async def test_interrupt_is_not_a_tool_failure(make_guard, sink, options):
    guard = make_guard("enforce")
    async with guard.run() as run:
        cb = StopLossHooks(run).callbacks()
        out = await cb["PostToolUseFailure"](failed("Bash", {}, "t1", "interrupted", interrupt=True), "t1", None)
    assert out == {} and sink.of("tool_result") == []


async def test_a_fault_in_a_hook_does_not_break_the_agent(make_guard, options):
    guard = make_guard("enforce")
    async with guard.run() as run:
        cb = StopLossHooks(run).callbacks()

        class Unprintable:
            def __repr__(self):
                raise RuntimeError("boom")

            __str__ = __repr__

        with pytest.warns(RuntimeWarning, match="hook failed"):
            out = await cb["PostToolUse"](post("t", {"x": Unprintable()}, "t1", None), "t1", None)
        assert out == {}


async def test_budget_stop_raised_from_observe(make_guard, sink, options):
    guard = make_guard("enforce", run_budget={"max_tokens": 2500})
    agent = Agent(script(10, lambda i: ("ok", {"page": i, "rows": [i]}), args=lambda i: {"page": i}))
    seen = []
    with pytest.raises(Stopped) as info:
        async for m in guarded_query(guard, prompt="report", query=agent.query):
            seen.append(m)
    assert info.value.mechanism == "run_budget"
    # Each turn costs 1,050 tokens; the fourth would start above 2,500.
    assert len([m for m in seen if isinstance(m, AssistantMessage)]) == 3
    assert sink.of("run_end")[0]["status"] == "stopped"


async def test_reader_leaving_early_ends_the_run_as_interrupted(make_guard, sink, options):
    guard = make_guard("shadow")
    agent = Agent(script(5, lambda i: ("ok", {"page": i}), args=lambda i: {"page": i}))
    gen = guarded_query(guard, prompt="report", query=agent.query)
    async for _ in gen:
        break
    await gen.aclose()
    end = sink.of("run_end")[0]
    assert (end["status"], end["outcome"]) == ("interrupted", "unknown")


def test_options_merge_with_the_sdk():
    sdk = pytest.importorskip("claude_agent_sdk")
    from agent_stoploss import StopLoss

    async def mine(data, tid, ctx):
        return {}

    guard = StopLoss({"mode": "shadow", "log": {"dir": None}})
    sl = StopLossHooks(guard.run())
    opts = sl.options(sdk.ClaudeAgentOptions(model="claude-haiku-5-5",
                                             hooks={"PreToolUse": [sdk.HookMatcher(hooks=[mine])]}))
    assert opts.model == "claude-haiku-5-5"
    assert [len(m.hooks) for m in opts.hooks["PreToolUse"]] == [1, 1]
    assert opts.hooks["PreToolUse"][1].hooks == [mine]       # theirs kept, after ours
    assert set(opts.hooks) == {"PreToolUse", "PostToolUse", "PostToolUseFailure"}


def model_usage(inp, out, read=0, write=0, model="claude-haiku-5-5"):
    return {model: {"inputTokens": inp, "outputTokens": out, "cacheReadInputTokens": read,
                    "cacheCreationInputTokens": write, "costUSD": 0.0}}


async def test_result_totals_correct_the_message_snapshots(make_guard, sink, options):
    """Recorded from a live run (claude-agent-sdk 0.2.189, Haiku 5.5): the
    messages showed 24 + 7 output tokens and 4 input; the result's model_usage,
    which also covers a call Claude Code made for itself, says 245 and 1,209.
    The SDK priced the query at $0.00033599."""
    guard = make_guard("shadow")
    async with guard.run() as run:
        sl = StopLossHooks(run)
        sl.observe(AssistantMessage([ToolUseBlock("t1", "x", {})], usage=usage(2, 24, read=539, write=481),
                                    message_id="m1"))
        sl.observe(AssistantMessage([TextBlock("thinking")], usage=usage(2, 7, read=1020, write=135), message_id="m2"))
        sl.observe(AssistantMessage([TextBlock("done")], usage=usage(2, 7, read=1020, write=135), message_id="m2"))
        sl.observe(ResultMessage(usage=usage(4, 226, read=1559, write=616),
                                 model_usage=model_usage(1209, 245, read=1559, write=616)))
        assert run.step == 2
        assert run.tokens == 1209 + 245 + 1559 + 616
        assert run.cost_usd == pytest.approx(0.00033599, abs=1e-9)
    assert sink.of("model_call")[-1]["step"] == 2      # the correction goes to the last step


async def test_session_totals_are_cumulative(make_guard, options):
    """ClaudeSDKClient: model_usage grows over the session; each query adds
    only its own share."""
    guard = make_guard("shadow")
    async with guard.run() as run:
        sl = StopLossHooks(run)
        sl.observe(AssistantMessage([TextBlock("one")], usage=usage(2, 1), message_id="a"))
        sl.observe(ResultMessage(model_usage=model_usage(2, 3)))
        assert run.tokens == 5
        sl.observe(AssistantMessage([TextBlock("two")], usage=usage(2, 1), message_id="b"))
        sl.observe(ResultMessage(model_usage=model_usage(4, 6)))
        assert run.tokens == 10


async def test_dated_model_name_is_not_counted_twice(make_guard, options):
    guard = make_guard("shadow")
    async with guard.run() as run:
        sl = StopLossHooks(run)
        sl.observe(AssistantMessage([TextBlock("a")], model="claude-haiku-5-5-20261001", usage=usage(100, 5),
                                    message_id="a"))
        sl.observe(ResultMessage(model_usage=model_usage(100, 40)))
        assert run.tokens == 140
