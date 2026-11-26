"""Claude Agent SDK integration.

    from claude_agent_sdk import ClaudeAgentOptions
    from agent_stoploss import StopLoss, Stopped
    from agent_stoploss.claude_agent import guarded_query

    guard = StopLoss.from_file("stoploss.yaml")
    try:
        async for message in guarded_query(guard, prompt="...", options=ClaudeAgentOptions(...),
                                           agent="weekly-report"):
            ...
    except Stopped as stop:
        print(stop.reason)

The SDK runs the agent loop itself, so agent-stoploss joins it in two places:

- hooks before and after every tool call (PreToolUse, PostToolUse,
  PostToolUseFailure) see the calls, the results and the errors;
- the messages the SDK yields carry each model response's token usage.

The usage on an assistant message is a snapshot taken early in the response:
input and cache counts are final, output tokens are not, and calls Claude
Code makes for itself do not appear at all. Each ResultMessage carries the
true totals per model (``model_usage``, cumulative over a client session);
the difference is added to the current step there. Budget checks during the
run therefore see input in full and output late.

In enforce mode a stop denies the pending tool call and ends the agent's turn
(``continue: false``); guarded_query then raises Stopped. A stop decided after
a tool result (a 401, say) takes effect at the agent's next tool call: Claude
Code lets the model answer once more before the hooks can end the turn. Warnings reach the
model as additional context on the next hook. In shadow mode the hooks only
record and always return an empty answer.

``claude-agent-sdk`` is imported only when the hooks are built, so this module
loads without it.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, AsyncIterator, Awaitable, Callable

from .errors import status_from_text
from .run import Run, Stopped, ToolFailed

HookOutput = dict[str, Any]
Callback = Callable[[dict[str, Any], "str | None", Any], Awaitable[HookOutput]]

_USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


class StopLossHooks:
    """Hooks and message tracking for one run.

    For ``query()`` use guarded_query. With ``ClaudeSDKClient``, or to keep
    your own control of the loop:

        async with guard.run(agent="report") as run:
            sl = StopLossHooks(run)
            async with ClaudeSDKClient(options=sl.options(options)) as client:
                await client.query("...")
                async for message in client.receive_response():
                    sl.observe(message)
            sl.finish()
    """

    def __init__(self, run: Run):
        self.run = run
        self.stopped: Stopped | None = None
        self._usage: dict[str, dict[str, int]] = {}   # message id -> usage recorded so far
        self._tool_ids: set[str] = set()              # tool_use ids of recorded messages
        self._hook_step = False                       # a hook opened the step before its message arrived
        self._since_result: dict[str, dict[str, int]] = {}   # model -> usage recorded since the last result
        self._session_totals: dict[str, dict[str, int]] = {}  # model -> model_usage at the last result

    # -- wiring -----------------------------------------------------------

    def callbacks(self) -> dict[str, Callback]:
        """The hook callbacks by event name, without the SDK's wrappers."""
        return {"PreToolUse": self._pre_tool, "PostToolUse": self._post_tool,
                "PostToolUseFailure": self._tool_failed}

    def hooks(self) -> dict[str, list[Any]]:
        """``ClaudeAgentOptions.hooks`` for every tool."""
        from claude_agent_sdk import HookMatcher

        return {event: [HookMatcher(matcher=None, hooks=[cb])] for event, cb in self.callbacks().items()}

    def options(self, options: Any = None) -> Any:
        """A copy of ``options`` with these hooks added ahead of any already set."""
        from claude_agent_sdk import ClaudeAgentOptions

        options = options if options is not None else ClaudeAgentOptions()
        merged: dict[str, list[Any]] = {event: list(matchers) for event, matchers in (options.hooks or {}).items()}
        for event, matchers in self.hooks().items():
            merged[event] = matchers + merged.get(event, [])
        return dataclasses.replace(options, hooks=merged)

    # -- messages ---------------------------------------------------------

    def observe(self, message: Any) -> None:
        """Pass every message the SDK yields. Records model usage; in enforce
        mode raises Stopped when the spend crosses a limit."""
        kind = type(message).__name__
        if kind == "ResultMessage":
            self._reconcile(message)
            return
        if kind != "AssistantMessage":
            return
        for block in getattr(message, "content", None) or ():
            if type(block).__name__ == "ToolUseBlock":
                self._tool_ids.add(block.id)
        usage = _usage_of(getattr(message, "usage", None))
        if usage is None:
            return
        key = getattr(message, "message_id", None) or f"obj-{id(message)}"
        model = getattr(message, "model", None)
        before = self._usage.get(key)
        try:
            if before is None:
                # The first part of a response: a model call, in the step a
                # hook may already have opened for its tool calls.
                self._usage[key] = usage
                self._record(usage, model, same_step=False)
                self._hook_step = False
            else:
                # A later part of the same response: only what it adds.
                delta = {k: max(0, usage[k] - before[k]) for k in _USAGE_FIELDS}
                self._usage[key] = {k: max(usage[k], before[k]) for k in _USAGE_FIELDS}
                if any(delta.values()):
                    self._record(delta, model, same_step=True)
        except Stopped as stop:
            self.stopped = stop
            raise

    def finish(self) -> None:
        """Call after the last message. Raises Stopped if this run was stopped."""
        if self.stopped is not None and self.run.enforce:
            self.stopped.summary = self.run.summary()   # with the result's totals added
            raise self.stopped

    def _reconcile(self, result: Any) -> None:
        """Add what the result's totals hold beyond what the messages showed."""
        totals = _model_usage_of(getattr(result, "model_usage", None))
        if totals is not None:
            # Cumulative over the session: this query is the growth since the last result.
            query = {m: {k: u[k] - self._session_totals.get(m, {}).get(k, 0) for k in _USAGE_FIELDS}
                     for m, u in totals.items()}
            self._session_totals = totals
        else:
            usage = _usage_of(getattr(result, "usage", None))      # per query
            model = next(reversed(self._since_result), None) if self._since_result else None
            query = {model: usage} if usage is not None else {}
        since, self._since_result = self._since_result, {}
        for model, usage in query.items():
            seen = _match(since, model)
            delta = {k: max(0, usage[k] - seen.get(k, 0)) for k in _USAGE_FIELDS}
            if any(delta.values()):
                self._record(delta, model, same_step=True)
        self._since_result = {}   # the next query starts from zero

    def _record(self, usage: dict[str, int], model: str | None, same_step: bool) -> None:
        bucket = self._since_result.setdefault(model, dict.fromkeys(_USAGE_FIELDS, 0))
        for k in _USAGE_FIELDS:
            bucket[k] += usage[k]
        self.run.record_model_call(usage["input_tokens"], usage["output_tokens"], model=model,
                                   cache_read_tokens=usage["cache_read_input_tokens"],
                                   cache_write_tokens=usage["cache_creation_input_tokens"],
                                   same_step=same_step or self._hook_step)

    # -- hooks ------------------------------------------------------------

    async def _pre_tool(self, data: dict[str, Any], tool_use_id: str | None, context: Any) -> HookOutput:
        def check() -> None:
            tid = tool_use_id or data.get("tool_use_id")
            if tid not in self._tool_ids and not self._hook_step:
                # The hook ran before the message that asked for this call
                # was observed: open its step now, so the call is counted in it.
                self.run.before_model_call()
                self._hook_step = True
            self.run.before_tool(data.get("tool_name", "?"), data.get("tool_input"))

        return self._answer("PreToolUse", check)

    async def _post_tool(self, data: dict[str, Any], tool_use_id: str | None, context: Any) -> HookOutput:
        def check() -> None:
            tool, args = data.get("tool_name", "?"), data.get("tool_input")
            response = data.get("tool_response")
            latency = data.get("duration_ms")
            failure = _error_result(response)
            if failure is not None:
                self.run.after_tool(tool, args, error=ToolFailed(failure, status_from_text(failure)),
                                    latency_ms=latency)
            else:
                self.run.after_tool(tool, args, result=response, latency_ms=latency)

        return self._answer("PostToolUse", check)

    async def _tool_failed(self, data: dict[str, Any], tool_use_id: str | None, context: Any) -> HookOutput:
        def check() -> None:
            if data.get("is_interrupt"):
                return  # cancelled by the user, not a failure of the tool
            error = str(data.get("error") or "tool failed")
            self.run.after_tool(data.get("tool_name", "?"), data.get("tool_input"),
                                error=ToolFailed(error, status_from_text(error)),
                                latency_ms=data.get("duration_ms"))

        return self._answer("PostToolUseFailure", check)

    def _answer(self, event: str, check: Callable[[], None]) -> HookOutput:
        try:
            check()
        except Stopped as stop:
            self.stopped = stop
            return _halt(event, stop)
        except Exception as exc:  # a fault here must not break the agent
            warnings.warn(f"agent-stoploss: {event} hook failed, call not checked: {exc!r}",
                          RuntimeWarning, stacklevel=2)
            return {}
        texts = self.run.notice_texts()
        if not texts:
            return {}
        return {"hookSpecificOutput": {"hookEventName": event, "additionalContext": "\n\n".join(texts)}}


async def guarded_query(guard: Any, *, prompt: Any, options: Any = None, agent: str | None = None,
                        run_id: str | None = None, metadata: dict[str, Any] | None = None,
                        query: Callable[..., AsyncIterator[Any]] | None = None) -> AsyncIterator[Any]:
    """``claude_agent_sdk.query`` as one run of ``guard``: yields the same
    messages; in enforce mode raises Stopped once the agent has been halted.
    ``query`` replaces the SDK function, for tests."""
    if query is None:
        from claude_agent_sdk import query

    async with guard.run(agent=agent, run_id=run_id, metadata=metadata) as run:
        sl = StopLossHooks(run)
        stream = query(prompt=prompt, options=sl.options(options))
        try:
            async for message in stream:
                sl.observe(message)
                yield message
        except GeneratorExit:
            # The caller stopped reading: neither a success nor a failure.
            run.end("interrupted")
            raise
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()
        sl.finish()


def _usage_of(usage: Any) -> dict[str, int] | None:
    if not isinstance(usage, dict):
        return None
    out = {k: int(usage.get(k) or 0) for k in _USAGE_FIELDS}
    return out if any(out.values()) else None


_MODEL_USAGE_FIELDS = {"input_tokens": "inputTokens", "output_tokens": "outputTokens",
                       "cache_read_input_tokens": "cacheReadInputTokens",
                       "cache_creation_input_tokens": "cacheCreationInputTokens"}


def _match(recorded: dict[str, dict[str, int]], model: str | None) -> dict[str, int]:
    """What was recorded for ``model``, also when one name is the other plus a
    date or provider suffix (``claude-haiku-5-5`` and ``claude-haiku-5-5-20261001``)."""
    if model in recorded:
        return recorded[model]
    for name, usage in recorded.items():
        if name and model and (name.startswith(model) or model.startswith(name)):
            return usage
    return {}


def _model_usage_of(model_usage: Any) -> dict[str, dict[str, int]] | None:
    if not isinstance(model_usage, dict) or not model_usage:
        return None
    return {model: {k: int((u or {}).get(src) or 0) for k, src in _MODEL_USAGE_FIELDS.items()}
            for model, u in model_usage.items() if isinstance(u, dict)}


def _error_result(response: Any) -> str | None:
    """The error text of a tool result flagged as an error (MCP tools report
    failures as results with is_error / isError), else None."""
    if not isinstance(response, dict) or not (response.get("is_error") or response.get("isError")):
        return None
    content = response.get("content")
    if isinstance(content, list):
        text = " ".join(str(c.get("text", "")) for c in content if isinstance(c, dict)).strip()
    else:
        text = str(content or "").strip()
    return text or "tool returned an error"


def _halt(event: str, stop: Stopped) -> HookOutput:
    reason = f"agent-stoploss stopped this run: {stop.reason}"
    out: HookOutput = {"continue_": False, "stopReason": reason, "systemMessage": reason}
    if event == "PreToolUse":
        out["hookSpecificOutput"] = {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                     "permissionDecisionReason": reason}
    return out
