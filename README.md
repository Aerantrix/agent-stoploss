# agent-stoploss

**Stop AI agents from burning tokens on broken tools.**

Based on the [Agent Fuel study](https://github.com/Aerantrix/Agent-Fuel): 660
recorded agent runs under injected tool failures, with the controller, the run
records and the analysis that this library's defaults come from.

An agent calls an API, the key has expired, and the agent tries again. And
again, sixty times, each attempt resending the whole history of failures. A
retry loop like this turns a task worth a few hundred tokens into a runaway
bill, and a step limit or a monthly budget only notices once the money is gone.

agent-stoploss watches each agent run and ends it when the agent is spending
without making progress:

- the same tool call again and again;
- the same answer from a tool again and again;
- an error that retrying cannot fix (401, 403, 404, ...);
- a tool that keeps failing.

It starts in **shadow mode**: it records what it would have done and changes
nothing. `agent-stoploss replay` then shows, on your own recorded runs, what a
configuration would have saved and which good runs it would have broken. Switch
to **enforce** when the numbers look right.

```
pip install agent-stoploss
```

Python 3.10+. One dependency (PyYAML). Apache-2.0.

## Try it

The demo needs no API key and no network. A scripted, stubborn agent makes
four runs, one healthy and three stuck on a broken tool: first in shadow mode,
then the replay report on what was recorded, then in enforce mode.

```
git clone https://github.com/Aerantrix/agent-stoploss
cd agent-stoploss
python -m venv .venv && .venv/bin/pip install -e .
.venv/bin/python examples/offline_demo.py ./stoploss_logs
.venv/bin/agent-stoploss dashboard ./stoploss_logs
```

With a real model, a few cents on Claude Haiku 5.5. The examples give the agent
a tool that always fails; run them in `shadow` or `enforce` and compare:

```
.venv/bin/pip install -e ".[claude-agent-sdk]" anthropic
export ANTHROPIC_API_KEY=...
export STOPLOSS_EXAMPLE_MODEL=claude-haiku-5-5      # the default is claude-opus-5-5
.venv/bin/python examples/anthropic_loop.py enforce
.venv/bin/python examples/agent_sdk.py enforce
.venv/bin/agent-stoploss replay ./stoploss_logs
```

On Windows the commands are under `.venv\Scripts\` instead of `.venv/bin/`.

## Why another spend guard?

Most guards count: steps, tokens, dollars. agent-stoploss also reads what the
tools answer, so it can tell an agent that is stuck from one that is simply
working through a long task.

| It stops | It leaves alone |
|---|---|
| a call answered with 401: retrying cannot help, so the run ends at the first one | forty different domains looked up one by one |
| the same call made three times, even with case or spacing changed | paging through results, `page=1..N` |
| the same result returned three times, timestamps and request ids ignored | a tool that fails now and then, fewer than 4 of its last 5 calls |
| a tool that failed in 4 of its last 5 calls | polling a job you marked as `loop_detector: off` |

A run budget is there too, off by default: a fixed ceiling also cuts honest long
tasks short, so replay suggests a value from your own runs instead.

## Quick start

### Claude Agent SDK

```
pip install "agent-stoploss[claude-agent-sdk]"
```

```python
from claude_agent_sdk import ClaudeAgentOptions
from agent_stoploss import StopLoss, Stopped
from agent_stoploss.claude_agent import guarded_query

guard = StopLoss.from_file("stoploss.yaml")      # or StopLoss(mode="shadow")

try:
    async for message in guarded_query(guard, prompt="...", options=ClaudeAgentOptions(...),
                                       agent="weekly-report"):
        ...                                       # the same messages query() yields
except Stopped as stop:
    print(stop.reason, stop.summary)
```

`guarded_query` adds hooks before and after every tool call and reads token
usage from the messages. In enforce mode a stop denies the pending tool call
and ends the agent's turn, then `Stopped` is raised. A stop decided on a tool's
answer, such as a 401, takes effect at the agent's next tool call: Claude Code
lets the model reply once more first. Token counts are corrected from the
SDK's final totals, so the journal matches what the SDK reports. With `ClaudeSDKClient`,
use `StopLossHooks` directly; see [`agent_stoploss/claude_agent.py`](src/agent_stoploss/claude_agent.py).

### Your own agent loop

```python
import anthropic
from agent_stoploss import StopLoss, Stopped

guard = StopLoss.from_file("stoploss.yaml")
client = guard.wrap(anthropic.Anthropic())        # tokens and cost recorded for you

@guard.tool                                        # calls, results and errors checked
def analytics(domain: str, period: str = "30d") -> dict: ...

with guard.run(agent="weekly-report") as run:
    try:
        while True:
            response = client.messages.create(model=..., max_tokens=..., tools=tools, messages=messages)
            if response.stop_reason == "end_turn":
                break
            ...                                    # run the tools, collect tool_result blocks
            results += [{"type": "text", "text": t} for t in run.notice_texts()]
            messages.append({"role": "user", "content": results})
        run.outcome("ok")
    except Stopped as stop:
        run.outcome("failed")
        report_partial_results(stop.reason)
```

`wrap()` works with the Anthropic and OpenAI clients, sync and async. Without
them, record each model call yourself with `run.record_usage(response.usage, model=...)`
and each tool call with `run.before_tool(...)` / `run.after_tool(...)`.

Full examples: [`examples/agent_sdk.py`](examples/agent_sdk.py),
[`examples/anthropic_loop.py`](examples/anthropic_loop.py), and
[`examples/offline_demo.py`](examples/offline_demo.py), which runs the whole
workflow on a scripted agent with no API key.

## Shadow, replay, enforce

1. **Shadow.** Deploy with `mode: shadow`. Every run is recorded to
   `./stoploss_logs/stoploss-YYYY-MM-DD.jsonl`; the agent never notices.
2. **Replay.** Point the replay at the logs:

   ```
   $ agent-stoploss replay ./stoploss_logs --config stoploss.yaml
   Runs                           660   (660 labelled by the developer, the rest by heuristic)
   Recorded spend            12,888,348 tokens
   Would have stopped             251
     same call repeated         129
     error retrying cannot fix    73
     same result repeated        46
     tool keeps failing           3
   Would have saved          2,312,156 tokens
     of the excess in runs not ok: 99%

   Runs labelled ok that would have been stopped: 6
   ...
   ```

   The two lists that matter are the good runs it would have stopped and the
   failed runs it would have missed. Label your runs with `run.outcome("ok")`
   or `run.outcome("failed")`; without a label the replay guesses and says so.
3. **Tune.** Exempt a tool, raise a threshold, replay again. Or use the dashboard.
4. **Enforce.** Set `mode: enforce`. Before a loop or a failing tool ends the
   run, the agent gets a short notice it can act on; an error that retrying
   cannot fix ends the run at once. On a stop your code gets `Stopped`.

## Dashboard

```
agent-stoploss dashboard ./stoploss_logs --config stoploss.yaml
```

A local page for tuning thresholds against your recorded runs: what would be
stopped and why, what it would save, which good runs it would break, and each
run step by step with the point where it would have ended. Every change is
replayed at once. It exports `stoploss.yaml` with only what differs from the
defaults.

It listens on 127.0.0.1 only, loads nothing from the internet, writes nothing
to disk and never controls a running agent.

## Configuration

Every key is optional. [`stoploss.example.yaml`](stoploss.example.yaml) lists
them all with their defaults:

```yaml
mode: shadow
loop_detector:   {threshold: 3}
circuit_breaker: {window: 5, max_failures: 4, cooldown_seconds: 60}
error_classifier:
  terminal_statuses:  [400, 401, 403, 404, 405, 410, 422]
  transient_statuses: [408, 425, 429, 500, 502, 503, 504]
run_budget:      {max_tokens: null, max_cost_usd: null}
tools:
  job_status: {loop_detector: off}        # polls the same status on purpose
```

`agent-stoploss check-config stoploss.yaml` rejects unknown keys and bad values
instead of quietly falling back to a default. Why the defaults are what they
are: [`docs/design-notes.md`](docs/design-notes.md).

## What it records

One JSON line per event: run start and end, each step, model call (tokens,
cost), tool call and result, verdict, notice. Arguments and results are kept as
fingerprints and shapes, not text, unless you set `log.raw_text: true`. The
journal stays on your disk; agent-stoploss sends nothing anywhere.

Built-in prices cover the Claude models (as of 2026-10-06). Other models need
a `prices:` entry, or their cost is counted as $0 with a warning;
`strict_pricing: true` stops such runs instead.

## What it does not catch

- **An agent that changes its request every time** and gets a different error
  each time. Each call looks new; only the circuit breaker and the budget see it.
- **Slow burns that make progress**: a long task that is merely expensive is a
  budget question, not a loop.
- **Streaming responses** inside a run: their tokens cannot be counted yet, so
  agent-stoploss raises `StreamingNotSupported` rather than count them as zero.
- **Behaviour after a warning** is not in the replay. It assumes the run went on
  as recorded; in enforce mode an agent that heeds the notice may stop earlier.
- **Large models and real APIs** have not been measured yet. The numbers below
  come from small models and a stub service.

## Based on the Agent Fuel study

The mechanisms and defaults come from the
[Agent Fuel study](https://github.com/Aerantrix/Agent-Fuel) (CSIT305, University
of Wollongong in Dubai, 2026): 660 runs of two small models, Qwen 2.5 7B and
Claude Haiku 5.5, against a stub analytics service in nine scenarios (six
failures of the service, three healthy tasks), under thirteen system prompts,
nine of them the default prompts of agent frameworks. Replayed with the default configuration, agent-stoploss stops 245
of the 444 failing runs and saves 99% of the tokens they spent beyond the
minimum needed to see the failure. It would have stopped 6 of the 216 healthy
runs, all with the local 7B model; 8 failing runs that spent above that minimum
got through. The test suite reproduces these numbers from the study's records.

## Development

```
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/ruff check src tests examples
```

Two tests are skipped without the Agent Fuel study: they replay its 660 recorded
runs and check that the library stops each one where the study's own replay
did. With a checkout of the study they run too, and the same checkout gives
the dashboard the study's runs to explore:

```
git clone https://github.com/Aerantrix/Agent-Fuel ../agent-fuel
AGENT_FUEL_REPO=../agent-fuel .venv/bin/python -m pytest
AGENT_FUEL_REPO=../agent-fuel .venv/bin/python tests/ui/make_research_logs.py research_logs
.venv/bin/agent-stoploss dashboard research_logs
```

## License

Apache-2.0. See [LICENSE](LICENSE).
