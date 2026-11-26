"""agent-stoploss in your own agent loop, with the Anthropic SDK.

    pip install agent-stoploss anthropic
    export ANTHROPIC_API_KEY=...
    python examples/anthropic_loop.py [shadow|enforce]

Three lines of agent-stoploss: wrap the client (tokens and cost are counted
by themselves), decorate the tool (calls, results and errors are checked),
and run the loop inside guard.run(). The analytics service in this example
is down; the circuit breaker and the loop detector end the run before the
agent burns through its turns.
"""

from __future__ import annotations

import os
import sys

import anthropic

from agent_stoploss import StopLoss, Stopped

MODEL = os.environ.get("STOPLOSS_EXAMPLE_MODEL", "claude-opus-5-5")


class ServiceDown(Exception):
    pass


def make_agent(mode: str):
    guard = StopLoss({"mode": mode, "log": {"dir": "./stoploss_logs"}})
    client = guard.wrap(anthropic.Anthropic())

    @guard.tool
    def analytics(domain: str, period: str = "30d") -> dict:
        raise ServiceDown("connection refused by analytics.internal:443")

    tools = [{
        "name": "analytics",
        "description": "Monthly traffic for a domain from the analytics service.",
        "input_schema": {"type": "object",
                         "properties": {"domain": {"type": "string"}, "period": {"type": "string"}},
                         "required": ["domain"]},
    }]
    return guard, client, analytics, tools


def main(mode: str) -> None:
    guard, client, analytics, tools = make_agent(mode)
    messages = [{"role": "user", "content": "Get last month's traffic for example.com and summarise it."}]

    with guard.run(agent="traffic-report") as run:
        try:
            for _ in range(20):                       # your own ceiling stays; agent-stoploss works inside it
                response = client.messages.create(model=MODEL, max_tokens=16000, tools=tools, messages=messages)
                if response.stop_reason == "end_turn":
                    break
                messages.append({"role": "assistant", "content": response.content})
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    try:
                        content, is_error = str(analytics(**block.input)), False
                    except ServiceDown as exc:
                        content, is_error = f"Tool error: {exc}", True
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": content, "is_error": is_error})
                # Warnings from agent-stoploss (enforce mode only) go to the
                # model with the tool results, one step before a stop.
                results += [{"type": "text", "text": t} for t in run.notice_texts()]
                messages.append({"role": "user", "content": results})
            text = next((b.text for b in response.content if b.type == "text"), "")
            print("agent:", text)
            run.outcome("failed")                     # the report was never produced
        except Stopped as stop:
            run.outcome("failed")
            print(f"stopped by agent-stoploss: {stop.reason} [{stop.mechanism}]")
        print(f"{run.step} steps, {run.tokens:,} tokens, ${run.cost_usd:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "enforce")
