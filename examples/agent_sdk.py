"""agent-stoploss with the Claude Agent SDK.

    pip install "agent-stoploss[claude-agent-sdk]"
    export ANTHROPIC_API_KEY=...
    python examples/agent_sdk.py [shadow|enforce]

The agent is asked for a traffic report from an analytics tool whose API key
has expired: every call answers 401. Without a guard an agent like this tends
to retry until max_turns. In enforce mode agent-stoploss ends the run on the
first 401; in shadow mode the run goes on and the journal records where it
would have stopped (see it with: agent-stoploss replay ./stoploss_logs).
"""

from __future__ import annotations

import asyncio
import os
import sys

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, create_sdk_mcp_server, tool

from agent_stoploss import StopLoss, Stopped
from agent_stoploss.claude_agent import guarded_query


@tool("report", "Monthly traffic for a domain from the analytics service.", {"domain": str, "period": str})
async def report(args):
    # The service is up, the key is not: an error retrying cannot fix.
    return {"content": [{"type": "text", "text": "HTTP 401 Unauthorized: invalid or expired API key"}],
            "is_error": True}


async def main(mode: str) -> None:
    guard = StopLoss({"mode": mode, "log": {"dir": "./stoploss_logs"}})
    options = ClaudeAgentOptions(
        model=os.environ.get("STOPLOSS_EXAMPLE_MODEL", "claude-opus-5-5"),
        mcp_servers={"analytics": create_sdk_mcp_server("analytics", tools=[report])},
        tools=[],                                     # no built-in tools: only the analytics one
        allowed_tools=["mcp__analytics__report"],
        max_turns=20,                                 # the SDK's own ceiling, kept as a backstop
        setting_sources=[],                           # keep your own Claude Code settings out of the demo
    )
    prompt = "Get last month's traffic for example.com from the analytics tool and summarise it."
    try:
        async for message in guarded_query(guard, prompt=prompt, options=options, agent="traffic-report"):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print("agent:", block.text)
            elif isinstance(message, ResultMessage):
                print(f"finished: {message.num_turns} turns, ${message.total_cost_usd or 0:.4f}")
    except Stopped as stop:
        print(f"stopped by agent-stoploss: {stop.reason} [{stop.mechanism}]")
        print("run summary:", stop.summary)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "enforce"))
