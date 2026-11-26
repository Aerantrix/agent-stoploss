# Changelog

## 0.1.0 — unreleased

First release.

- `StopLoss` with shadow and enforce modes; one guard serves any number of
  concurrent runs (threads or asyncio tasks).
- Mechanisms: loop detector on call and result fingerprints, error classifier
  (terminal and transient HTTP statuses, exception names), circuit breaker per
  tool with a sliding window, cooldown and probe, run budget in tokens or USD
  (off by default), strict pricing.
- Notices to the agent one step before a loop or failing-tool stop; `Stopped`
  with the reason and a run summary; `on_stop` and `on_verdict` callbacks.
- Integrations: `@guard.tool` for sync and async tools; `guard.wrap()` for the
  Anthropic and OpenAI clients, sync and async; the Claude Agent SDK through
  `agent_stoploss.claude_agent` (`guarded_query`, `StopLossHooks`), checked
  live against claude-agent-sdk 0.2.189.
- JSONL journal, schema 1, fingerprints and shapes instead of raw text unless
  `log.raw_text` is set.
- `agent-stoploss replay`: what a config would have stopped, saved and broken
  on recorded runs, with a suggested run budget.
- `agent-stoploss dashboard`: a local page to tune thresholds against recorded
  runs and export `stoploss.yaml`.
- `agent-stoploss check-config`.
- Built-in prices for the Claude models as of 2026-10-06.
