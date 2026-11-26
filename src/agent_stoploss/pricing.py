"""Token prices and the usage objects of the model SDKs.

Prices are USD per million tokens. The built-in table covers Anthropic's
first-party API and is dated; prices change, so anything in ``prices:`` in the
config wins over it. A model that is in neither is counted at $0 with a
warning, or stops the run when ``strict_pricing`` is on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

PRICES_AS_OF = "2026-10-06"

# Input and output are the published rates. Cache prices follow the standard
# rule (read 0.1x input, 5-minute write 1.25x input) except where a model's
# published cache-read price differs (Fable 5.1, Opus 5.5, Sonnet 5.5, Sonnet 5).
BUILTIN_PRICES: dict[str, dict[str, float]] = {
    "claude-fable-5-1": {"input": 10.00, "output": 50.00, "cache_read": 0.25, "cache_write": 12.50},
    "claude-fable-5": {"input": 10.00, "output": 50.00, "cache_read": 1.00, "cache_write": 12.50},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_read": 0.20, "cache_write": 5.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00, "cache_read": 0.50, "cache_write": 6.25},
    "claude-opus-4-8": {"input": 5.00, "output": 25.00, "cache_read": 0.50, "cache_write": 6.25},
    "claude-opus-4-7": {"input": 5.00, "output": 25.00, "cache_read": 0.50, "cache_write": 6.25},
    "claude-opus-4-6": {"input": 5.00, "output": 25.00, "cache_read": 0.50, "cache_write": 6.25},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20, "cache_write": 2.50},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00, "cache_read": 0.20, "cache_write": 2.50},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cache_read": 0.30, "cache_write": 3.75},
    # Haiku 5.5 is priced higher above 100K input tokens ($0.50 / $2.50);
    # the table uses the base rate.
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00, "cache_read": 0.10, "cache_write": 1.25},
}


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_read_tokens + self.cache_write_tokens


def _get(obj: Any, name: str) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name)
    return getattr(obj, name, None)


def _int(v: Any) -> int:
    return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def parse_usage(usage: Any) -> Usage:
    """Read a usage object from the Anthropic or OpenAI SDK, or a plain dict.

    Anthropic: input_tokens excludes cached tokens, which come separately.
    OpenAI chat: prompt_tokens includes cached tokens
    (prompt_tokens_details.cached_tokens). OpenAI responses: input_tokens
    includes input_tokens_details.cached_tokens.
    """
    if usage is None:
        return Usage()
    if _get(usage, "prompt_tokens") is not None:  # OpenAI chat completions
        cached = _int(_get(_get(usage, "prompt_tokens_details") or {}, "cached_tokens"))
        return Usage(_int(_get(usage, "prompt_tokens")) - cached,
                     _int(_get(usage, "completion_tokens")), cached, 0)
    details = _get(usage, "input_tokens_details")
    if details is not None:  # OpenAI responses
        cached = _int(_get(details, "cached_tokens"))
        return Usage(_int(_get(usage, "input_tokens")) - cached,
                     _int(_get(usage, "output_tokens")), cached, 0)
    return Usage(  # Anthropic
        _int(_get(usage, "input_tokens")),
        _int(_get(usage, "output_tokens")),
        _int(_get(usage, "cache_read_input_tokens")),
        _int(_get(usage, "cache_creation_input_tokens")),
    )


def _base_id(model: str) -> str:
    m = model.strip().lower()
    # Bedrock and cross-region ids: "anthropic.claude-...", "us.anthropic.claude-..."
    if "anthropic." in m:
        m = m.split("anthropic.", 1)[1]
    return m


def find_price(model: str | None, overrides: Mapping[str, Mapping[str, float]] | None = None):
    """The price entry for a model id, or None.

    Exact match first, then the longest known id the model starts with, so
    dated or versioned ids ("...-20261001", "...@20261001", "...-v1:0") match.
    """
    if not model:
        return None
    for table in (overrides or {}, BUILTIN_PRICES):
        if model in table:
            return table[model]
    m = _base_id(model)
    for table in (overrides or {}, BUILTIN_PRICES):
        keys = sorted((k for k in table if m == k.lower() or m.startswith(k.lower() + "-")
                       or m.startswith(k.lower() + "@")), key=len, reverse=True)
        if keys:
            return table[keys[0]]
    return None


def cost_of(usage: Usage, price: Mapping[str, float]) -> float:
    inp = price["input"]
    return (usage.input_tokens * inp
            + usage.output_tokens * price["output"]
            + usage.cache_read_tokens * price.get("cache_read", inp * 0.1)
            + usage.cache_write_tokens * price.get("cache_write", inp * 1.25)) / 1_000_000
