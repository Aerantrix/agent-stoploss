"""Configuration: what agent-stoploss watches and when it intervenes.

A config is a plain mapping, usually read from stoploss.yaml. Every key is
optional. Unknown keys are an error rather than a silent no-op, because a
misspelt threshold that quietly falls back to the default is exactly the kind
of mistake nobody notices until the bill arrives.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

MODES = ("shadow", "enforce")

DEFAULTS: dict[str, Any] = {
    "mode": "shadow",
    "run_budget": {
        # Off by default: a fixed ceiling cuts honest long tasks short.
        # The replay report suggests a value from your own runs.
        "max_tokens": None,
        "max_cost_usd": None,
        "warn_at": 0.8,
    },
    "loop_detector": {
        "enabled": True,
        "threshold": 3,
        "result_volatile_fields": ["timestamp", "request_id", "trace_id"],
    },
    "circuit_breaker": {
        "enabled": True,
        "window": 5,
        "max_failures": 4,
        "cooldown_seconds": 60,
    },
    "error_classifier": {
        "enabled": True,
        "terminal_statuses": [400, 401, 403, 404, 405, 410, 422],
        "transient_statuses": [408, 425, 429, 500, 502, 503, 504],
        # Exception class names that are terminal whatever their status.
        "terminal_exceptions": [],
    },
    "tools": {},
    "notice": "agent-stoploss: {reason}.",
    "log": {"dir": "./stoploss_logs", "raw_text": False},
    "prices": {},
    "strict_pricing": False,
    # The spend by the end of this step is treated as what any run of the
    # task costs; the replay report calls everything above it excess.
    "minimum_steps": 3,
}

_TOOL_KEYS = {
    "loop_detector": {"enabled", "threshold"},
    "circuit_breaker": {"enabled", "window", "max_failures"},
    "error_classifier": {"enabled"},
}


class ConfigError(ValueError):
    """The config has one or more problems. ``problems`` lists all of them."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("invalid agent-stoploss config:\n  " + "\n  ".join(problems))


@dataclass(frozen=True)
class ToolRules:
    """Per-tool overrides, already merged with the global settings."""

    loop_enabled: bool
    loop_threshold: int
    breaker_enabled: bool
    breaker_window: int
    breaker_max_failures: int
    classifier_enabled: bool


@dataclass
class Config:
    mode: str = "shadow"
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    budget_warn_at: float = 0.8
    loop_enabled: bool = True
    loop_threshold: int = 3
    result_volatile_fields: tuple[str, ...] = ("timestamp", "request_id", "trace_id")
    breaker_enabled: bool = True
    breaker_window: int = 5
    breaker_max_failures: int = 4
    breaker_cooldown_seconds: float = 60.0
    classifier_enabled: bool = True
    terminal_statuses: frozenset[int] = frozenset({400, 401, 403, 404, 405, 410, 422})
    transient_statuses: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})
    terminal_exceptions: frozenset[str] = frozenset()
    tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    notice: str = "agent-stoploss: {reason}."
    log_dir: str | None = "./stoploss_logs"
    raw_text: bool = False
    prices: dict[str, dict[str, float]] = field(default_factory=dict)
    strict_pricing: bool = False
    minimum_steps: int = 3
    source: dict[str, Any] = field(default_factory=dict, repr=False)

    # -- construction -----------------------------------------------------

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None = None) -> "Config":
        merged, problems = _merge(DEFAULTS, dict(data or {}))
        if problems:
            raise ConfigError(problems)
        problems = _validate(merged)
        if problems:
            raise ConfigError(problems)
        b, ld, cb, ec, log = (merged[k] for k in
                              ("run_budget", "loop_detector", "circuit_breaker",
                               "error_classifier", "log"))
        return cls(
            mode=merged["mode"],
            max_tokens=b["max_tokens"],
            max_cost_usd=b["max_cost_usd"],
            budget_warn_at=float(b["warn_at"]),
            loop_enabled=bool(ld["enabled"]),
            loop_threshold=int(ld["threshold"]),
            result_volatile_fields=tuple(ld["result_volatile_fields"]),
            breaker_enabled=bool(cb["enabled"]),
            breaker_window=int(cb["window"]),
            breaker_max_failures=int(cb["max_failures"]),
            breaker_cooldown_seconds=float(cb["cooldown_seconds"]),
            classifier_enabled=bool(ec["enabled"]),
            terminal_statuses=frozenset(int(s) for s in ec["terminal_statuses"]),
            transient_statuses=frozenset(int(s) for s in ec["transient_statuses"]),
            terminal_exceptions=frozenset(ec["terminal_exceptions"]),
            tools={name: _tool_entry(rules) for name, rules in merged["tools"].items()},
            notice=merged["notice"],
            log_dir=log["dir"],
            raw_text=bool(log["raw_text"]),
            prices=merged["prices"],
            strict_pricing=bool(merged["strict_pricing"]),
            minimum_steps=int(merged["minimum_steps"]),
            source=merged,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            data = json.loads(text)
        else:
            import yaml  # PyYAML; a declared dependency

            data = yaml.safe_load(text)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigError([f"{path}: the top level must be a mapping"])
        return cls.from_dict(data)

    # -- derived ----------------------------------------------------------

    def replace(self, **changes: Any) -> "Config":
        """A copy with some top-level settings changed, e.g. mode="enforce"."""
        new = copy.copy(self)
        for key, value in changes.items():
            if not hasattr(new, key):
                raise AttributeError(key)
            setattr(new, key, value)
        return new

    @property
    def hash(self) -> str:
        """Short, stable id of the settings that affect decisions."""
        relevant = {k: v for k, v in self.__dict__.items()
                    if k not in ("source", "log_dir", "raw_text", "mode")}
        blob = json.dumps(relevant, sort_keys=True, default=sorted)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def rules_for(self, tool: str) -> ToolRules:
        t = self.tools.get(tool, {})
        ld, cb, ec = t.get("loop_detector", {}), t.get("circuit_breaker", {}), t.get("error_classifier", {})
        return ToolRules(
            loop_enabled=self.loop_enabled and ld.get("enabled", True),
            loop_threshold=int(ld.get("threshold", self.loop_threshold)),
            breaker_enabled=self.breaker_enabled and cb.get("enabled", True),
            breaker_window=int(cb.get("window", self.breaker_window)),
            breaker_max_failures=int(cb.get("max_failures", self.breaker_max_failures)),
            classifier_enabled=self.classifier_enabled and ec.get("enabled", True),
        )


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _merge(defaults: dict, data: dict, prefix: str = "") -> tuple[dict, list[str]]:
    """Overlay data on defaults, reporting keys the defaults do not know.

    ``tools`` and ``prices`` are open mappings and are taken as given.
    A section set to false (``loop_detector: off`` in YAML) disables it.
    """
    out = copy.deepcopy(defaults)
    problems = []
    for key, value in data.items():
        where = f"{prefix}{key}"
        if key not in defaults:
            problems.append(f"{where}: unknown setting")
            continue
        default = defaults[key]
        if key in ("tools", "prices"):
            if not isinstance(value, dict):
                problems.append(f"{where}: must be a mapping")
            else:
                out[key] = copy.deepcopy(value)
        elif isinstance(default, dict) and "enabled" in default and isinstance(value, bool):
            out[key]["enabled"] = value
        elif isinstance(default, dict):
            if not isinstance(value, dict):
                problems.append(f"{where}: must be a mapping")
                continue
            sub, sub_problems = _merge(default, value, prefix=f"{where}.")
            out[key] = sub
            problems += sub_problems
        else:
            out[key] = value
    return out, problems


def _tool_entry(rules: Any) -> dict[str, Any]:
    out = {}
    for section, value in (rules or {}).items():
        out[section] = {"enabled": value} if isinstance(value, bool) else dict(value)
    return out


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _validate(c: dict) -> list[str]:
    p = []
    if c["mode"] not in MODES:
        p.append(f"mode: must be one of {', '.join(MODES)}, got {c['mode']!r}")

    b = c["run_budget"]
    if b["max_tokens"] is not None and not (_is_int(b["max_tokens"]) and b["max_tokens"] > 0):
        p.append("run_budget.max_tokens: must be a positive integer or null")
    if b["max_cost_usd"] is not None and not (_is_num(b["max_cost_usd"]) and b["max_cost_usd"] > 0):
        p.append("run_budget.max_cost_usd: must be a positive number or null")
    if not (_is_num(b["warn_at"]) and 0 < b["warn_at"] <= 1):
        p.append("run_budget.warn_at: must be a fraction in (0, 1]")

    ld = c["loop_detector"]
    if not (_is_int(ld["threshold"]) and ld["threshold"] >= 2):
        p.append("loop_detector.threshold: must be an integer of at least 2")
    if not (isinstance(ld["result_volatile_fields"], list)
            and all(isinstance(f, str) for f in ld["result_volatile_fields"])):
        p.append("loop_detector.result_volatile_fields: must be a list of field names")

    cb = c["circuit_breaker"]
    if not (_is_int(cb["window"]) and cb["window"] >= 1):
        p.append("circuit_breaker.window: must be a positive integer")
    if not (_is_int(cb["max_failures"]) and cb["max_failures"] >= 1):
        p.append("circuit_breaker.max_failures: must be a positive integer")
    elif _is_int(cb["window"]) and cb["max_failures"] > cb["window"]:
        p.append("circuit_breaker.max_failures: cannot exceed window, or it never trips")
    if not (_is_num(cb["cooldown_seconds"]) and cb["cooldown_seconds"] >= 0):
        p.append("circuit_breaker.cooldown_seconds: must be a number of seconds")

    ec = c["error_classifier"]
    for key in ("terminal_statuses", "transient_statuses"):
        v = ec[key]
        if not (isinstance(v, list) and all(_is_int(s) and 100 <= s <= 599 for s in v)):
            p.append(f"error_classifier.{key}: must be a list of HTTP status codes")
    if isinstance(ec["terminal_statuses"], list) and isinstance(ec["transient_statuses"], list):
        both = set(ec["terminal_statuses"]) & set(ec["transient_statuses"])
        if both:
            p.append(f"error_classifier: statuses listed as both terminal and transient: {sorted(both)}")
    if not (isinstance(ec["terminal_exceptions"], list)
            and all(isinstance(s, str) for s in ec["terminal_exceptions"])):
        p.append("error_classifier.terminal_exceptions: must be a list of exception class names")

    for name, rules in c["tools"].items():
        if rules is None:
            continue
        if not isinstance(rules, dict):
            p.append(f"tools.{name}: must be a mapping")
            continue
        for section, value in rules.items():
            if section not in _TOOL_KEYS:
                p.append(f"tools.{name}.{section}: unknown setting "
                         f"(expected one of {', '.join(sorted(_TOOL_KEYS))})")
            elif isinstance(value, bool):
                continue
            elif not isinstance(value, dict):
                p.append(f"tools.{name}.{section}: must be off, on or a mapping")
            else:
                for k, v in value.items():
                    if k not in _TOOL_KEYS[section]:
                        p.append(f"tools.{name}.{section}.{k}: unknown setting")
                    elif k == "enabled" and not isinstance(v, bool):
                        p.append(f"tools.{name}.{section}.enabled: must be true or false")
                    elif k != "enabled" and not (_is_int(v) and v >= 1):
                        p.append(f"tools.{name}.{section}.{k}: must be a positive integer")

    if not isinstance(c["notice"], str) or "{reason}" not in c["notice"]:
        p.append("notice: must be text containing {reason}")
    else:
        try:
            c["notice"].format(reason="x")
        except (KeyError, IndexError, ValueError):
            p.append("notice: the only placeholder allowed is {reason}")

    log = c["log"]
    if log["dir"] is not None and not isinstance(log["dir"], str):
        p.append("log.dir: must be a path or null")
    if not isinstance(log["raw_text"], bool):
        p.append("log.raw_text: must be true or false")

    for model, price in c["prices"].items():
        if not isinstance(price, dict) or not {"input", "output"} <= set(price):
            p.append(f"prices.{model}: needs input and output, in USD per million tokens")
            continue
        for k, v in price.items():
            if k not in ("input", "output", "cache_read", "cache_write"):
                p.append(f"prices.{model}.{k}: unknown price "
                         "(expected input, output, cache_read, cache_write)")
            elif not (_is_num(v) and v >= 0):
                p.append(f"prices.{model}.{k}: must be a non-negative number")

    if not isinstance(c["strict_pricing"], bool):
        p.append("strict_pricing: must be true or false")
    if not (_is_int(c["minimum_steps"]) and c["minimum_steps"] >= 0):
        p.append("minimum_steps: must be a non-negative integer")
    return p
