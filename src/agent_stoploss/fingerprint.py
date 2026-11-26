"""Fingerprints of tool calls and tool results.

Two prints per call:

    call print    tool name + normalised arguments
    result print  tool name + normalised result

The result print is what catches an agent that keeps rewording its query and
keeps getting the same empty answer: the call print differs every time, the
result print does not. Normalisation is deliberately conservative. Forty
calls with genuinely different arguments must stay forty different prints.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from typing import Any, Iterable

_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
# A date with a time of day. A bare date is left alone: a report for
# 2026-11-01 and one for 2026-11-02 are different answers.
_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def to_plain(value: Any) -> Any:
    """Best-effort conversion of SDK objects to JSON-able data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [to_plain(v) for v in value]
        return sorted(items, key=repr) if isinstance(value, (set, frozenset)) else items
    if isinstance(value, bytes):
        return {"<bytes>": digest(value.hex())}
    dump = getattr(value, "model_dump", None)  # pydantic v2
    if callable(dump):
        try:
            return to_plain(dump())
        except Exception:
            pass
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return to_plain(dataclasses.asdict(value))
    return str(value)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def normalise_args(args: Any) -> Any:
    """Sorted keys, strings stripped and lowercased."""
    def norm(v: Any) -> Any:
        if isinstance(v, str):
            return v.strip().lower()
        if isinstance(v, dict):
            return {k: norm(x) for k, x in v.items()}
        if isinstance(v, list):
            return [norm(x) for x in v]
        return v

    return norm(to_plain(args if args is not None else {}))


def call_print(tool: str, args: Any) -> str:
    return digest(f"{tool}|{canonical(normalise_args(args))}")


def normalise_result(result: Any, args: Any, volatile: Iterable[str]) -> Any:
    """Drop what changes between identical answers.

    - top-level fields that echo an argument back: the request is not part of
      the answer, or a changed query would always look like a new result;
    - fields named in ``volatile`` at any depth (timestamp, request_id, ...);
    - string values that are a UUID or a date with a time of day.
    """
    volatile = set(volatile)
    value = to_plain(result)
    if isinstance(value, str) and value[:1] in "{[":
        try:
            value = json.loads(value)
        except ValueError:
            pass
    arg_keys = set(to_plain(args).keys()) if isinstance(to_plain(args), dict) else set()
    if isinstance(value, dict):
        value = {k: v for k, v in value.items() if k not in arg_keys}

    def scrub(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: scrub(x) for k, x in v.items() if k not in volatile}
        if isinstance(v, list):
            return [scrub(x) for x in v]
        if isinstance(v, str):
            if _UUID.match(v):
                return "<uuid>"
            if _DATETIME.match(v):
                return "<datetime>"
        return v

    return scrub(value)


def result_print(tool: str, args: Any, result: Any, volatile: Iterable[str] = ()) -> str:
    return digest(f"{tool}|{canonical(normalise_result(result, args, volatile))}")


def shape(value: Any) -> dict[str, Any]:
    """What the journal keeps instead of raw text: type, keys and size."""
    plain = to_plain(value)
    out: dict[str, Any] = {"type": type(plain).__name__, "size": len(canonical(plain))}
    if isinstance(plain, dict):
        out["keys"] = sorted(plain)[:50]
    elif isinstance(plain, list):
        out["len"] = len(plain)
    return out
