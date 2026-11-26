"""Reading tool errors: the HTTP status, if any, and whether a retry can help.

This is the one place agent-stoploss makes a judgement rather than counting a
fact, so the rule is small and explicit: a status in the terminal list (401,
404, ...) means retrying the same call cannot succeed. Everything else is left
to the circuit breaker and the loop detector.
"""

from __future__ import annotations

import re
from typing import Any


def _as_status(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and 100 <= value <= 599:
        return value
    if isinstance(value, str) and value.isdigit() and 100 <= int(value) <= 599:
        return int(value)
    return None


def _attr(obj: Any, name: str) -> Any:
    # Some clients raise from a property when the field was never set
    # (httpx's RequestError.request, for one).
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def status_of(error: BaseException) -> int | None:
    """The HTTP status carried by an exception from a common client or SDK.

    Covers requests, httpx, aiohttp, urllib, the Anthropic and OpenAI SDKs and
    any exception with a ``status_code`` or ``status`` attribute.
    """
    for name in ("status_code", "status", "http_status"):
        status = _as_status(_attr(error, name))
        if status is not None:
            return status
    response = _attr(error, "response")
    if response is not None:
        for name in ("status_code", "status"):
            status = _as_status(_attr(response, name))
            if status is not None:
                return status
    return _as_status(_attr(error, "code"))  # urllib.error.HTTPError


# Only explicit forms: "HTTP 401", "status: 503", "status_code=429",
# "Error code: 404", "401 Unauthorized". A bare number in a message is too
# often a count, a port or an id to be read as a status.
_STATUS_LABELLED = re.compile(
    r"\b(?:HTTP(?:/\d(?:\.\d)?)?|status(?:[ _]?code)?|error[ _]code|response[ _]code)\s*[:=#]?\s*([1-5]\d\d)\b",
    re.IGNORECASE)
_STATUS_REASON = re.compile(
    r"\b([45]\d\d)\s+(?:Bad Request|Unauthorized|Payment Required|Forbidden|Not Found|"
    r"Method Not Allowed|Gone|Unprocessable(?: Entity| Content)?|Too Many Requests|"
    r"Internal Server Error|Bad Gateway|Service Unavailable|Gateway Timeout)\b",
    re.IGNORECASE)


def status_from_text(text: str | None) -> int | None:
    """The HTTP status named in an error message, for frameworks that pass
    tool errors on as text. None unless the message names it explicitly."""
    if not text:
        return None
    m = _STATUS_LABELLED.search(text) or _STATUS_REASON.search(text)
    return int(m.group(1)) if m else None


def classify(status: int | None, error_type: str | None, terminal_statuses, transient_statuses,
             terminal_exceptions=()) -> str:
    """terminal, transient or unknown."""
    if error_type and error_type in terminal_exceptions:
        return "terminal"
    if status in terminal_statuses:
        return "terminal"
    if status in transient_statuses:
        return "transient"
    return "unknown"
