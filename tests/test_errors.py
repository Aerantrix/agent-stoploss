import urllib.error

import pytest

from agent_stoploss.errors import classify, status_from_text, status_of


class Resp:
    status_code = 503


class WithResponse(Exception):
    response = Resp()


class BrokenProperty(Exception):
    @property
    def response(self):
        raise RuntimeError("not set")


def test_status_from_common_shapes():
    e = Exception()
    e.status_code = 401
    assert status_of(e) == 401
    e2 = Exception()
    e2.status = "404"
    assert status_of(e2) == 404
    assert status_of(WithResponse()) == 503
    assert status_of(urllib.error.HTTPError("u", 410, "gone", {}, None)) == 410
    assert status_of(ValueError("x")) is None
    assert status_of(BrokenProperty()) is None


def test_bool_is_not_a_status():
    e = Exception()
    e.status = True
    assert status_of(e) is None


def test_classify():
    term, trans = {401, 404}, {503}
    assert classify(401, "X", term, trans) == "terminal"
    assert classify(503, "X", term, trans) == "transient"
    assert classify(None, "X", term, trans) == "unknown"
    assert classify(None, "AuthError", term, trans, {"AuthError"}) == "terminal"


@pytest.mark.parametrize("text,status", [
    ("HTTP 401 Unauthorized", 401),
    ("Error code: 404 - {'error': 'not found'}", 404),
    ("status_code=429", 429),
    ("upstream said 503 Service Unavailable", 503),
    ("HTTP/1.1 500", 500),
    ("found 404 files", None),          # a count, not a status
    ("listening on port 8080", None),
    ("Tool error: Invalid or expired credentials", None),
    ("", None),
    (None, None),
])
def test_status_from_text(text, status):
    assert status_from_text(text) == status
