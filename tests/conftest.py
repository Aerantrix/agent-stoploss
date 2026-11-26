import pytest

from agent_stoploss import StopLoss
from agent_stoploss.journal import JournalWriter


class Sink(list):
    def __call__(self, event):
        self.append(event)

    def of(self, ev):
        return [e for e in self if e["ev"] == ev]


@pytest.fixture
def sink():
    return Sink()


@pytest.fixture
def make_guard(sink):
    def make(mode="shadow", **settings):
        return StopLoss({**settings, "mode": mode}, journal=JournalWriter(sink=sink))
    return make


class HTTPError(Exception):
    def __init__(self, status):
        super().__init__(f"HTTP {status}")
        self.status_code = status


class Unavailable(Exception):
    """A connection failure: no status at all."""
