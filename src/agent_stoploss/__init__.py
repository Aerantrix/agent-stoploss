"""agent-stoploss: stop AI agents from burning tokens on broken tools."""

__version__ = "0.1.0"

from .config import Config, ConfigError  # noqa: E402
from .engine import Decision, Verdict  # noqa: E402
from .guard import StopLoss  # noqa: E402
from .run import Run, StopEvent, Stopped, StreamingNotSupported, ToolFailed, current_run  # noqa: E402

__all__ = [
    "StopLoss", "Stopped", "StopEvent", "ToolFailed", "StreamingNotSupported",
    "Run", "current_run", "Config", "ConfigError", "Decision", "Verdict",
]
