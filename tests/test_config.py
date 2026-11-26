import pytest

from agent_stoploss import Config, ConfigError


def test_defaults():
    c = Config.from_dict({})
    assert c.mode == "shadow"
    assert c.max_tokens is None and c.max_cost_usd is None
    assert c.loop_threshold == 3
    assert (c.breaker_window, c.breaker_max_failures) == (5, 4)
    assert 401 in c.terminal_statuses and 503 in c.transient_statuses
    assert c.notice == "agent-stoploss: {reason}."


def test_unknown_keys_are_errors_and_all_are_reported():
    with pytest.raises(ConfigError) as e:
        Config.from_dict({"mdoe": "enforce", "loop_detector": {"treshold": 4},
                          "run_budget": {"max_tokens": -5}})
    text = str(e.value)
    assert "mdoe: unknown setting" in text
    assert "loop_detector.treshold: unknown setting" in text


def test_value_problems_are_all_reported():
    with pytest.raises(ConfigError) as e:
        Config.from_dict({"mode": "on", "run_budget": {"max_tokens": 0},
                          "circuit_breaker": {"window": 3, "max_failures": 4},
                          "notice": "no placeholder"})
    assert len(e.value.problems) == 4


def test_section_off_and_tool_overrides(tmp_path):
    f = tmp_path / "stoploss.yaml"
    f.write_text("mode: enforce\n"
                 "loop_detector: off\n"
                 "tools:\n"
                 "  job_status: {circuit_breaker: off}\n"
                 "  search: {loop_detector: {threshold: 5}}\n"
                 "prices: {my-model: {input: 1, output: 2}}\n")
    c = Config.from_file(f)
    assert c.mode == "enforce"
    assert not c.loop_enabled
    assert not c.rules_for("job_status").breaker_enabled
    assert c.rules_for("other").breaker_enabled
    assert c.rules_for("search").loop_threshold == 5
    assert c.prices["my-model"]["output"] == 2


def test_tool_override_typos():
    with pytest.raises(ConfigError) as e:
        Config.from_dict({"tools": {"x": {"loop": False, "circuit_breaker": {"windw": 3}}}})
    assert len(e.value.problems) == 2


def test_status_in_both_lists():
    with pytest.raises(ConfigError):
        Config.from_dict({"error_classifier": {"transient_statuses": [401]}})


def test_hash_ignores_mode_and_logging():
    a = Config.from_dict({"mode": "shadow"})
    b = Config.from_dict({"mode": "enforce", "log": {"dir": None}})
    c = Config.from_dict({"loop_detector": {"threshold": 4}})
    assert a.hash == b.hash != c.hash
