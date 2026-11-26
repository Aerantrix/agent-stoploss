from dataclasses import dataclass

from agent_stoploss.fingerprint import call_print, normalise_result, result_print, shape


def test_call_print_ignores_key_order_case_and_spaces():
    assert call_print("t", {"domain": "A.com ", "period": "30d"}) == \
        call_print("t", {"period": "30d", "domain": "a.com"})


def test_distinct_arguments_stay_distinct():
    prints = {call_print("analytics", {"domain": f"site{i}.com"}) for i in range(1, 41)}
    assert len(prints) == 40
    pages = {call_print("list", {"page": p}) for p in range(1, 21)}
    assert len(pages) == 20


def test_tool_name_is_part_of_the_print():
    assert call_print("a", {"x": 1}) != call_print("b", {"x": 1})


def test_result_print_drops_echoed_arguments():
    a = result_print("t", {"q": "one"}, {"q": "one", "results": []})
    b = result_print("t", {"q": "two"}, {"q": "two", "results": []})
    assert a == b


def test_volatile_fields_uuids_and_timestamps_are_ignored():
    r1 = {"results": [], "meta": {"request_id": "abc", "timestamp": 1}, "id": "6f1c2a9e-1b2c-4d5e-8f90-123456789abc",
          "generated": "2026-11-25T10:00:01Z"}
    r2 = {"results": [], "meta": {"request_id": "xyz", "timestamp": 2}, "id": "0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d",
          "generated": "2026-11-25T10:00:09Z"}
    vol = ["request_id", "timestamp"]
    assert result_print("t", {}, r1, vol) == result_print("t", {}, r2, vol)


def test_bare_dates_are_kept():
    assert normalise_result({"day": "2026-11-01"}, {}, []) != normalise_result({"day": "2026-11-02"}, {}, [])


def test_json_strings_are_parsed():
    assert result_print("t", {}, '{"b": 1, "a": 2}') == result_print("t", {}, {"a": 2, "b": 1})


def test_objects_are_converted():
    @dataclass
    class R:
        rows: list

    class Pyd:
        def model_dump(self):
            return {"rows": [1]}

    assert result_print("t", {}, R([1])) == result_print("t", {}, Pyd()) == result_print("t", {}, {"rows": [1]})


def test_shape_has_no_values():
    s = shape({"secret": "customer@example.com", "n": 5})
    assert s["keys"] == ["n", "secret"]
    assert "customer" not in str(s)
