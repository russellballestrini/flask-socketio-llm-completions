"""Edge-case contracts for condition evaluation."""

import pytest

from activity_utils import evaluate_condition


@pytest.mark.parametrize("suffix", ["gt", "gte", "lt", "lte"])
@pytest.mark.parametrize("actual, expected", [(None, 1), ("bad", 1), (1, None), (1, "bad")])
def test_invalid_numeric_operands(suffix, actual, expected):
    assert evaluate_condition({"x": actual}, "x_" + suffix, expected) is False


@pytest.mark.parametrize("suffix, expected", [("gt", False), ("gte", True), ("lt", False), ("lte", True)])
def test_missing_numeric_value_defaults_to_zero(suffix, expected):
    assert evaluate_condition({}, "x_" + suffix, "0") is expected


@pytest.mark.parametrize("bounds", [(0, 2), [], [0, 1, 2], [None, 2], [0, None]])
def test_invalid_between_bounds(bounds):
    assert evaluate_condition({"x": 1}, "x_between", bounds) is False


def test_between_invalid_actual_and_missing_default():
    assert evaluate_condition({"x": None}, "x_between", [0, 2]) is False
    assert evaluate_condition({}, "x_between", [0, 0]) is True


def test_between_does_not_convert_upper_bound_when_lower_fails():
    class ExplodingFloat:
        def __float__(self):
            raise RuntimeError("must not convert")

    assert evaluate_condition({"x": 1}, "x_between", [2, ExplodingFloat()]) is False
    with pytest.raises(RuntimeError, match="must not convert"):
        evaluate_condition({"x": 1}, "x_between", [0, ExplodingFloat()])


@pytest.mark.parametrize("value", [None, False, 0, "", [], "false", [1]])
@pytest.mark.parametrize("metadata", [{}, {"x": None}])
@pytest.mark.parametrize("suffix", ["exists", "not_exists"])
def test_existence_uses_truthiness_and_presence(metadata, suffix, value):
    expected = ("x" in metadata) == bool(value)
    if suffix == "not_exists":
        expected = not expected
    assert evaluate_condition(metadata, "x_" + suffix, value) is expected


@pytest.mark.parametrize("metadata, value, expected", [
    ({"x": " a, , b ,, "}, "a", True),
    ({"x": " a, , b ,, "}, " a", False),
    ({"x": " , "}, "", False),
    ({}, "", False),
    ({"x": None}, None, True),
    ({"x": 12}, 12, True),
])
def test_contains_stringification_and_whitespace(metadata, value, expected):
    assert evaluate_condition(metadata, "x_contains", value) is expected
    assert evaluate_condition(metadata, "x_not_contains", value) is not expected


def test_suffix_dispatch_and_equality_defaults():
    assert evaluate_condition({}, "missing", None) is True
    assert evaluate_condition({}, "missing_ne", None) is False
    assert evaluate_condition({"x_unknown": 3}, "x_unknown", 3) is True
    assert evaluate_condition({"x_gt": 3}, "x_gt_ne", 4) is True
    assert evaluate_condition({"": 3}, "_gte", 3) is True


@pytest.mark.parametrize("metadata, pattern, expected", [({}, "^$", True), ({"x": None}, "None", True), ({"x": 123}, 2, True), ({"x": "abc"}, "[", False)])
def test_regex_stringifies_and_searches(metadata, pattern, expected):
    assert evaluate_condition(metadata, "x_matches", pattern) is expected
