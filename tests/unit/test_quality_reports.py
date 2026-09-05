"""Regression coverage for metric arithmetic and navigation edge cases."""
import pytest

from quality_report import crap_score, report
from activity_utils import resolve_conditional_navigation, create_completion_skip_thinking
from unittest.mock import MagicMock


@pytest.mark.parametrize("cc,covered,expected", [(10, 0, 110), (10, 1, 10), (10, .5, 22.5)])
def test_crap_formula(cc, covered, expected):
    assert crap_score(cc, covered) == expected


@pytest.mark.parametrize("covered", [-.1, 1.1])
def test_invalid_fraction(covered):
    with pytest.raises(ValueError):
        crap_score(2, covered)


def test_methods_are_not_duplicated(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text("class A:\n    def run(self):\n        return 1\n")
    path = str(source)
    rows = report([path], {path: {"executed_lines": [1, 2], "missing_lines": [3]}})
    assert len(rows) == 1
    assert rows[0]["coverage"] == 50
    assert rows[0]["crap"] == 1.12
    with pytest.raises(ValueError, match="Missing coverage"):
        report([path], {})


@pytest.mark.parametrize("branches,expected", [
    ([{}, {"else": True, "goto": "fallback"}], "fallback"),
    ([{"if": {"x": 2}, "else": True, "goto": "wrong"}], None),
    ([{"elif": {"x": 1}, "goto": "yes"}], "yes"),
    ([{"if": {}, "elif": {"x": 2}, "goto": "yes"}], "yes"),
    (None, None),
])
def test_navigation_precedence(branches, expected):
    assert resolve_conditional_navigation(branches, {"x": 1}) == expected


def test_completion_retry_and_passthrough():
    client = MagicMock()
    create = client.chat.completions.create
    error = RuntimeError("unsupported chat_template_kwargs")
    error.status_code = 400
    create.side_effect = [error, "ok"]
    assert create_completion_skip_thinking(client, model="test") == "ok"
    assert create.call_args.kwargs == {"model": "test"}
    create.side_effect = RuntimeError("unrelated")
    with pytest.raises(RuntimeError, match="unrelated"):
        create_completion_skip_thinking(client, model="test")
