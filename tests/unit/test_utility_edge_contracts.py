"""Regression coverage for utility boundary inputs."""
import pytest

from activity_utils import (
    filter_content_blocks,
    resolve_conditional_navigation,
    strip_reasoning,
)


def test_content_blocks_ignore_unsupported_values():
    assert filter_content_blocks([None, 42, [], "hello {{username}}", {}], {},
                                 {"username": "Ada"}) == ["hello Ada", ""]


@pytest.mark.parametrize("value", [None, 42, {}, ()])
def test_navigation_unsupported_values_have_no_destination(value):
    assert resolve_conditional_navigation(value, {}) is None


@pytest.mark.parametrize("text", ["\u003cthink>\u003c/think>", "\u003cthink>   "])
def test_empty_reasoning_has_nothing_to_salvage(text):
    assert strip_reasoning(text) == ""
