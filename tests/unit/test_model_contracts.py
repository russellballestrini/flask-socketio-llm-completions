"""Model edge cases without application startup or external services."""
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
import models

@pytest.mark.parametrize("used,delta,expected", [(False, 60, True), (False, -60, False), (True, 60, False)])
def test_token_validity(used, delta, expected):
    token = models.OTPToken("a@example.test", "012345")
    token.used = used
    token.expires_at = datetime.utcnow() + timedelta(seconds=delta)
    assert token.is_valid() is expected
    assert "a@example.test" in repr(token)

def test_room_membership_transitions():
    room = models.Room(name="test")
    assert room.get_active_users() == room.get_inactive_users() == []
    room.remove_user("missing")
    room.add_user("z")
    room.add_user("a")
    room.add_user("z")
    assert room.get_active_users() == ["a", "z"]
    room.remove_user("z")
    assert room.get_inactive_users() == ["z"]
    room.add_user("z")
    assert room.get_inactive_users() == []

@pytest.mark.parametrize("content,expected", [("", False), ("plain", False), ('<img src="data:image/png;base64,AA">', True), ("data:image/png;base64,AA", False), ("<img data:image/png", False)])
def test_image_detection(monkeypatch, content, expected):
    monkeypatch.setattr(models, "TIKTOKEN_AVAILABLE", False)
    message = models.Message("user", content, 1)
    assert message.is_base64_image() is expected
    assert message.token_count == (0 if expected else len(content) // 4 + 1)

def test_tokenizer_error_and_cached_count(monkeypatch):
    encode = Mock(side_effect=RuntimeError("offline"))
    monkeypatch.setattr(models, "TIKTOKEN_AVAILABLE", True)
    monkeypatch.setattr(models, "tiktoken", SimpleNamespace(encoding_for_model=encode))
    message = models.Message("user", "hello", 1)
    assert message.token_count == 2
    assert message.count_tokens() == 2
    encode.assert_called_once_with("gpt-4")

def test_metadata_roundtrip():
    state = models.ActivityState()
    assert state.dict_metadata == {}
    state.add_metadata("unicode", "雪")
    state.add_metadata("nested", {"items": [1, True]})
    state.remove_metadata("missing")
    assert state.dict_metadata == {"unicode": "雪", "nested": {"items": [1, True]}}
    state.remove_metadata("unicode")
    assert "unicode" not in state.dict_metadata
    state.clear_metadata()
    assert state.dict_metadata == {}
