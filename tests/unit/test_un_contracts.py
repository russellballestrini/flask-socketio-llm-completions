"""Offline SDK transport and polling contracts."""
from unittest.mock import Mock
import pytest
import un

@pytest.mark.parametrize("method", ["GET", "POST", "PATCH", "DELETE"])
def test_transport(monkeypatch, method):
    response = Mock()
    response.json.return_value = {"ok": True}
    transport = Mock(return_value=response)
    monkeypatch.setattr(un.requests, method.lower(), transport)
    monkeypatch.setattr(un.time, "time", lambda: 123)
    assert un._make_request(method, "/test", "public", "secret", {"x": 1}) == {"ok": True}
    args, kwargs = transport.call_args
    assert args == (un.API_BASE + "/test",)
    assert kwargs["timeout"] == 120
    assert kwargs["headers"]["Authorization"] == "Bearer public"
    assert kwargs["headers"]["X-Timestamp"] == "123"
    assert kwargs["headers"]["X-Signature"] == un._sign_request("secret", 123, method, "/test", '{"x": 1}')
    if method in ("POST", "PATCH"):
        assert kwargs["json"] == {"x": 1}
    response.raise_for_status.assert_called_once()

def test_unsupported_method():
    with pytest.raises(ValueError, match="Unsupported HTTP method"):
        un._make_request("PUT", "/test", "public", "secret")

@pytest.mark.parametrize("status", ["completed", "failed", "timeout", "cancelled"])
def test_poll_terminal_status(monkeypatch, status):
    sleep = Mock()
    monkeypatch.setattr(un.time, "sleep", sleep)
    get_job = Mock(side_effect=[{"status": "running"}, {"status": status}])
    monkeypatch.setattr(un, "get_job", get_job)
    assert un.wait_for_job("job", "public", "secret") == {"status": status}
    assert [c.args[0] for c in sleep.call_args_list] == [0.3, 0.45]
    get_job.assert_called_with("job", "public", "secret")

def test_poll_timeout(monkeypatch):
    job = Mock()
    monkeypatch.setattr(un, "get_job", job)
    with pytest.raises(TimeoutError):
        un.wait_for_job("job", "public", "secret", timeout=0)
    job.assert_not_called()

@pytest.mark.parametrize("name,expected", [("main.PY", "python"), ("a.test.cpp", "cpp"), ("", None), ("README", None), ("x.unknown", None)])
def test_language_detection(name, expected):
    assert un.detect_language(name) == expected

def test_credentials_priority(monkeypatch, tmp_path):
    monkeypatch.setenv("UNSANDBOX_PUBLIC_KEY", "env-public")
    monkeypatch.setenv("UNSANDBOX_SECRET_KEY", "env-secret")
    assert un._resolve_credentials("explicit", "secret") == ("explicit", "secret")
    assert un._resolve_credentials() == ("env-public", "env-secret")
    monkeypatch.delenv("UNSANDBOX_PUBLIC_KEY")
    monkeypatch.delenv("UNSANDBOX_SECRET_KEY")
    monkeypatch.delenv("UNSANDBOX_ACCOUNT", raising=False)
    monkeypatch.setattr(un, "_get_unsandbox_dir", lambda: tmp_path)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(un.CredentialsError):
        un._resolve_credentials()
    (tmp_path / "accounts.csv").write_text("public,secret\n")
    assert un._resolve_credentials() == ("public", "secret")
