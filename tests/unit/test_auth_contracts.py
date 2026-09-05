"""Authentication contracts using an isolated in-memory database."""
import pytest
from flask import Flask, session
import auth
from models import db, OTPToken

@pytest.fixture
def context():
    app = Flask(__name__)
    app.config.update(SECRET_KEY="test-only", SQLALCHEMY_DATABASE_URI="sqlite:///:memory:")
    db.init_app(app)
    with app.app_context():
        db.create_all()
        with app.test_request_context():
            yield app
        db.session.remove()
        db.drop_all()

def test_token_replacement_and_single_use(context, monkeypatch):
    monkeypatch.setattr(auth, "generate_otp", lambda: "012345")
    old = auth.create_otp_token("a@example.test")
    other = auth.create_otp_token("b@example.test")
    new = auth.create_otp_token("a@example.test")
    assert old.used and not other.used
    assert auth.verify_otp("a@example.test", "wrong") is None
    assert auth.verify_otp("a@example.test", "012345") is new
    assert auth.verify_otp("a@example.test", "012345") is None

def test_expired_token(context):
    token = OTPToken("a@example.test", "123456", expiration_minutes=-1)
    db.session.add(token)
    db.session.commit()
    assert auth.verify_otp(token.email, token.otp_code) is None
    assert not token.used

def test_user_uniqueness_and_session(context):
    assert auth.get_current_user() is None
    assert not auth.is_authenticated()
    assert auth.get_or_create_user("a@example.test") is None
    user, error = auth.create_user("a@example.test", "Alice")
    assert error is None
    assert auth.create_user("b@example.test", "Alice") == (None, "Display name already taken")
    assert auth.create_user("a@example.test", "Other") == (None, "Email already registered")
    assert auth.get_or_create_user(user.email) is user
    auth.login_user(user)
    assert auth.get_current_user() is user
    assert auth.is_authenticated() and session.permanent
    assert session["display_name"] == "Alice"
    session["unrelated"] = 1
    auth.logout_user()
    auth.logout_user()
    assert not auth.is_authenticated()
    assert session["unrelated"] == 1

def test_auth_decorator(context):
    @auth.require_auth
    def protected(value):
        return value
    response, status = protected("ok")
    assert status == 401
    assert response.json == {"error": "Authentication required"}
    user, _ = auth.create_user("a@example.test", "Alice")
    auth.login_user(user)
    assert protected("ok") == "ok"
    assert protected.__name__ == "protected"

def test_otp_leading_zero(monkeypatch):
    digits = iter([0, 1, 2, 3, 4, 5])
    monkeypatch.setattr(auth.random, "randint", lambda a, b: next(digits))
    assert auth.generate_otp() == "012345"
