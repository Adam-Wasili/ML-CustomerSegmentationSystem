"""Tests for registration, login, logout and page protection."""
import pytest

from app import create_app
from config import TestingConfig
from models import User, db


@pytest.fixture
def client():
    """A test browser with a fresh, empty database for every test."""
    app = create_app(TestingConfig)
    with app.app_context():
        db.create_all()
        yield app.test_client()
        db.session.remove()
        db.drop_all()


def register(client, username="adam", password="strongpass1", confirm=None):
    """Submit the registration form (confirm defaults to the same password)."""
    return client.post("/register", data={
        "username": username, "password": password,
        "confirm_password": confirm if confirm is not None else password})


def login(client, username="adam", password="strongpass1"):
    """Submit the login form."""
    return client.post("/login", data={"username": username, "password": password})


def test_dashboard_requires_login(client):
    # A visitor who is not logged in is redirected to the login page
    response = client.get("/dashboard")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_register_and_login_success(client):
    assert register(client).status_code == 302          # redirected to the login page
    assert login(client).status_code == 302             # redirected to the dashboard
    assert client.get("/dashboard").status_code == 200  # the dashboard is now accessible


def test_password_is_hashed_not_plain(client):
    register(client)
    user = User.query.filter_by(username="adam").first()
    assert user.password_hash != "strongpass1"          # the plain password is never stored
    assert user.check_password("strongpass1")           # but the hash still verifies it


def test_wrong_password_rejected(client):
    register(client)
    assert login(client, password="wrongpass99").status_code == 401


def test_short_password_rejected(client):
    assert register(client, password="short").status_code == 400
    assert User.query.count() == 0                      # no account was created


def test_password_mismatch_rejected(client):
    assert register(client, confirm="different123").status_code == 400


def test_duplicate_username_rejected(client):
    register(client)
    assert register(client).status_code == 400


def test_logout(client):
    register(client)
    login(client)
    assert client.post("/logout").status_code == 302    # logged out, redirected to login
    assert client.get("/dashboard").status_code == 302  # the dashboard is protected again