"""Application configuration.

Sensitive values (such as the secret key) come from environment variables, so
they are never written in the code that is shared or submitted.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# The folder that contains this file (the project root). Building paths from it
# means the project works on any computer without hard-coded paths.
BASE_DIR = Path(__file__).resolve().parent

# Read SECRET_KEY (and any other setting) from the .env file if it exists
load_dotenv(BASE_DIR / ".env")


class Config:
    """Settings used when the application runs normally."""

    # Used by Flask to sign login sessions and CSRF tokens.
    # The fallback is for local development only: always set a real key in .env.
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-change-me")

    # SQLite database file stored in the database/ folder
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + str(BASE_DIR / "database" / "app.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False   # turns off an unneeded feature

    # Where uploaded files are saved, and the limits on them
    UPLOAD_FOLDER = str(BASE_DIR / "uploads")
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024    # reject uploads larger than 10 MB
    ALLOWED_EXTENSIONS = {"csv", "xlsx"}     # nothing else may be uploaded

    # Session cookie hardening
    SESSION_COOKIE_HTTPONLY = True           # JavaScript cannot read the cookie
    SESSION_COOKIE_SAMESITE = "Lax"          # reduces cross-site request attacks


class TestingConfig(Config):
    """Settings used by the automated tests."""

    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"   # temporary database, thrown away after each test
    WTF_CSRF_ENABLED = False                         # lets tests submit forms without a CSRF token
    SECRET_KEY = "test-secret-key"