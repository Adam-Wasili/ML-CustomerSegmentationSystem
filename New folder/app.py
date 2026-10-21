from functools import wraps
from pathlib import Path
import sqlite3

from flask import Flask, flash, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "database" / "customer_segmentation.db"

app = Flask(__name__)
app.config["SECRET_KEY"] = "*123456789#"


def get_db_connection():
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    with get_db_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS user_account (
                user_id INTEGER PRIMARY KEY CHECK (user_id = 1),
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL
            )
        """)


def create_initial_account():
    # This creates only one local account on the first run.
    with get_db_connection() as connection:
        existing = connection.execute(
            "SELECT user_id FROM user_account LIMIT 1"
        ).fetchone()
        if existing is None:
            connection.execute(
                "INSERT INTO user_account (user_id, username, password_hash) VALUES (?, ?, ?)",
                (1, "admin", generate_password_hash("12345"))
            )


def login_required(view_function):
    @wraps(view_function)
    def wrapped_view(*args, **kwargs):
        if session.get("logged_in") is not True:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return view_function(*args, **kwargs)
    return wrapped_view


@app.route("/")
def home():
    if session.get("logged_in") is True:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in") is True:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Enter both username and password.", "error")
            return render_template("login.html")

        with get_db_connection() as connection:
            account = connection.execute(
                "SELECT user_id, username, password_hash FROM user_account WHERE username = ?",
                (username,)
            ).fetchone()

        if account and check_password_hash(account["password_hash"], password):
            session.clear()
            session["logged_in"] = True
            session["username"] = account["username"]
            return redirect(url_for("dashboard"))

        flash("Incorrect username or password.", "error")

    return render_template("login.html")


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", username=session.get("username", "User"))


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    session.clear()
    flash("You have logged out successfully.", "success")
    return redirect(url_for("login"))


if __name__ == "__main__":
    initialize_database()
    create_initial_account()
    app.run(host="127.0.0.1", port=5000, debug=True)
