"""Registration, login and logout (Module 1)."""
import re

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from models import User, db

auth_bp = Blueprint("auth", __name__)

# Usernames: 3-30 characters, only letters, numbers and underscore
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_]{3,30}$")
MIN_PASSWORD_LENGTH = 8


def _is_safe_redirect(target):
    """Allow redirects only to pages inside this site.

    Without this check an attacker could send a link such as
    /login?next=https://evil.com and send users to another website after login.
    """
    return bool(target) and target.startswith("/") \
        and not target.startswith("//") and "\\" not in target


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    """Create a new account."""
    if current_user.is_authenticated:          # already logged in: nothing to register
        return redirect(url_for("dashboard.index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")

        # Collect every problem so the user can fix them all at once
        errors = []
        if not USERNAME_PATTERN.match(username):
            errors.append("Username must be 3-30 characters: letters, numbers or underscore.")
        if len(password) < MIN_PASSWORD_LENGTH:
            errors.append(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
        if password != confirm:
            errors.append("Passwords do not match.")
        if not errors and User.query.filter_by(username=username).first():
            errors.append("That username is already taken.")

        if errors:
            for message in errors:
                flash(message, "danger")
            return render_template("register.html", username=username), 400

        # Save the new user; only the password HASH is stored
        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash("Account created. You can now log in.", "success")
        return redirect(url_for("auth.login"))

    return render_template("register.html", username="")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    """Log a user in."""
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))

    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()

        # One generic message for both cases, so attackers cannot find out
        # which usernames exist.
        if user is None or not user.check_password(password):
            flash("Invalid username or password.", "danger")
            return render_template("login.html", username=username), 401

        login_user(user)
        # If the user was sent here from a protected page, return them there
        next_page = request.args.get("next")
        if _is_safe_redirect(next_page):
            return redirect(next_page)
        return redirect(url_for("dashboard.index"))

    return render_template("login.html", username="")


@auth_bp.route("/logout", methods=["POST"])
@login_required
def logout():
    """Log the user out. POST only, so a link on another site cannot log users out."""
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("auth.login"))