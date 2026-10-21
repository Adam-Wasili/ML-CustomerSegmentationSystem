"""Dashboard routes. Currently a placeholder that proves login protection works.

The full dashboard (totals and charts) is built in a later step.
"""
from flask import Blueprint, redirect, render_template, url_for
from flask_login import current_user, login_required

dashboard_bp = Blueprint("dashboard", __name__)


@dashboard_bp.route("/")
def root():
    """The home address sends users to the dashboard or to the login page."""
    if current_user.is_authenticated:
        return redirect(url_for("dashboard.index"))
    return redirect(url_for("auth.login"))


@dashboard_bp.route("/dashboard")
@login_required        # users who are not logged in are redirected to the login page
def index():
    return render_template("dashboard.html")