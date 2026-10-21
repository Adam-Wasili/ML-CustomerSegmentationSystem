"""Application factory.

create_app() builds and configures the Flask application. Using a function
(instead of creating the app at the top of the file) lets the tests create their
own separate app with a temporary database. Later steps register more blueprints here.
"""
import os

from flask import Flask
from flask_login import LoginManager
from flask_wtf.csrf import CSRFProtect

from config import Config
from models import User, db

login_manager = LoginManager()   # handles logged-in users
csrf = CSRFProtect()             # protects every form against forged requests


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Make sure the folders the app writes to exist
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "database"), exist_ok=True)

    # Connect the extensions to this app
    db.init_app(app)
    csrf.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"          # where users who are not logged in are sent
    login_manager.login_message = "Please log in to access this page."
    login_manager.login_message_category = "warning"

    @login_manager.user_loader
    def load_user(user_id):
        """Flask-Login calls this on every request to find the logged-in user."""
        return db.session.get(User, int(user_id))

    # Blueprints: each module of the system has its own group of routes
    from routes.auth import auth_bp
    from routes.dashboard import dashboard_bp
    from routes.upload import upload_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(upload_bp)
    # Step 3+: rfm_bp, clustering_bp, analytics_bp will be added here

    # Create the database tables if they do not exist yet
    with app.app_context():
        db.create_all()

    return app


if __name__ == "__main__":
    # Running "python app.py" starts the development server
    create_app().run(debug=True)