from __future__ import annotations

from flask import Flask
from flask_cors import CORS

from config import Config
from auth.routes import auth_bp
from vision.routes import vision_bp
from summary.routes import summary_bp
from notification.routes import notification_bp
from vision.model_loader import load_models


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = Config.FLASK_SECRET_KEY

    CORS(app, supports_credentials=True)

    app.register_blueprint(auth_bp)
    app.register_blueprint(vision_bp)
    app.register_blueprint(summary_bp)
    app.register_blueprint(notification_bp)

    return app


if __name__ == "__main__":
    app = create_app()
    load_models()
    app.run(debug=False, host="127.0.0.1", port=5000, use_reloader=False, threaded=True)
