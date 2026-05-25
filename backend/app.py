# -*- coding: utf-8 -*-
from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
import sys
from pathlib import Path

from flask import Flask
from flask_cors import CORS

from config import Config
from auth.routes import auth_bp
from inference.routes import inference_bp
from notification.routes import notification_bp
from session.routes import session_bp
from summary.routes import summary_bp
from welfare.routes import welfare_bp

app = Flask(__name__)
app.secret_key = Config.FLASK_SECRET_KEY
CORS(app, supports_credentials=True)

app.register_blueprint(auth_bp)
app.register_blueprint(inference_bp)
app.register_blueprint(notification_bp)
app.register_blueprint(session_bp)
app.register_blueprint(summary_bp)
app.register_blueprint(welfare_bp)

if __name__ == "__main__":
    from inference.model_loader import load_models
    load_models()
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False, threaded=True)
