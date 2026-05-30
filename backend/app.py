# -*- coding: utf-8 -*-
from __future__ import annotations
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import os
import sys
from pathlib import Path

from flask import Flask, request as flask_request
from flask_cors import CORS

from config import Config
from auth.routes import auth_bp
from inference.routes import inference_bp
from notification.routes import notification_bp
from session.routes import session_bp
from summary.routes import summary_bp
from welfare.routes import welfare_bp
from logs.routes import logs_bp
from logs.store import push_log
from inference.model_state import mediapipe_landmarks_to_frame
from inference.predictor import frames_to_model_tensor, frames_to_sentence_v2_tensor, smooth_segment_frames, softmax

SENTENCE_T_MAX = 128
POSE_POINTS_V2 = 33
HAND_POINTS_V2 = 21

app = Flask(__name__)
app.secret_key = Config.FLASK_SECRET_KEY
CORS(app, supports_credentials=True)

app.register_blueprint(auth_bp)
app.register_blueprint(inference_bp)
app.register_blueprint(notification_bp)
app.register_blueprint(session_bp)
app.register_blueprint(summary_bp)
app.register_blueprint(welfare_bp)
app.register_blueprint(logs_bp)


@app.after_request
def _log_non_2xx(response):
    """2xx(성공) 응답을 제외한 모든 HTTP 요청을 로그에 기록."""
    status = response.status_code
    path = flask_request.path

    # 로그 엔드포인트 자체는 기록하지 않음 (무한루프 방지)
    if path.startswith("/api/logs"):
        return response

    # 2xx 성공 응답은 제외
    if 200 <= status < 300:
        return response

    method = flask_request.method
    if status >= 500:
        level = "error"
    elif status >= 400:
        level = "warning"
    else:
        level = "info"

    message = f"{method} {path} → {status}"
    push_log(level, "Backend", message, status=status, method=method, path=path)
    return response

if __name__ == "__main__":
    from inference.model_loader import ensure_models_loaded
    ensure_models_loaded()
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False, threaded=True)
