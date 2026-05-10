from __future__ import annotations

from flask import Blueprint, jsonify, request

from auth.routes import login_required
from config import Config
from notification.email_sender import send_email
from notification.kakao_sender import send_kakao_message

notification_bp = Blueprint("notification", __name__)


@notification_bp.route("/api/notify/email", methods=["POST"])
@login_required
def notify_email():
    data = request.get_json(silent=True) or {}
    to_address = str(data.get("to", "")).strip()
    summary = str(data.get("summary", "")).strip()

    if not to_address or "@" not in to_address:
        return jsonify({"error": "유효한 수신 이메일 주소가 필요합니다."}), 400
    if not summary:
        return jsonify({"error": "전송할 요약 내용이 없습니다."}), 400

    try:
        send_email(to_address, summary)
        return jsonify({"message": f"메일 전송 완료: {to_address}"}), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 500
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502


@notification_bp.route("/api/notify/kakao", methods=["POST"])
@login_required
def notify_kakao():
    data = request.get_json(silent=True) or {}
    access_token = str(data.get("access_token", "")).strip() or Config.KAKAO_ACCESS_TOKEN
    summary = str(data.get("summary", "")).strip()

    if not access_token:
        return jsonify({"error": "카카오 액세스 토큰(access_token)이 필요합니다."}), 400
    if not summary:
        return jsonify({"error": "전송할 요약 내용이 없습니다."}), 400

    try:
        send_kakao_message(access_token, summary)
        return jsonify({"message": "카카오 나에게 전송 완료"}), 200
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 500
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502
