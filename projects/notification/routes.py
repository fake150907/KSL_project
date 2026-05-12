from __future__ import annotations

from flask import Blueprint, jsonify, request

from auth.routes import login_required
from notification.email_sender import send_email

notification_bp = Blueprint("notification", __name__)


@notification_bp.route("/api/notify/email", methods=["POST"])
@login_required
def notify_email():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body가 필요합니다."}), 400

    to_address: str = data.get("to", "").strip()
    summary: str    = data.get("summary", "").strip()

    if not to_address:
        return jsonify({"error": "수신자 이메일(to)이 필요합니다."}), 400
    if "@" not in to_address:
        return jsonify({"error": "유효하지 않은 이메일 주소입니다."}), 400
    if not summary:
        return jsonify({"error": "전송할 요약 내용(summary)이 없습니다."}), 400

    try:
        send_email(to_address, summary)
        return jsonify({"message": f"이메일 전송 완료 → {to_address}"}), 200
    except ValueError as exc:
        # Gmail 설정 누락
        return jsonify({"error": str(exc)}), 500
    except RuntimeError as exc:
        # 전송 실패
        return jsonify({"error": str(exc)}), 502
    except Exception as exc:
        return jsonify({"error": f"알 수 없는 오류: {exc}"}), 500
