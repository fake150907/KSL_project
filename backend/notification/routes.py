from __future__ import annotations

from flask import Blueprint, jsonify, request

from auth.routes import login_required
from config import Config
from notification.kakao_sender import KakaoAccessTokenError, refresh_kakao_access_token, send_kakao_message

notification_bp = Blueprint("notification", __name__)


@notification_bp.route("/api/notify/kakao", methods=["POST"])
@login_required
def notify_kakao():
    data = request.get_json(silent=True) or {}
    access_token = str(data.get("access_token", "")).strip() or Config.KAKAO_ACCESS_TOKEN
    refresh_token = str(data.get("refresh_token", "")).strip() or Config.KAKAO_REFRESH_TOKEN
    summary = str(data.get("summary", "")).strip()

    if not summary:
        return jsonify({"error": "전송할 민원 상담 요약 내용이 없습니다."}), 400
    if not access_token and not refresh_token:
        return jsonify({"error": "카카오 access_token 또는 refresh_token이 필요합니다. 먼저 카카오 OAuth 로그인을 완료해주세요."}), 400

    try:
        refreshed: dict[str, str] = {}
        if not access_token and refresh_token:
            refreshed = refresh_kakao_access_token(refresh_token)
            access_token = refreshed["access_token"]

        try:
            send_kakao_message(access_token, summary)
        except KakaoAccessTokenError:
            if not refresh_token:
                raise
            refreshed = refresh_kakao_access_token(refresh_token)
            access_token = refreshed["access_token"]
            send_kakao_message(access_token, summary)

        payload: dict[str, str] = {"message": "카카오톡 나에게 보내기 완료"}
        if refreshed.get("access_token"):
            payload["access_token"] = refreshed["access_token"]
        if refreshed.get("refresh_token"):
            payload["refresh_token"] = refreshed["refresh_token"]
        return jsonify(payload), 200

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 500
    except KakaoAccessTokenError as exc:
        return jsonify({"error": f"{exc} 카카오 OAuth 로그인을 다시 진행해주세요."}), 502
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502
