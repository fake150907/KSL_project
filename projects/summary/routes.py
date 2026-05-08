from __future__ import annotations

from flask import Blueprint, jsonify, request

from auth.routes import login_required
from summary.ai_client import summarize

summary_bp = Blueprint("summary", __name__)


@summary_bp.route("/api/summary", methods=["POST"])
@login_required
def create_summary():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body가 필요합니다."}), 400

    conversation: list[str] = data.get("conversation", [])
    if not conversation:
        return jsonify({"error": "conversation 리스트가 비어있습니다."}), 400
    if not isinstance(conversation, list):
        return jsonify({"error": "conversation은 문자열 리스트여야 합니다."}), 400

    try:
        summary_text = summarize(conversation)
        return jsonify({"summary": summary_text}), 200
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502
    except Exception as exc:
        return jsonify({"error": f"요약 중 오류가 발생했습니다: {exc}"}), 500
