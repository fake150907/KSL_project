from __future__ import annotations

from flask import Blueprint, jsonify, request

from auth.routes import login_required
from summary.ai_client import summarize

summary_bp = Blueprint("summary", __name__)


@summary_bp.route("/api/summary", methods=["POST"])
@login_required
def create_summary():
    data = request.get_json(silent=True) or {}
    conversation = data.get("conversation", [])

    if not isinstance(conversation, list):
        return jsonify({"error": "conversation은 문자열 리스트여야 합니다."}), 400
    if not conversation:
        return jsonify({"error": "요약할 대화 내용이 없습니다."}), 400

    try:
        return jsonify({"summary": summarize([str(item) for item in conversation])}), 200
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 502

