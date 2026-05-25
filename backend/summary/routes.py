from __future__ import annotations

import traceback

from flask import Blueprint, jsonify, request

from summary.ai_client import summarize

summary_bp = Blueprint("summary", __name__)

@summary_bp.route("/api/summary", methods=["POST"])
def create_summary():
    data = request.get_json(silent=True) or {}
    conversation = data.get("conversation", [])

    if not isinstance(conversation, list):
        return jsonify({"error": "conversation은 문자열 리스트여야 합니다."}), 400
    if not conversation:
        return jsonify({"error": "요약할 대화 내용이 없습니다."}), 400

    try:
        result_text = summarize([str(item) for item in conversation])
        return jsonify({"summary": result_text}), 200
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 502
