from __future__ import annotations

import time

from flask import Blueprint, jsonify, request

from auth.routes import login_required
from session.store import (
    add_message,
    clear_citizen_session,
    clear_messages,
    clear_session_state,
    get_citizen_session,
    get_messages,
    get_session_state,
    set_citizen_session,
    set_session_ended,
)

session_bp = Blueprint("session", __name__)


@session_bp.route("/api/citizen-session", methods=["GET", "POST", "DELETE"])
@login_required
def api_citizen_session():
    if request.method == "GET":
        return jsonify(get_citizen_session()), 200

    if request.method == "DELETE":
        return jsonify(clear_citizen_session()), 200

    data = request.get_json(silent=True) or {}
    citizen_data = data.get("citizenData") or data.get("citizen_data") or data
    if not isinstance(citizen_data, dict):
        return jsonify({"error": "citizenData must be an object"}), 400

    if not str(citizen_data.get("name", "")).strip() or not str(citizen_data.get("phone", "")).strip():
        return jsonify({"error": "citizen name and phone are required"}), 400

    return jsonify(set_citizen_session(citizen_data)), 200


@session_bp.route("/api/messages", methods=["GET", "POST", "DELETE"])
@login_required
def api_messages():
    if request.method == "GET":
        return jsonify({"messages": get_messages()}), 200

    if request.method == "DELETE":
        clear_messages()
        return jsonify({"messages": []}), 200

    data = request.get_json(silent=True) or {}
    message_id = str(data.get("id", "")).strip()
    sender = str(data.get("sender", "")).strip()
    text = str(data.get("text", "")).strip()

    if not message_id or sender not in {"citizen", "agent"} or not text:
        return jsonify({"error": "id, sender, and text are required"}), 400

    message = {
        "id": message_id,
        "sender": sender,
        "text": text,
        "timestamp": str(data.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        "label": str(data.get("label", "")).strip(),
    }

    result = add_message(message)
    if result is None:
        return jsonify({"message": message}), 200

    return jsonify({"message": result}), 200


@session_bp.route("/api/session-state", methods=["GET", "POST", "DELETE"])
@login_required
def api_session_state():
    if request.method == "GET":
        return jsonify(get_session_state()), 200

    if request.method == "DELETE":
        return jsonify(clear_session_state()), 200

    data = request.get_json(silent=True) or {}
    return jsonify(set_session_ended(bool(data.get("ended")))), 200
