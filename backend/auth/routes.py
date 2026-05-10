from __future__ import annotations

import hashlib
import time
from functools import wraps
from typing import Any, Callable

from flask import Blueprint, jsonify, request, session

from config import Config

auth_bp = Blueprint("auth", __name__)


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def is_logged_in() -> bool:
    if not session.get("admin_logged_in"):
        return False
    if time.time() > float(session.get("expire_at", 0)):
        session.clear()
        return False
    return True


def login_required(fn: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(fn)
    def decorated(*args: Any, **kwargs: Any):
        if not is_logged_in():
            return jsonify({"error": "Unauthorized", "message": "관리자 로그인이 필요합니다."}), 401
        return fn(*args, **kwargs)

    return decorated


@auth_bp.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", ""))

    if not username or not password:
        return jsonify({"error": "username과 password를 모두 입력해 주세요."}), 400

    if username != Config.ADMIN_USERNAME or _hash_password(password) != Config.ADMIN_PASSWORD_HASH:
        return jsonify({"error": "아이디 또는 비밀번호가 올바르지 않습니다."}), 401

    now = time.time()
    session.clear()
    session["admin_logged_in"] = True
    session["admin_username"] = username
    session["login_time"] = now
    session["expire_at"] = now + Config.SESSION_TIMEOUT
    session.permanent = False

    return jsonify({"message": "로그인 성공", "username": username, "session_timeout": Config.SESSION_TIMEOUT}), 200


@auth_bp.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"message": "로그아웃 되었습니다."}), 200


@auth_bp.route("/api/auth/status", methods=["GET"])
def auth_status():
    if not is_logged_in():
        return jsonify({"authenticated": False}), 200

    remaining = int(float(session.get("expire_at", 0)) - time.time())
    return jsonify({
        "authenticated": True,
        "username": session.get("admin_username"),
        "session_remaining_seconds": max(remaining, 0),
    }), 200

