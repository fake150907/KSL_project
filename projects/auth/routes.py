from __future__ import annotations

import hashlib
import time
from functools import wraps

from flask import Blueprint, jsonify, request, session

from config import Config

auth_bp = Blueprint("auth", __name__)

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def is_logged_in() -> bool:
    if not session.get("admin_logged_in"):
        return False
    if time.time() > session.get("expire_at", 0):
        session.clear()
        return False
    return True


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not is_logged_in():
            return jsonify({
                "error": "Unauthorized",
                "message": "관리자 로그인이 필요합니다.",
            }), 401
        return f(*args, **kwargs)
    return decorated

@auth_bp.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "JSON body가 필요합니다."}), 400

    username = data.get("username", "").strip()
    password = data.get("password", "")

    if not username or not password:
        return jsonify({"error": "username과 password를 모두 입력해주세요."}), 400

    if username != Config.ADMIN_USERNAME or _hash_password(password) != Config.ADMIN_PASSWORD_HASH:
        return jsonify({"error": "아이디 또는 비밀번호가 올바르지 않습니다."}), 401

    session.clear()
    session["admin_logged_in"] = True
    session["admin_username"] = username
    session["login_time"] = time.time()
    session["expire_at"] = time.time() + Config.SESSION_TIMEOUT
    session.permanent = False

    print(f"[AUTH] 로그인 성공: {username}", flush=True)
    return jsonify({
        "message": "로그인 성공",
        "username": username,
        "session_timeout": Config.SESSION_TIMEOUT,
    }), 200


@auth_bp.route("/api/logout", methods=["POST"])
def logout():
    username = session.get("admin_username", "unknown")
    session.clear()
    print(f"[AUTH] 로그아웃: {username}", flush=True)
    return jsonify({"message": "로그아웃 되었습니다."}), 200


@auth_bp.route("/api/auth/status", methods=["GET"])
def auth_status():
    """현재 세션 상태 반환."""
    if is_logged_in():
        remaining = int(Config.SESSION_TIMEOUT - (time.time() - session.get("login_time", 0)))
        return jsonify({
            "authenticated": True,
            "username": session.get("admin_username"),
            "session_remaining_seconds": max(remaining, 0),
        }), 200
    return jsonify({"authenticated": False}), 200
