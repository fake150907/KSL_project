from __future__ import annotations

import hashlib
import time
from functools import wraps
from typing import Any, Callable

from flask import Blueprint, jsonify, request, session
from werkzeug.security import check_password_hash

from config import Config

auth_bp = Blueprint("auth", __name__)

MAX_LOGIN_FAILURES = 5
LOGIN_FAILURE_WINDOW_SECONDS = 10 * 60
LOGIN_LOCK_SECONDS = 15 * 60
_login_failures: dict[str, list[float]] = {}
_login_locks: dict[str, float] = {}


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _verify_password(password: str, stored_hash: str) -> bool:
    if not stored_hash:
        return False
    # Werkzeug hashes look like "scrypt:...$..." or "pbkdf2:...$...".
    # Keep SHA-256 compatibility so existing env/SQLite accounts do not break.
    if "$" in stored_hash and ":" in stored_hash.split("$", 1)[0]:
        try:
            return check_password_hash(stored_hash, password)
        except Exception:
            return False
    return _hash_password(password) == stored_hash


def _login_key(username: str) -> str:
    return f"{request.remote_addr or 'unknown'}:{username.lower()}"


def _is_login_locked(key: str) -> int:
    until = _login_locks.get(key, 0.0)
    remaining = int(until - time.time())
    if remaining > 0:
        return remaining
    _login_locks.pop(key, None)
    return 0


def _record_login_failure(key: str) -> None:
    now = time.time()
    recent = [ts for ts in _login_failures.get(key, []) if now - ts < LOGIN_FAILURE_WINDOW_SECONDS]
    recent.append(now)
    _login_failures[key] = recent
    if len(recent) >= MAX_LOGIN_FAILURES:
        _login_locks[key] = now + LOGIN_LOCK_SECONDS
        _login_failures.pop(key, None)


def _clear_login_failures(key: str) -> None:
    _login_failures.pop(key, None)
    _login_locks.pop(key, None)


def is_logged_in() -> bool:
    if not session.get("admin_logged_in"):
        return False
    if time.time() > float(session.get("expire_at", 0)):
        session.clear()
        return False
    session["expire_at"] = time.time() + Config.SESSION_TIMEOUT
    session.modified = True
    return True


def current_branch_id() -> str | None:
    """로그인한 세션의 지점 ID. 데이터/소켓 분리에 사용."""
    return session.get("branch_id")


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

    key = _login_key(username)
    retry_after = _is_login_locked(key)
    if retry_after:
        return jsonify({
            "error": "로그인 실패 횟수가 많습니다. 잠시 후 다시 시도해 주세요.",
            "retry_after": retry_after,
        }), 429

    import db as _db
    # 1) 지점 계정 검증
    branch = _db.verify_branch_login(username, password)
    if branch is None:
        # 2) (호환) 기존 관리자 계정 폴백 — branch_id 없는 슈퍼관리자
        if username == Config.ADMIN_USERNAME and _verify_password(password, Config.ADMIN_PASSWORD_HASH):
            branch = {"branch_id": None, "name": "관리자", "region": None}
        else:
            _record_login_failure(key)
            return jsonify({"error": "아이디 또는 비밀번호가 올바르지 않습니다."}), 401

    now = time.time()
    session.clear()
    session["admin_logged_in"] = True
    session["admin_username"] = username
    session["branch_id"] = branch["branch_id"]
    session["branch_name"] = branch["name"]
    session["login_time"] = now
    session["expire_at"] = now + Config.SESSION_TIMEOUT
    session.permanent = False
    _clear_login_failures(key)

    return jsonify({
        "message": "로그인 성공",
        "username": username,
        "branch_id": branch["branch_id"],
        "branch_name": branch["name"],
        "session_timeout": Config.SESSION_TIMEOUT,
    }), 200


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
        "branch_id": session.get("branch_id"),
        "branch_name": session.get("branch_name"),
        "session_remaining_seconds": max(remaining, 0),
    }), 200

