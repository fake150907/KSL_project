"""PostgreSQL 연결 및 테이블 초기화."""
from __future__ import annotations

import base64
import os
import threading
from typing import Any

_conn = None
_lock = threading.Lock()


def get_connection():
    """싱글턴 DB 커넥션 반환. DATABASE_URL 없으면 None."""
    global _conn
    database_url = os.environ.get("DATABASE_URL", "")
    if not database_url:
        return None

    with _lock:
        try:
            if _conn is None or _conn.closed:
                import psycopg2
                _conn = psycopg2.connect(database_url, sslmode="require")
                _conn.autocommit = True
        except Exception as exc:
            print(f"[db] 연결 실패: {exc}")
            _conn = None
    return _conn


def init_db() -> bool:
    """citizen_sessions 테이블 생성 (없으면). 성공 시 True."""
    conn = get_connection()
    if conn is None:
        print("[db] DATABASE_URL 없음 — DB 저장 비활성화")
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS citizen_sessions (
                    id          SERIAL PRIMARY KEY,
                    name        TEXT,
                    dob         TEXT,
                    gender      TEXT,
                    phone       TEXT,
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    ended_at    TIMESTAMPTZ
                )
            """)
        print("[db] 테이블 초기화 완료")
        return True
    except Exception as exc:
        print(f"[db] 테이블 초기화 실패: {exc}")
        return False


def _get_encryption_key() -> bytes | None:
    """DB_ENCRYPTION_KEY 환경변수에서 32바이트 AES-256 키 로드."""
    key_hex = os.environ.get("DB_ENCRYPTION_KEY", "")
    if len(key_hex) < 64:
        return None
    try:
        return bytes.fromhex(key_hex[:64])
    except Exception:
        return None


def encrypt_value(value: str) -> str:
    """AES-256-GCM으로 암호화 후 base64 반환. 키 없으면 원본 반환."""
    if not value:
        return value
    key = _get_encryption_key()
    if key is None:
        return value
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)
        encrypted = aesgcm.encrypt(nonce, value.encode("utf-8"), None)
        return base64.b64encode(nonce + encrypted).decode("utf-8")
    except Exception as exc:
        print(f"[db] 암호화 실패: {exc}")
        return value


def decrypt_value(value: str) -> str:
    """AES-256-GCM 복호화. 키 없거나 실패 시 원본 반환."""
    if not value:
        return value
    key = _get_encryption_key()
    if key is None:
        return value
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        raw = base64.b64decode(value.encode("utf-8"))
        nonce, ciphertext = raw[:12], raw[12:]
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ciphertext, None).decode("utf-8")
    except Exception:
        return value


def _mask_name(name: str) -> str:
    """홍길동 → 홍x동"""
    name = (name or "").strip()
    if len(name) <= 1:
        return name
    if len(name) == 2:
        return f"{name[0]}x"
    return f"{name[0]}{'x' * (len(name) - 2)}{name[-1]}"


def _mask_phone(phone: str) -> str:
    """010-1234-5678 → 010-****-5678"""
    d = "".join(c for c in (phone or "") if c.isdigit())[:11]
    if len(d) <= 3:
        return d
    if len(d) <= 7:
        return f"{d[:3]}-{'*' * (len(d) - 3)}"
    middle_len = 3 if len(d) == 10 else 4
    return f"{d[:3]}-{'*' * middle_len}-{d[3 + middle_len:]}"


def _mask_dob(dob: str) -> str:
    """990101 → 99****"""
    dob = (dob or "").strip()
    if len(dob) <= 2:
        return dob
    return f"{dob[:2]}{'*' * (len(dob) - 2)}"


def save_citizen_session(data: dict[str, Any]) -> int | None:
    """민원인 세션을 마스킹 + AES-256 암호화 후 DB에 저장하고 생성된 id 반환."""
    conn = get_connection()
    if conn is None:
        return None
    masked_name  = _mask_name(str(data.get("name") or ""))
    masked_phone = _mask_phone(str(data.get("phone") or ""))
    masked_dob   = _mask_dob(str(data.get("dob") or ""))
    stored_name  = encrypt_value(masked_name)
    stored_phone = encrypt_value(masked_phone)
    stored_dob   = encrypt_value(masked_dob)
    gender       = str(data.get("gender") or "")
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO citizen_sessions (name, dob, gender, phone)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (stored_name, stored_dob, gender, stored_phone),
            )
            row = cur.fetchone()
            return row[0] if row else None
    except Exception as exc:
        print(f"[db] 세션 저장 실패: {exc}")
        return None


def end_citizen_session(session_id: int) -> None:
    """민원인 세션 종료 처리 (ended_at, status 업데이트)."""
    conn = get_connection()
    if conn is None or session_id is None:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE citizen_sessions SET ended_at=NOW() WHERE id=%s",
                (session_id,),
            )
    except Exception as exc:
        print(f"[db] 세션 종료 업데이트 실패: {exc}")


def get_recent_sessions(limit: int = 50) -> list[dict[str, Any]]:
    """최근 민원인 세션 목록 반환."""
    conn = get_connection()
    if conn is None:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, name, dob, gender, phone, created_at, ended_at
                FROM citizen_sessions
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            cols = ["id", "name", "dob", "gender", "phone", "created_at", "ended_at"]
            return [
                {col: (str(val) if val is not None else None) for col, val in zip(cols, row)}
                for row in cur.fetchall()
            ]
    except Exception as exc:
        print(f"[db] 세션 조회 실패: {exc}")
        return []
