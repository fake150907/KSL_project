"""로컬 SQLite 저장소 + 테이블 초기화.

동사무소 지점별 로컬 운영을 가정한 설계:
- 민원인 PII(이름/생일/전화)는 이 지점 로컬 SQLite 파일에만 저장(마스킹 + AES-256 암호화).
- 수어 인식 로그는 비식별 데이터라 별도 테이블에 저장 → 추후 중앙 집계/모델 개선용.

DB 파일 경로는 KSL_DB_PATH 환경변수로 바꿀 수 있고, 기본값은 backend/ksl_local.db.
"""
from __future__ import annotations

import base64
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

_conn: sqlite3.Connection | None = None
_lock = threading.Lock()


def _db_path() -> str:
    """로컬 SQLite 파일 경로. KSL_DB_PATH로 덮어쓸 수 있음."""
    env_path = os.environ.get("KSL_DB_PATH", "").strip()
    if env_path:
        return env_path
    return str(Path(__file__).resolve().parent / "ksl_local.db")


def get_connection() -> sqlite3.Connection | None:
    """싱글턴 SQLite 커넥션 반환. 실패 시 None."""
    global _conn
    with _lock:
        try:
            if _conn is None:
                path = _db_path()
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                # check_same_thread=False: Flask 멀티스레드에서 공유. 쓰기는 _lock으로 직렬화.
                _conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
                _conn.execute("PRAGMA journal_mode=WAL")
        except Exception as exc:
            print(f"[db] 연결 실패: {exc}")
            _conn = None
    return _conn


def init_db() -> bool:
    """테이블 생성 (없으면). 성공 시 True."""
    conn = get_connection()
    if conn is None:
        print("[db] SQLite 연결 실패 — DB 저장 비활성화")
        return False
    try:
        with _lock:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS citizen_sessions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT,
                    dob         TEXT,
                    gender      TEXT,
                    phone       TEXT,
                    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                    ended_at    TEXT
                )
            """)
            # 비식별 수어 인식 로그 (PII 없음) — 분석/모델 개선용
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                    client_id       TEXT,
                    model_type      TEXT,
                    raw_label       TEXT,
                    display_label   TEXT,
                    confidence      REAL,
                    below_threshold INTEGER,
                    segment_frames  INTEGER,
                    process_ms      REAL,
                    scenario_mode   INTEGER,
                    scenario_text   TEXT
                )
            """)
        print(f"[db] SQLite 테이블 초기화 완료 ({_db_path()})")
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
    """민원인 세션을 마스킹 + AES-256 암호화 후 로컬 DB에 저장하고 생성된 id 반환."""
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
        with _lock:
            cur = conn.execute(
                """
                INSERT INTO citizen_sessions (name, dob, gender, phone)
                VALUES (?, ?, ?, ?)
                """,
                (stored_name, stored_dob, gender, stored_phone),
            )
            return cur.lastrowid
    except Exception as exc:
        print(f"[db] 세션 저장 실패: {exc}")
        return None


def end_citizen_session(session_id: int) -> None:
    """민원인 세션 종료 처리 (ended_at 업데이트)."""
    conn = get_connection()
    if conn is None or session_id is None:
        return
    try:
        with _lock:
            conn.execute(
                "UPDATE citizen_sessions SET ended_at=datetime('now') WHERE id=?",
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
        with _lock:
            cur = conn.execute(
                """
                SELECT id, name, dob, gender, phone, created_at, ended_at
                FROM citizen_sessions
                ORDER BY created_at DESC
                LIMIT ?
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


def save_prediction_log(prediction: dict[str, Any], client_id: str = "default") -> int | None:
    """비식별 수어 인식 로그 1건 저장. prediction(예측 결과 dict)에서 필요한 필드만 추출.

    PII는 저장하지 않음 — 라벨/신뢰도/지연시간 등 분석·모델 개선용 데이터만 기록.
    """
    conn = get_connection()
    if conn is None:
        return None
    try:
        with _lock:
            cur = conn.execute(
                """
                INSERT INTO prediction_logs
                    (client_id, model_type, raw_label, display_label, confidence,
                     below_threshold, segment_frames, process_ms, scenario_mode, scenario_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(client_id),
                    prediction.get("model_type"),
                    prediction.get("raw_label"),
                    prediction.get("display_label"),
                    float(prediction.get("confidence") or 0.0),
                    1 if prediction.get("below_threshold") else 0,
                    prediction.get("segment_frames"),
                    prediction.get("process_ms"),
                    1 if prediction.get("scenario_mode") else 0,
                    prediction.get("scenario_text"),
                ),
            )
            return cur.lastrowid
    except Exception as exc:
        print(f"[db] 인식 로그 저장 실패: {exc}")
        return None


def get_recent_predictions(limit: int = 100) -> list[dict[str, Any]]:
    """최근 수어 인식 로그 반환 (분석/대시보드용)."""
    conn = get_connection()
    if conn is None:
        return []
    try:
        with _lock:
            cur = conn.execute(
                """
                SELECT id, created_at, client_id, model_type, raw_label, display_label,
                       confidence, below_threshold, segment_frames, process_ms,
                       scenario_mode, scenario_text
                FROM prediction_logs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            cols = ["id", "created_at", "client_id", "model_type", "raw_label",
                    "display_label", "confidence", "below_threshold", "segment_frames",
                    "process_ms", "scenario_mode", "scenario_text"]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as exc:
        print(f"[db] 인식 로그 조회 실패: {exc}")
        return []
