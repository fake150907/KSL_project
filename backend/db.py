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
                    branch_id   TEXT,
                    name        TEXT,
                    dob         TEXT,
                    gender      TEXT,
                    phone       TEXT,
                    created_at  TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                    ended_at    TEXT
                )
            """)
            # 비식별 수어 인식 로그 (PII 없음) — 분석/모델 개선용
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    branch_id       TEXT,
                    created_at      TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
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
            # 상담 메타데이터 (비식별) — 시나리오/소요시간/처리결과. consultation_id로 다른 데이터 연결
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consultations (
                    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                    branch_id          TEXT,
                    citizen_session_id INTEGER,
                    scenario           TEXT,
                    status             TEXT NOT NULL DEFAULT '진행중',
                    result             TEXT,
                    started_at         TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                    ended_at           TEXT,
                    duration_sec       INTEGER
                )
            """)
            # 오인식 수정 이력 (비식별) — 모델 예측 vs 사람이 고친 정답. 재학습용 라벨
            conn.execute("""
                CREATE TABLE IF NOT EXISTS prediction_corrections (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at        TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                    prediction_log_id INTEGER,
                    consultation_id   INTEGER,
                    predicted_label   TEXT,
                    corrected_label   TEXT,
                    confidence        REAL,
                    corrected_by      TEXT,
                    model_type        TEXT
                )
            """)
            # 상담 대화 기록 (PII 가능성 → 본문 암호화). 보존기간 후 purge_old_data로 삭제
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consultation_messages (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at      TEXT NOT NULL DEFAULT (datetime('now', 'localtime')),
                    consultation_id INTEGER,
                    message_id      TEXT,
                    sender          TEXT,
                    text_enc        TEXT
                )
            """)
            # 지점(동사무소) 계정 — 지점별 로그인. branch_id가 모든 데이터의 소속 지점이 됨
            conn.execute("""
                CREATE TABLE IF NOT EXISTS branches (
                    branch_id      TEXT PRIMARY KEY,
                    name           TEXT NOT NULL,
                    region         TEXT,
                    username       TEXT UNIQUE NOT NULL,
                    password_hash  TEXT NOT NULL,
                    created_at     TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
                )
            """)
            # 기존 DB 호환: branch_id 컬럼이 없으면 추가 (데이터 보존 마이그레이션).
            # fresh DB는 이미 위 CREATE에 컬럼이 있어 건너뛴다 (idempotent).
            for _table in ("citizen_sessions", "prediction_logs", "consultations"):
                _cols = [r[1] for r in conn.execute(f"PRAGMA table_info({_table})").fetchall()]
                if "branch_id" not in _cols:
                    conn.execute(f"ALTER TABLE {_table} ADD COLUMN branch_id TEXT")
                    print(f"[db] 마이그레이션: {_table}.branch_id 컬럼 추가")
        print(f"[db] SQLite 테이블 초기화 완료 ({_db_path()})")
        seed_demo_branches()  # 데모 지점 계정 (비어있을 때만)
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
    """AES-256-GCM으로 암호화 후 base64 반환.

    fail-closed: 키가 없거나 암호화에 실패하면 PII를 평문으로 저장하지 않도록
    예외를 발생시킨다. (호출부의 try/except가 받아서 저장을 안전하게 중단함)
    """
    if not value:
        return value
    key = _get_encryption_key()
    if key is None:
        raise RuntimeError(
            "DB_ENCRYPTION_KEY가 설정되지 않아 암호화할 수 없습니다. 평문 저장을 막기 위해 중단합니다."
        )
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    aesgcm = AESGCM(key)
    nonce = os.urandom(12)
    encrypted = aesgcm.encrypt(nonce, value.encode("utf-8"), None)
    return base64.b64encode(nonce + encrypted).decode("utf-8")


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


def save_citizen_session(data: dict[str, Any], branch_id: str | None = None) -> int | None:
    """민원인 세션을 마스킹 + AES-256 암호화 후 로컬 DB에 저장하고 생성된 id 반환."""
    conn = get_connection()
    if conn is None:
        return None
    try:
        masked_name  = _mask_name(str(data.get("name") or ""))
        masked_phone = _mask_phone(str(data.get("phone") or ""))
        masked_dob   = _mask_dob(str(data.get("dob") or ""))
        # 암호화는 try 안에서 — 키가 없으면 여기서 예외 → 평문 저장 없이 안전하게 중단
        stored_name  = encrypt_value(masked_name)
        stored_phone = encrypt_value(masked_phone)
        stored_dob   = encrypt_value(masked_dob)
        gender       = str(data.get("gender") or "")
        with _lock:
            cur = conn.execute(
                """
                INSERT INTO citizen_sessions (branch_id, name, dob, gender, phone)
                VALUES (?, ?, ?, ?, ?)
                """,
                (branch_id, stored_name, stored_dob, gender, stored_phone),
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
                "UPDATE citizen_sessions SET ended_at=datetime('now', 'localtime') WHERE id=?",
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


def save_prediction_log(prediction: dict[str, Any], client_id: str = "default", branch_id: str | None = None) -> int | None:
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
                    (branch_id, client_id, model_type, raw_label, display_label, confidence,
                     below_threshold, segment_frames, process_ms, scenario_mode, scenario_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    branch_id,
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


# ─────────────────────────────────────────────
# 상담 메타데이터 (consultations) — 비식별, 평문
# ─────────────────────────────────────────────
def save_consultation(citizen_session_id: int | None = None, scenario: str | None = None, branch_id: str | None = None) -> int | None:
    """상담 1건 시작 기록. 생성된 consultation id 반환."""
    conn = get_connection()
    if conn is None:
        return None
    try:
        with _lock:
            cur = conn.execute(
                "INSERT INTO consultations (branch_id, citizen_session_id, scenario) VALUES (?, ?, ?)",
                (branch_id, citizen_session_id, scenario),
            )
            return cur.lastrowid
    except Exception as exc:
        print(f"[db] 상담 저장 실패: {exc}")
        return None


def end_consultation(consultation_id: int, result: str = "완료") -> None:
    """상담 종료 처리. ended_at/소요시간(duration_sec)/처리결과 기록."""
    conn = get_connection()
    if conn is None or consultation_id is None:
        return
    try:
        with _lock:
            conn.execute(
                """
                UPDATE consultations
                SET ended_at = datetime('now', 'localtime'),
                    status   = '완료',
                    result   = ?,
                    duration_sec = CAST((julianday(datetime('now', 'localtime')) - julianday(started_at)) * 86400 AS INTEGER)
                WHERE id = ?
                """,
                (result, consultation_id),
            )
    except Exception as exc:
        print(f"[db] 상담 종료 업데이트 실패: {exc}")


def get_recent_consultations(limit: int = 100) -> list[dict[str, Any]]:
    """최근 상담 메타데이터 반환 (대시보드/통계용)."""
    conn = get_connection()
    if conn is None:
        return []
    try:
        with _lock:
            cur = conn.execute(
                """
                SELECT id, citizen_session_id, scenario, status, result,
                       started_at, ended_at, duration_sec
                FROM consultations
                ORDER BY started_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            cols = ["id", "citizen_session_id", "scenario", "status", "result",
                    "started_at", "ended_at", "duration_sec"]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as exc:
        print(f"[db] 상담 조회 실패: {exc}")
        return []


# ─────────────────────────────────────────────
# 오인식 수정 이력 (prediction_corrections) — 비식별, 평문
# ─────────────────────────────────────────────
def save_correction(
    predicted_label: str,
    corrected_label: str,
    confidence: float | None = None,
    corrected_by: str | None = None,
    model_type: str | None = None,
    prediction_log_id: int | None = None,
    consultation_id: int | None = None,
) -> int | None:
    """모델 예측 vs 사람이 고친 정답 1건 기록. 재학습용 라벨 데이터."""
    conn = get_connection()
    if conn is None:
        return None
    try:
        with _lock:
            cur = conn.execute(
                """
                INSERT INTO prediction_corrections
                    (prediction_log_id, consultation_id, predicted_label, corrected_label,
                     confidence, corrected_by, model_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prediction_log_id, consultation_id,
                    str(predicted_label) if predicted_label is not None else None,
                    str(corrected_label) if corrected_label is not None else None,
                    float(confidence) if confidence is not None else None,
                    corrected_by, model_type,
                ),
            )
            return cur.lastrowid
    except Exception as exc:
        print(f"[db] 수정 이력 저장 실패: {exc}")
        return None


def get_recent_corrections(limit: int = 100) -> list[dict[str, Any]]:
    """최근 오인식 수정 이력 반환."""
    conn = get_connection()
    if conn is None:
        return []
    try:
        with _lock:
            cur = conn.execute(
                """
                SELECT id, created_at, prediction_log_id, consultation_id,
                       predicted_label, corrected_label, confidence, corrected_by, model_type
                FROM prediction_corrections
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            cols = ["id", "created_at", "prediction_log_id", "consultation_id",
                    "predicted_label", "corrected_label", "confidence", "corrected_by", "model_type"]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as exc:
        print(f"[db] 수정 이력 조회 실패: {exc}")
        return []


# ─────────────────────────────────────────────
# 상담 대화 기록 (consultation_messages) — PII 가능 → 본문 암호화
# ─────────────────────────────────────────────
def save_message(consultation_id: int | None, sender: str, text: str, message_id: str | None = None) -> int | None:
    """상담 대화 1건 저장. 본문(text)은 AES-256으로 암호화하여 저장."""
    conn = get_connection()
    if conn is None:
        return None
    try:
        text_enc = encrypt_value(str(text or ""))
        with _lock:
            cur = conn.execute(
                """
                INSERT INTO consultation_messages (consultation_id, message_id, sender, text_enc)
                VALUES (?, ?, ?, ?)
                """,
                (consultation_id, message_id, sender, text_enc),
            )
            return cur.lastrowid
    except Exception as exc:
        print(f"[db] 대화 저장 실패: {exc}")
        return None


def get_consultation_messages(consultation_id: int, limit: int = 500) -> list[dict[str, Any]]:
    """특정 상담의 대화 기록 반환. 본문은 복호화하여 text로 제공."""
    conn = get_connection()
    if conn is None:
        return []
    try:
        with _lock:
            cur = conn.execute(
                """
                SELECT id, created_at, consultation_id, message_id, sender, text_enc
                FROM consultation_messages
                WHERE consultation_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (consultation_id, limit),
            )
            rows = cur.fetchall()
        result = []
        for r in rows:
            result.append({
                "id": r[0], "created_at": r[1], "consultation_id": r[2],
                "message_id": r[3], "sender": r[4], "text": decrypt_value(r[5]),
            })
        return result
    except Exception as exc:
        print(f"[db] 대화 조회 실패: {exc}")
        return []


# ─────────────────────────────────────────────
# 보존기간 정책 — 오래된 PII 자동 삭제 (retention)
# ─────────────────────────────────────────────
def purge_old_data(days: int = 30) -> dict[str, int]:
    """보존기간(days)이 지난 PII 데이터 삭제. 비식별 분석 데이터는 보존.

    삭제 대상: citizen_sessions(민원인 정보), consultation_messages(대화 기록).
    """
    conn = get_connection()
    if conn is None:
        return {"citizen_sessions": 0, "consultation_messages": 0}
    cutoff = f"-{int(days)} days"
    try:
        with _lock:
            c1 = conn.execute(
                "DELETE FROM consultation_messages WHERE created_at < datetime('now', 'localtime', ?)",
                (cutoff,),
            ).rowcount
            c2 = conn.execute(
                "DELETE FROM citizen_sessions WHERE created_at < datetime('now', 'localtime', ?)",
                (cutoff,),
            ).rowcount
        print(f"[db] 보존기간({days}일) 경과 데이터 삭제: 대화 {c1}건, 민원인 {c2}건")
        return {"consultation_messages": c1, "citizen_sessions": c2}
    except Exception as exc:
        print(f"[db] 보존기간 삭제 실패: {exc}")
        return {"citizen_sessions": 0, "consultation_messages": 0}


# ─────────────────────────────────────────────
# 지점(branches) 계정 — 지점별 로그인 / 목록 / 시드
# ─────────────────────────────────────────────
def _hash_pw(password: str) -> str:
    import hashlib
    return hashlib.sha256(str(password).encode("utf-8")).hexdigest()


def get_branch_by_username(username: str) -> dict[str, Any] | None:
    """username으로 지점 계정 조회 (로그인 검증용)."""
    conn = get_connection()
    if conn is None:
        return None
    try:
        with _lock:
            cur = conn.execute(
                "SELECT branch_id, name, region, username, password_hash FROM branches WHERE username = ?",
                (str(username),),
            )
            row = cur.fetchone()
        if not row:
            return None
        cols = ["branch_id", "name", "region", "username", "password_hash"]
        return dict(zip(cols, row))
    except Exception as exc:
        print(f"[db] 지점 조회 실패: {exc}")
        return None


def verify_branch_login(username: str, password: str) -> dict[str, Any] | None:
    """지점 로그인 검증. 성공 시 {branch_id, name, region}, 실패 시 None."""
    branch = get_branch_by_username(username)
    if branch is None:
        return None
    if branch.get("password_hash") != _hash_pw(password):
        return None
    return {"branch_id": branch["branch_id"], "name": branch["name"], "region": branch.get("region")}


def list_branches() -> list[dict[str, Any]]:
    """지점 목록 (비밀번호 제외)."""
    conn = get_connection()
    if conn is None:
        return []
    try:
        with _lock:
            cur = conn.execute(
                "SELECT branch_id, name, region, username, created_at FROM branches ORDER BY branch_id"
            )
            cols = ["branch_id", "name", "region", "username", "created_at"]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as exc:
        print(f"[db] 지점 목록 조회 실패: {exc}")
        return []


def upsert_branch(branch_id: str, name: str, username: str, password: str, region: str | None = None) -> bool:
    """지점 계정 생성/갱신 (비밀번호는 해시 저장)."""
    conn = get_connection()
    if conn is None:
        return False
    try:
        with _lock:
            conn.execute(
                """
                INSERT INTO branches (branch_id, name, region, username, password_hash)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(branch_id) DO UPDATE SET
                    name=excluded.name, region=excluded.region,
                    username=excluded.username, password_hash=excluded.password_hash
                """,
                (branch_id, name, region, username, _hash_pw(password)),
            )
        return True
    except Exception as exc:
        print(f"[db] 지점 저장 실패: {exc}")
        return False


def seed_demo_branches() -> None:
    """데모용 지점 계정 시드 (branches 비어있을 때만)."""
    if list_branches():
        return
    demo = [
        ("seocho-01", "서초구 서초1동", "seocho", "seocho1234", "서울특별시 서초구"),
        ("gangnam-01", "강남구 역삼1동", "gangnam", "gangnam1234", "서울특별시 강남구"),
    ]
    for bid, name, user, pw, region in demo:
        upsert_branch(bid, name, user, pw, region)
    print(f"[db] 데모 지점 계정 {len(demo)}개 시드 완료")
