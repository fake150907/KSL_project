"""PostgreSQL 연결 및 테이블 초기화."""
from __future__ import annotations

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


def save_citizen_session(data: dict[str, Any]) -> int | None:
    """민원인 세션을 DB에 저장하고 생성된 id 반환."""
    conn = get_connection()
    if conn is None:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO citizen_sessions (name, dob, gender, phone)
                VALUES (%s, %s, %s, %s)
                RETURNING id
                """,
                (data.get("name"), data.get("dob"), data.get("gender"), data.get("phone")),
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
