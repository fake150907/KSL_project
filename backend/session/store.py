from __future__ import annotations

import threading
import time
from typing import Any

import db as _db

# 민원인 세션
citizen_session_lock = threading.Lock()
citizen_session: dict[str, Any] = {
    "waiting": False,
    "citizenData": None,
    "updatedAt": None,
    "_db_session_id": None,
    "_db_consultation_id": None,
}

# 채팅 메시지
chat_messages_lock = threading.Lock()
chat_messages: list[dict[str, Any]] = []

# 세션 종료 상태
session_state_lock = threading.Lock()
session_state: dict[str, Any] = {
    "ended": False,
    "updatedAt": None,
}

def get_citizen_session() -> dict[str, Any]:
    with citizen_session_lock:
        return dict(citizen_session)

def set_citizen_session(citizen_data: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        "name": str(citizen_data.get("name", "")).strip(),
        "dob": str(citizen_data.get("dob", "")).strip(),
        "gender": str(citizen_data.get("gender", "")).strip(),
        "phone": str(citizen_data.get("phone", "")).strip(),
    }
    db_id = _db.save_citizen_session(normalized)
    consult_id = _db.save_consultation(
        citizen_session_id=db_id,
        scenario=str(citizen_data.get("scenario") or "") or None,
    )
    with citizen_session_lock:
        citizen_session.update({
            "waiting": True,
            "citizenData": normalized,
            "updatedAt": int(time.time()),
            "_db_session_id": db_id,
            "_db_consultation_id": consult_id,
        })
        return dict(citizen_session)

def clear_citizen_session() -> dict[str, Any]:
    with citizen_session_lock:
        db_id = citizen_session.get("_db_session_id")
        consult_id = citizen_session.get("_db_consultation_id")
        citizen_session.update({"waiting": False, "citizenData": None, "updatedAt": None,
                                "_db_session_id": None, "_db_consultation_id": None})
        result = dict(citizen_session)
    if db_id:
        _db.end_citizen_session(db_id)
    if consult_id:
        _db.end_consultation(consult_id)
    return result

def get_messages() -> list[dict[str, Any]]:
    with chat_messages_lock:
        return list(chat_messages)

def add_message(message: dict[str, Any]) -> dict[str, Any] | None:
    # 현재 상담 id를 chat 락 밖에서 먼저 조회 (락 중첩 방지)
    with citizen_session_lock:
        consult_id = citizen_session.get("_db_consultation_id")
    with chat_messages_lock:
        if any(item.get("id") == message["id"] for item in chat_messages):
            return None
        chat_messages.append(message)
        del chat_messages[:-200]
    # DB에 영구 저장 (본문 암호화). 실패해도 실시간 채팅 흐름엔 영향 없음
    try:
        _db.save_message(consult_id, str(message.get("sender") or ""),
                         str(message.get("text") or ""), str(message.get("id") or ""))
    except Exception:
        pass
    return message

def clear_messages() -> None:
    with chat_messages_lock:
        chat_messages.clear()

def get_session_state() -> dict[str, Any]:
    with session_state_lock:
        return dict(session_state)

def set_session_ended(ended: bool) -> dict[str, Any]:
    with session_state_lock:
        session_state.update({"ended": ended, "updatedAt": int(time.time())})
        return dict(session_state)

def clear_session_state() -> dict[str, Any]:
    with session_state_lock:
        session_state.update({"ended": False, "updatedAt": None})
        return dict(session_state)
