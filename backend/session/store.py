from __future__ import annotations

import threading
import time
from typing import Any

# 민원인 세션
citizen_session_lock = threading.Lock()
citizen_session: dict[str, Any] = {
    "waiting": False,
    "citizenData": None,
    "updatedAt": None,
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
    with citizen_session_lock:
        citizen_session.update({
            "waiting": True,
            "citizenData": normalized,
            "updatedAt": int(time.time()),
        })
        return dict(citizen_session)

def clear_citizen_session() -> dict[str, Any]:
    with citizen_session_lock:
        citizen_session.update({"waiting": False, "citizenData": None, "updatedAt": None})
        return dict(citizen_session)

def get_messages() -> list[dict[str, Any]]:
    with chat_messages_lock:
        return list(chat_messages)

def add_message(message: dict[str, Any]) -> dict[str, Any] | None:
    with chat_messages_lock:
        if any(item.get("id") == message["id"] for item in chat_messages):
            return None
        chat_messages.append(message)
        del chat_messages[:-200] 
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
