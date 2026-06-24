from __future__ import annotations

import threading
import time
from typing import Any

import db as _db

# 지점(branch)별 인메모리 상태.
# 한 백엔드가 여러 동사무소를 처리하므로 branch_id로 상태를 분리한다.
# (branch_id=None 이면 슈퍼관리자/단일 운영용 "_default" 키 사용 — 하위호환)
_lock = threading.Lock()
_branches: dict[str, dict[str, Any]] = {}


def _bkey(branch_id: str | None) -> str:
    return branch_id or "_default"


def _state(branch_id: str | None) -> dict[str, Any]:
    """해당 지점 상태 dict 반환 (없으면 생성). 호출 측이 _lock 보유한 상태에서 사용."""
    key = _bkey(branch_id)
    st = _branches.get(key)
    if st is None:
        st = {
            "citizen": {
                "waiting": False, "citizenData": None, "updatedAt": None,
                "_db_session_id": None, "_db_consultation_id": None,
            },
            "messages": [],
            "session": {"ended": False, "updatedAt": None},
        }
        _branches[key] = st
    return st


def get_citizen_session(branch_id: str | None = None) -> dict[str, Any]:
    with _lock:
        return dict(_state(branch_id)["citizen"])


def set_citizen_session(citizen_data: dict[str, Any], branch_id: str | None = None) -> dict[str, Any]:
    normalized = {
        "name": str(citizen_data.get("name", "")).strip(),
        "dob": str(citizen_data.get("dob", "")).strip(),
        "gender": str(citizen_data.get("gender", "")).strip(),
        "phone": str(citizen_data.get("phone", "")).strip(),
    }
    db_id = _db.save_citizen_session(normalized, branch_id=branch_id)
    consult_id = _db.save_consultation(
        citizen_session_id=db_id,
        scenario=str(citizen_data.get("scenario") or "") or None,
        branch_id=branch_id,
    )
    with _lock:
        cit = _state(branch_id)["citizen"]
        cit.update({
            "waiting": True,
            "citizenData": normalized,
            "updatedAt": int(time.time()),
            "_db_session_id": db_id,
            "_db_consultation_id": consult_id,
        })
        return dict(cit)


def clear_citizen_session(branch_id: str | None = None) -> dict[str, Any]:
    with _lock:
        cit = _state(branch_id)["citizen"]
        db_id = cit.get("_db_session_id")
        consult_id = cit.get("_db_consultation_id")
        cit.update({"waiting": False, "citizenData": None, "updatedAt": None,
                    "_db_session_id": None, "_db_consultation_id": None})
        result = dict(cit)
    if db_id:
        _db.end_citizen_session(db_id)
    if consult_id:
        _db.end_consultation(consult_id)
    return result


def get_messages(branch_id: str | None = None) -> list[dict[str, Any]]:
    with _lock:
        return list(_state(branch_id)["messages"])


def add_message(message: dict[str, Any], branch_id: str | None = None) -> dict[str, Any] | None:
    with _lock:
        st = _state(branch_id)
        msgs = st["messages"]
        if any(item.get("id") == message["id"] for item in msgs):
            return None
        msgs.append(message)
        del msgs[:-200]
        consult_id = st["citizen"].get("_db_consultation_id")
    # DB에 영구 저장 (본문 암호화). 실패해도 실시간 채팅 흐름엔 영향 없음
    try:
        _db.save_message(consult_id, str(message.get("sender") or ""),
                         str(message.get("text") or ""), str(message.get("id") or ""))
    except Exception:
        pass
    return message


def clear_messages(branch_id: str | None = None) -> None:
    with _lock:
        _state(branch_id)["messages"].clear()


def get_session_state(branch_id: str | None = None) -> dict[str, Any]:
    with _lock:
        return dict(_state(branch_id)["session"])


def set_session_ended(ended: bool, branch_id: str | None = None) -> dict[str, Any]:
    with _lock:
        sess = _state(branch_id)["session"]
        sess.update({"ended": ended, "updatedAt": int(time.time())})
        return dict(sess)


def clear_session_state(branch_id: str | None = None) -> dict[str, Any]:
    with _lock:
        sess = _state(branch_id)["session"]
        sess.update({"ended": False, "updatedAt": None})
        return dict(sess)
