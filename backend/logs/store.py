# -*- coding: utf-8 -*-
"""
실시간 로그 저장소 (in-memory deque + SSE pub/sub)

백엔드 after_request 훅과 프론트엔드 에러 리포터 양쪽에서
push_log()를 호출하면 구독 중인 모든 SSE 클라이언트에게 즉시 전달됩니다.
서버 재시작 전까지 최대 MAX_LOGS 건을 메모리에 누적합니다.
"""
from __future__ import annotations

import queue
import threading
import time
from collections import deque

MAX_LOGS = 1000

_log_store: deque[dict] = deque(maxlen=MAX_LOGS)
_log_lock = threading.Lock()

_subscribers: list[queue.Queue] = []
_subscribers_lock = threading.Lock()

_log_counter = 0
_counter_lock = threading.Lock()


def _next_id() -> int:
    global _log_counter
    with _counter_lock:
        _log_counter += 1
        return _log_counter


def push_log(
    level: str,
    source: str,
    message: str,
    status: int | None = None,
    method: str | None = None,
    path: str | None = None,
) -> None:
    """로그 항목을 저장소에 추가하고 모든 SSE 구독자에게 전송."""
    entry: dict = {
        "id": _next_id(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "level": level,
        "source": source,
        "message": message,
    }
    if status is not None:
        entry["status"] = status
    if method is not None:
        entry["method"] = method
    if path is not None:
        entry["path"] = path

    with _log_lock:
        _log_store.append(entry)

    # SSE 구독자에게 브로드캐스트
    with _subscribers_lock:
        dead: list[queue.Queue] = []
        for q in _subscribers:
            try:
                q.put_nowait(entry)
            except queue.Full:
                dead.append(q)
        for q in dead:
            try:
                _subscribers.remove(q)
            except ValueError:
                pass


def get_history() -> list[dict]:
    """현재까지 누적된 모든 로그 반환."""
    with _log_lock:
        return list(_log_store)


def subscribe() -> queue.Queue:
    """새 SSE 클라이언트 구독 등록 후 Queue 반환."""
    q: queue.Queue = queue.Queue(maxsize=300)
    with _subscribers_lock:
        _subscribers.append(q)
    return q


def unsubscribe(q: queue.Queue) -> None:
    """SSE 클라이언트 구독 해제."""
    with _subscribers_lock:
        try:
            _subscribers.remove(q)
        except ValueError:
            pass
