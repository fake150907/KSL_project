# -*- coding: utf-8 -*-
"""
로그 API Blueprint

GET  /api/logs          → 현재까지 누적된 전체 로그 JSON 배열 반환
POST /api/logs/frontend → 프론트엔드에서 에러/경고를 서버로 전송
GET  /api/logs/stream   → Server-Sent Events (SSE) 실시간 스트림
"""
from __future__ import annotations

import json
import queue as queue_module

from flask import Blueprint, Response, jsonify, request, stream_with_context

from .store import get_history, push_log, subscribe, unsubscribe

logs_bp = Blueprint("logs", __name__)


@logs_bp.route("/api/logs", methods=["GET"])
def get_logs():
    """누적 로그 전체를 JSON으로 반환."""
    return jsonify(get_history()), 200


@logs_bp.route("/api/logs/frontend", methods=["POST"])
def receive_frontend_log():
    """프론트엔드 에러/경고 수신 후 저장소에 추가."""
    data = request.get_json(silent=True) or {}
    level = str(data.get("level", "error"))
    message = str(data.get("message", "")).strip()
    source = str(data.get("source", "Frontend")).strip()
    if message:
        push_log(level, source, message)
    return jsonify({"ok": True}), 200


@logs_bp.route("/api/logs/stream")
def stream_logs():
    """SSE 실시간 스트림.

    연결 즉시 기존 누적 로그를 전송한 뒤 새 로그가 들어올 때마다 push.
    25초 마다 keepalive comment를 전송하여 연결을 유지합니다.
    """

    def generate():
        # 구독 등록 (history 전송 전에 먼저 등록해야 gap 없음)
        q = subscribe()
        try:
            # ① 기존 누적 로그 전송
            for entry in get_history():
                yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"

            # ② 이후 실시간 push
            while True:
                try:
                    entry = q.get(timeout=25)
                    yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
                except queue_module.Empty:
                    # ping — 브라우저가 연결 상태를 감지할 수 있도록 data 이벤트로 전송
                    yield 'data: {"type":"ping"}\n\n'
        finally:
            unsubscribe(q)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
