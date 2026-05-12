from __future__ import annotations
import traceback
from flask import Blueprint, jsonify, request
from summary.ai_client import summarize

summary_bp = Blueprint("summary", __name__)

@summary_bp.route("/api/summary", methods=["POST"])
def create_summary():
    print("\n[DEBUG - routes.py] /api/summary POST 요청 수신됨")
    data = request.get_json(silent=True) or {}
    conversation = data.get("conversation", [])

    print(f"[DEBUG - routes.py] 받은 데이터 타입: {type(data)}")
    print(f"[DEBUG - routes.py] 대화 리스트 길이: {len(conversation)}")
    
    if conversation:
        print(f"[DEBUG - routes.py] 대화 첫 번째 줄 미리보기: {conversation[0]}")

    if not isinstance(conversation, list):
        print("[DEBUG - routes.py] ❌ 에러: conversation이 리스트가 아님")
        return jsonify({"error": "conversation은 문자열 리스트여야 합니다."}), 400
    if not conversation:
        print("[DEBUG - routes.py] ❌ 에러: conversation 내용이 비어있음")
        return jsonify({"error": "요약할 대화 내용이 없습니다."}), 400

    try:
        print("[DEBUG - routes.py] Claude 요약 함수(summarize) 호출 시작...")
        result_text = summarize([str(item) for item in conversation])
        print(f"[DEBUG - routes.py] ✅ 요약 성공. 반환된 텍스트 길이: {len(result_text)}")
        print(f"[DEBUG - routes.py] ✅ 반환 텍스트 미리보기: {result_text[:100]}...")
        return jsonify({"summary": result_text}), 200
    except Exception as exc:
        print(f"[DEBUG - routes.py] ❌ 서버 내부 에러 발생: {exc}")
        traceback.print_exc()   
        return jsonify({"error": str(exc)}), 502