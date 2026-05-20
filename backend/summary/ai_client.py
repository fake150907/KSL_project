from __future__ import annotations
from config import Config

def summarize(conversation: list[str] | str) -> str:
    if not Config.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY가 설정되어 있지 않습니다.")

    if isinstance(conversation, str):
        conversation = [line for line in conversation.splitlines() if line.strip()]

    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic 패키지가 설치되어 있지 않습니다.") from exc

    prompt = _build_prompt(conversation)
    
    print("\n[DEBUG - ai_client.py] Claude API 통신 준비")
    print(f"[DEBUG - ai_client.py] 사용할 모델: {Config.ANTHROPIC_MODEL}")
    print(f"[DEBUG - ai_client.py] 프롬프트 길이: {len(prompt)}")

    try:
        client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        print("[DEBUG - ai_client.py] Anthropic API에 요청 전송 중...")
        
        message = client.messages.create(
            model=Config.ANTHROPIC_MODEL,
            max_tokens=900,
            messages=[{"role": "user", "content": prompt}],
        )
        
        print("[DEBUG - ai_client.py] Anthropic API 응답 수신 완료")
        print(f"[DEBUG - ai_client.py] 원본 응답 객체: {message}")
        
        content = message.content[0]
        text = getattr(content, "text", "")
        
        if not text.strip():
            print("[DEBUG - ai_client.py] ❌ 경고: 반환된 텍스트가 비어있습니다.")
            
        return text.strip()
        
    except Exception as exc:
        print(f"[DEBUG - ai_client.py] ❌ Claude 통신/처리 중 예외 발생: {exc}")
        raise RuntimeError(f"Claude 요약 실패: {exc}") from exc

def _build_prompt(conversation: list[str]) -> str:
    joined = "\n".join(str(item).strip() for item in conversation if str(item).strip())
    return (
        "당신은 민원 수어 상담 보조 기록 작성자입니다.\n"
        "아래 자료에는 민원인과 상담원의 대화, 수어 인식 결과, 상담원이 작성한 메모와 처리 내용이 섞여 있습니다.\n"
        "시간순 대화 목록을 다시 나열하지 말고, 민원인이 나중에 카카오톡으로 받아볼 수 있는 자연스러운 상담 요약문으로 정리하세요.\n"
        "의학적 사실을 새로 만들어내지 말고, 자료에 없는 내용은 '기록 없음'이라고 적으세요.\n\n"
        "[입력 자료]\n"
        f"{joined}\n\n"
        "[출력 형식]\n"
        "수어 민원 상담 요약\n"
        "- 주요 증상: \n"
        "- 증상 위치/기간/정도: \n"
        "- 상담원 확인 또는 처리: \n"
        "- 처방 및 안내: \n"
        "- 추가 확인 필요 사항: \n\n"
        "마지막에 민원인가 이해하기 쉬운 한두 문장 요약을 덧붙이세요."
    )