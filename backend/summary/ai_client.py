from __future__ import annotations

from config import Config


def summarize(conversation: list[str] | str) -> str:
    if not Config.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY 또는 ANTROPIC_API_KEY 환경변수가 설정되어 있지 않습니다.")

    # conversation이 string으로 잘못 넘어와도 list로 정규화
    if isinstance(conversation, str):
        conversation = [line for line in conversation.splitlines() if line.strip()]

    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic 패키지가 설치되어 있지 않습니다. requirements.txt를 설치해주세요.") from exc

    prompt = _build_prompt(conversation)
    try:
        client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=Config.ANTHROPIC_MODEL,
            max_tokens=900,
            messages=[{"role": "user", "content": prompt}],
        )
        content = message.content[0]
        text = getattr(content, "text", "")
        return text.strip()
    except anthropic.AuthenticationError as exc:
        raise RuntimeError("Anthropic API 키가 유효하지 않습니다.") from exc
    except anthropic.RateLimitError as exc:
        raise RuntimeError("Anthropic API 요청 한도를 초과했습니다.") from exc
    except anthropic.APIConnectionError as exc:
        raise RuntimeError("Anthropic API 서버에 연결할 수 없습니다.") from exc
    except Exception as exc:
        raise RuntimeError(f"Claude 요약 실패: {exc}") from exc


def _build_prompt(conversation: list[str]) -> str:
    joined = "\n".join(str(item).strip() for item in conversation if str(item).strip())
    return (
        "당신은 병원 수어 진료 보조 기록 작성자입니다.\n"
        "아래 자료에는 환자와 의사의 대화, 수어 인식 결과, 의사가 작성한 메모와 처방이 섞여 있습니다.\n"
        "시간순 대화 목록을 다시 나열하지 말고, 환자가 나중에 카카오톡으로 받아볼 수 있는 자연스러운 진료 요약문으로 정리하세요.\n"
        "의학적 사실을 새로 만들어내지 말고, 자료에 없는 내용은 '기록 없음'이라고 적으세요.\n\n"
        "[입력 자료]\n"
        f"{joined}\n\n"
        "[출력 형식]\n"
        "수어 진료 요약\n"
        "- 주요 증상: \n"
        "- 증상 위치/기간/정도: \n"
        "- 의사 판단 또는 진단: \n"
        "- 처방 및 안내: \n"
        "- 추가 확인 필요 사항: \n\n"
        "마지막에 환자가 이해하기 쉬운 한두 문장 요약을 덧붙이세요."
    )