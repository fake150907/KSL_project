from __future__ import annotations

from config import Config


def summarize(conversation: list[str]) -> str:
    if not Config.ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY 또는 ANTROPIC_API_KEY 환경변수가 설정되어 있지 않습니다.")

    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("anthropic 패키지가 설치되어 있지 않습니다. requirements.txt를 설치해 주세요.") from exc

    prompt = _build_prompt(conversation)
    try:
        client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=Config.ANTHROPIC_MODEL,
            max_tokens=512,
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
        "다음은 수어 인식 진료 보조 시스템에서 수집한 환자와 의사의 대화입니다.\n\n"
        f"{joined}\n\n"
        "진료 인계용으로 핵심 증상, 환자 요청, 의사 안내를 한국어로 간결하게 요약해 주세요."
    )
