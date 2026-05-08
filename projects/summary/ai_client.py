from __future__ import annotations

from config import Config


def summarize(conversation: list[str]) -> str:
    provider = Config.AI_PROVIDER.upper()

    if provider == "ANTHROPIC":
        return _summarize_anthropic(conversation)

def _build_prompt(conversation: list[str]) -> str:
    joined = " / ".join(conversation)
    return (
        f"다음은 수어 인식 시스템이 인식한 단어 및 문장 목록입니다:\n\n{joined}\n\n"
        "위 내용을 자연스러운 한국어 문장으로 요약해주세요. "
        "핵심 내용을 간결하게 정리하되, 원래 의미가 왜곡되지 않도록 해주세요."
    )

def _summarize_anthropic(conversation: list[str]) -> str:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            messages=[{"role": "user", "content": _build_prompt(conversation)}],
        )
        return message.content[0].text.strip()
    except Exception as exc:
        raise RuntimeError(f"Anthropic 요약 실패: {exc}") from exc