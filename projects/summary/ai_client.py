from __future__ import annotations

import anthropic

from config import Config


def summarize(conversation: list[str]) -> str:
    if not Config.ANTHROPIC_API_KEY:
        raise RuntimeError(
            "ANTHROPIC_API_KEY가 설정되지 않았습니다. "
            "환경변수를 확인해주세요."
        )

    prompt = _build_prompt(conversation)

    try:
        client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text.strip()
    except anthropic.AuthenticationError:
        raise RuntimeError("ANTHROPIC_API_KEY가 유효하지 않습니다.")
    except anthropic.RateLimitError:
        raise RuntimeError("Anthropic API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
    except anthropic.APIConnectionError:
        raise RuntimeError("Anthropic API 서버에 연결할 수 없습니다. 네트워크를 확인해주세요.")
    except Exception as exc:
        raise RuntimeError(f"Claude 요약 실패: {exc}") from exc


def _build_prompt(conversation: list[str]) -> str:
    joined = " / ".join(conversation)
    return (
        f"다음은 수어 인식 시스템이 인식한 단어 및 문장 목록입니다:\n\n{joined}\n\n"
        "위 내용을 자연스러운 한국어 문장으로 요약해주세요. "
        "핵심 내용을 간결하게 정리하되, 원래 의미가 왜곡되지 않도록 해주세요."
    )
