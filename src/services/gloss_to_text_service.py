"""Convert Korean Sign Language gloss sequences into natural Korean text."""

from __future__ import annotations

import argparse
import os


DEFAULT_PROVIDER = os.environ.get("GLOSS_TO_TEXT_PROVIDER", "anthropic").strip().lower()
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_GLOSS_MODEL", "gpt-4.1-mini")
DEFAULT_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_GLOSS_MODEL", "claude-haiku-4-5-20251001")

SYSTEM_PROMPT = """당신은 한국 수어(KSL) 글로스를 자연스러운 한국어 문장으로 바꾸는 의료 통역 보조자입니다.

입력은 수어 단어를 순서대로 나열한 글로스입니다. 예: "저 + 아프다 + 의사 + 원하다"

규칙:
- 병원 방문 상황의 환자 발화로 해석합니다.
- 수어에서 생략되기 쉬운 조사, 어미, 높임말을 자연스럽게 복원합니다.
- 단어 순서에 억지로 묶이지 말고 한국어로 자연스럽게 재배열합니다.
- 의미를 추가로 꾸며내지 말고, 입력 글로스에서 추론 가능한 내용만 반영합니다.
- 출력은 설명 없이 자연스러운 한국어 문장 하나만 반환합니다.

예시:
- 저 + 아프다 + 의사 + 원하다 -> 저 아파서 의사 선생님을 만나고 싶어요.
- 저 + 머리 + 아프다 -> 저는 머리가 아파요.
- 검사 + 받다 + 원하다 -> 검사를 받고 싶어요.
- 의사 + 감사 -> 의사 선생님 감사합니다.
"""


def _normalize_gloss(gloss: str) -> str:
    return " + ".join(part.strip() for part in gloss.replace(",", "+").split("+") if part.strip())


def gloss_to_text(
    gloss: str,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Convert a KSL gloss string to a natural Korean sentence.

    Provider is selected by `provider` or `GLOSS_TO_TEXT_PROVIDER`.
    Claude/Anthropic is the default because this project is set up for ANTHROPIC_API_KEY.
    Supported providers: `openai`, `anthropic`/`claude`.
    """
    normalized = _normalize_gloss(gloss)
    if not normalized:
        return ""

    selected_provider = (provider or DEFAULT_PROVIDER or "anthropic").strip().lower()
    if selected_provider in {"anthropic", "claude"}:
        return _gloss_to_text_anthropic(normalized, model or DEFAULT_ANTHROPIC_MODEL, api_key)
    return _gloss_to_text_openai(normalized, model or DEFAULT_OPENAI_MODEL, api_key)


def _gloss_to_text_openai(gloss: str, model: str, api_key: str | None = None) -> str:
    key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not key:
        return f"[API key missing: set OPENAI_API_KEY] 원문: {gloss}"

    try:
        from openai import OpenAI

        client = OpenAI(api_key=key)
        response = client.responses.create(
            model=model,
            instructions=SYSTEM_PROMPT,
            input=gloss,
            max_output_tokens=256,
        )
        return response.output_text.strip()
    except ImportError:
        return f"[openai package missing: pip install openai] 원문: {gloss}"
    except Exception as exc:
        return f"[변환 실패: {exc}] 원문: {gloss}"


def _gloss_to_text_anthropic(gloss: str, model: str, api_key: str | None = None) -> str:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return f"[API key missing: set ANTHROPIC_API_KEY] 원문: {gloss}"

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=key)
        message = client.messages.create(
            model=model,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": gloss}],
        )
        return message.content[0].text.strip()
    except ImportError:
        return f"[anthropic package missing: pip install anthropic] 원문: {gloss}"
    except Exception as exc:
        return f"[변환 실패: {exc}] 원문: {gloss}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gloss", help="Space-separated or plus-separated KSL gloss string")
    parser.add_argument("--provider", choices=["openai", "anthropic", "claude"], default=DEFAULT_PROVIDER)
    parser.add_argument("--model")
    parser.add_argument("--api-key")
    args = parser.parse_args()
    print(gloss_to_text(args.gloss, provider=args.provider, model=args.model, api_key=args.api_key))


if __name__ == "__main__":
    main()
