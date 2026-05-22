"""Convert Korean Sign Language gloss sequences into natural Korean text."""

from __future__ import annotations

import argparse
import os


DEFAULT_PROVIDER = os.environ.get("GLOSS_TO_TEXT_PROVIDER", "anthropic").strip().lower()
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_GLOSS_MODEL", "gpt-4.1-mini")
DEFAULT_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_GLOSS_MODEL", "claude-haiku-4-5-20251001")

SYSTEM_PROMPT = """You are a medical Korean Sign Language gloss translation assistant.

Convert the input KSL gloss sequence into one natural Korean sentence.
Keep the meaning grounded in the input glosses and do not add extra symptoms.

Examples:
- 오른쪽 + 위 + 통증 + 못견디다 -> 오른쪽 위가 아파서 못 견디겠어요.
- 소화불량 + 어떻게 + 치료 -> 소화불량은 어떻게 치료하나요?
- 골절 + 회복 + 얼마 -> 골절은 회복하는 데 얼마나 걸리나요?
"""


def _normalize_gloss(gloss: str) -> str:
    return " + ".join(part.strip() for part in gloss.replace(",", "+").split("+") if part.strip())


def _local_gloss_to_text(gloss: str) -> str:
    parts = [part.strip() for part in gloss.split("+") if part.strip()]
    part_set = set(parts)

    if {"오른쪽", "위", "통증", "못견디다"}.issubset(part_set):
        return "오른쪽 위가 아파서 못 견디겠어요."
    if {"소화불량", "어떻게", "치료"}.issubset(part_set):
        return "소화불량은 어떻게 치료하나요?"
    if {"골절", "회복", "얼마"}.issubset(part_set):
        return "골절은 회복하는 데 얼마나 걸리나요?"

    replacements = {
        "오른쪽": "오른쪽",
        "왼쪽": "왼쪽",
        "위": "위",
        "아래": "아래",
        "배": "배",
        "머리": "머리",
        "통증": "아파요",
        "아프다": "아파요",
        "못견디다": "못 견디겠어요",
        "소화불량": "소화가 잘 안 돼요",
        "어떻게": "어떻게",
        "치료": "치료하나요",
        "골절": "골절",
        "회복": "회복",
        "얼마": "얼마나 걸리나요",
    }
    sentence = " ".join(replacements.get(part, part) for part in parts).strip()
    if not sentence:
        return gloss
    return sentence if sentence.endswith(("요", "요?", "다", "다.", "?", ".")) else f"{sentence}."


def gloss_to_text(
    gloss: str,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Convert a KSL gloss string to a natural Korean sentence.

    Provider is selected by `provider` or `GLOSS_TO_TEXT_PROVIDER`.
    When the configured API key or package is missing, a local deterministic
    fallback is used so demos do not expose backend setup errors to patients.
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
        return _local_gloss_to_text(gloss)

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
        return _local_gloss_to_text(gloss)
    except Exception as exc:
        print(f"OpenAI gloss_to_text failed: {exc}")
        return _local_gloss_to_text(gloss)


def _gloss_to_text_anthropic(gloss: str, model: str, api_key: str | None = None) -> str:
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        return _local_gloss_to_text(gloss)

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
        return _local_gloss_to_text(gloss)
    except Exception as exc:
        print(f"Anthropic gloss_to_text failed: {exc}")
        return _local_gloss_to_text(gloss)


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
