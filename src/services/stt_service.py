"""Role: speech-to-text adapter.

Input: microphone audio
Output: recognized text
Example:
  python -m src.services.stt_service --backend local
"""

from __future__ import annotations

import argparse


def listen_once(backend: str = "local", language_code: str = "ko-KR", timeout: int = 5) -> str:
    if backend == "google":
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            audio = recognizer.listen(source, timeout=timeout)
        return recognizer.recognize_google_cloud(audio, language=language_code)

    try:
        import speech_recognition as sr

        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source, timeout=timeout)
        return recognizer.recognize_google(audio, language=language_code)
    except Exception as exc:
        return f"[STT fallback: {exc}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="local", choices=["local", "google"])
    parser.add_argument("--language_code", default="ko-KR")
    args = parser.parse_args()
    print(listen_once(args.backend, args.language_code))


if __name__ == "__main__":
    main()
