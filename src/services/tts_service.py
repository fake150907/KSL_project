"""Role: text-to-speech adapter.

Input: text string
Output: spoken audio or synthesized MP3
Example:
  python -m src.services.tts_service "hello"
"""

from __future__ import annotations

import argparse
from pathlib import Path


def speak_text(text: str, backend: str = "local", output_path: str | None = None) -> str:
    if not text:
        return ""
    if backend == "google":
        from google.cloud import texttospeech

        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code="ko-KR", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        out = Path(output_path or "outputs/tts_output.mp3")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(response.audio_content)
        return str(out)

    try:
        import pyttsx3

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        return "spoken_local"
    except Exception:
        print(f"[TTS fallback] {text}")
        return "printed_fallback"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--backend", default="local", choices=["local", "google"])
    parser.add_argument("--output_path")
    args = parser.parse_args()
    print(speak_text(args.text, args.backend, args.output_path))


if __name__ == "__main__":
    main()
