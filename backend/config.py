from __future__ import annotations
import hashlib
import os


def _default_password_hash() -> str:
    # 기본 비밀번호 해싱 (보안을 위해 실제 운영 시에는 .env에 해시값을 직접 넣는 것이 좋습니다)
    return hashlib.sha256("admin1234".encode("utf-8")).hexdigest()

class Config:
    # os.environ.get(KEY, DEFAULT) -> KEY가 환경변수에 없으면 DEFAULT를 사용함
    FLASK_SECRET_KEY: str = os.environ.get("FLASK_SECRET_KEY", "sign-interpreter-secret")
    ADMIN_USERNAME: str = os.environ.get("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD_HASH: str = os.environ.get("ADMIN_PASSWORD_HASH", _default_password_hash())
    SESSION_TIMEOUT: int = int(os.environ.get("SESSION_TIMEOUT", "43200"))

    KAKAO_REST_API_KEY: str = os.environ.get("KAKAO_REST_API_KEY", "")
    KAKAO_CLIENT_SECRET: str = os.environ.get("KAKAO_CLIENT_SECRET", "")
    KAKAO_ACCESS_TOKEN: str = os.environ.get("KAKAO_ACCESS_TOKEN", "")
    KAKAO_REFRESH_TOKEN: str = os.environ.get("KAKAO_REFRESH_TOKEN", "")

    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str = os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")