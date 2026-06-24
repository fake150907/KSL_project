from __future__ import annotations

import hashlib
import os


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"필수 환경변수 {name!r}가 설정되지 않았습니다. .env 파일을 확인하세요."
        )
    return value


class Config:
    FLASK_SECRET_KEY: str    = _require_env("FLASK_SECRET_KEY")
    ADMIN_USERNAME: str      = os.environ.get("ADMIN_USERNAME", "admin")
    ADMIN_PASSWORD_HASH: str = _require_env("ADMIN_PASSWORD_HASH")
    SESSION_TIMEOUT: int   = int(os.environ.get("SESSION_TIMEOUT", "43200"))

    KAKAO_REST_API_KEY: str   = os.environ.get("KAKAO_REST_API_KEY", "")
    KAKAO_CLIENT_SECRET: str  = os.environ.get("KAKAO_CLIENT_SECRET", "")
    KAKAO_ACCESS_TOKEN: str   = os.environ.get("KAKAO_ACCESS_TOKEN", "")
    KAKAO_REFRESH_TOKEN: str  = os.environ.get("KAKAO_REFRESH_TOKEN", "")

    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTROPIC_API_KEY", "")
    ANTHROPIC_MODEL: str   = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    DATABASE_URL: str = os.environ.get("DATABASE_URL", "")
