from __future__ import annotations

import os


class Config:
    FLASK_SECRET_KEY: str = os.environ.get("FLASK_SECRET_KEY", "sign-interpreter-secret")

    # ── 관리자 계정 ────────────────────────────────────
    ADMIN_USERNAME: str = os.environ.get("ADMIN_USERNAME", "admin")
    # 기본 비밀번호: admin1234  (운영 시 환경변수로 교체)
    # 생성: python3 -c "import hashlib; print(hashlib.sha256('비밀번호'.encode()).hexdigest())"
    ADMIN_PASSWORD_HASH: str = os.environ.get(
        "ADMIN_PASSWORD_HASH",
        "ac3de8f6e13342d0d926cb39d0987fa4b47f8b5b3f4fbbefb1dde0a14cbfb34c",  # admin1234
    )
    SESSION_TIMEOUT: int = int(os.environ.get("SESSION_TIMEOUT", 43200)) # 병원 진료시간 별 세션 시간 변경 가능

    # ── Gmail SMTP ─────────────────────────────────────
    # Gmail 앱 비밀번호 발급: Google 계정 → 보안 → 2단계 인증 → 앱 비밀번호
    GMAIL_ADDRESS: str = os.environ.get("GMAIL_ADDRESS", "")
    GMAIL_APP_PASSWORD: str = os.environ.get("GMAIL_APP_PASSWORD", "")

    # ── AI API key ───────────────────────
    ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")

    # ── 비전 모델 ──────────────────────────────────────
    SIGN_CONFIG: str = os.environ.get("SIGN_CONFIG", "config/web_demo.yaml")
