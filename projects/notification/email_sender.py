from __future__ import annotations

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import Config


def send_email(to_address: str, summary: str) -> None:
    if not Config.GMAIL_ADDRESS or not Config.GMAIL_APP_PASSWORD:
        raise ValueError(
            "Gmail 계정 정보가 설정되지 않았습니다. "
            "환경변수 GMAIL_ADDRESS와 GMAIL_APP_PASSWORD를 설정해주세요."
        )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "수어 대화 요약본"
    msg["From"]    = Config.GMAIL_ADDRESS
    msg["To"]      = to_address

    # 일반 텍스트 본문
    text_body = f"수어 대화 요약본\n\n{summary}"

    # HTML 본문 (보기 좋게)
    html_body = f"""
    <html>
      <body style="font-family: sans-serif; padding: 24px; color: #222;">
        <h2 style="color: #333;">수어 대화 요약본</h2>
        <hr style="border: none; border-top: 1px solid #ddd;">
        <p style="line-height: 1.8; font-size: 15px;">{summary.replace(chr(10), '<br>')}</p>
        <hr style="border: none; border-top: 1px solid #ddd;">
        <p style="font-size: 12px; color: #999;">본 메일은 수어 인식 시스템에서 자동 발송되었습니다.</p>
      </body>
    </html>
    """

    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(Config.GMAIL_ADDRESS, Config.GMAIL_APP_PASSWORD)
            smtp.sendmail(Config.GMAIL_ADDRESS, to_address, msg.as_string())
        print(f"[EMAIL] 전송 완료 → {to_address}", flush=True)
    except smtplib.SMTPAuthenticationError:
        raise RuntimeError(
            "Gmail 인증 실패. 앱 비밀번호를 확인해주세요.\n"
            "발급: Google 계정 → 보안 → 2단계 인증 → 앱 비밀번호"
        )
    except Exception as exc:
        raise RuntimeError(f"메일 전송 실패: {exc}") from exc
