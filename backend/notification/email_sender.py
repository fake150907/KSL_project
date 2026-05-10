from __future__ import annotations

import html
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from config import Config


def send_email(to_address: str, summary: str) -> None:
    if not Config.GMAIL_ADDRESS or not Config.GMAIL_APP_PASSWORD:
        raise ValueError("GMAIL_ADDRESS와 GMAIL_APP_PASSWORD 환경변수가 필요합니다.")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "수어 진료 대화 요약본"
    msg["From"] = Config.GMAIL_ADDRESS
    msg["To"] = to_address

    text_body = f"수어 진료 대화 요약본\n\n{summary}\n\n본 메일은 수어 인식 시스템에서 자동 발송되었습니다."
    html_body = (
        "<html><body style=\"font-family: sans-serif; color: #222; padding: 24px;\">"
        "<h2>수어 진료 대화 요약본</h2>"
        "<hr>"
        f"<p style=\"line-height: 1.7; white-space: pre-wrap;\">{html.escape(summary)}</p>"
        "<hr>"
        "<p style=\"font-size: 12px; color: #777;\">본 메일은 수어 인식 시스템에서 자동 발송되었습니다.</p>"
        "</body></html>"
    )

    msg.attach(MIMEText(text_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html", "utf-8"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(Config.GMAIL_ADDRESS, Config.GMAIL_APP_PASSWORD)
            smtp.sendmail(Config.GMAIL_ADDRESS, to_address, msg.as_string())
    except smtplib.SMTPAuthenticationError as exc:
        raise RuntimeError("Gmail 인증 실패. 앱 비밀번호를 확인해 주세요.") from exc
    except Exception as exc:
        raise RuntimeError(f"메일 전송 실패: {exc}") from exc

