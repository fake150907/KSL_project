from __future__ import annotations

import json

import requests

from config import Config

KAKAO_MEMO_URL = "https://kapi.kakao.com/v2/api/talk/memo/default/send"


def send_kakao_message(access_token: str, summary: str) -> None:
    if not Config.KAKAO_REST_API_KEY:
        raise ValueError(
            "KAKAO_REST_API_KEY가 설정되어 있지 않습니다. 환경변수를 확인해주세요."
        )
    if not access_token:
        raise ValueError("카카오 액세스 토큰이 없습니다.")
    if not summary:
        raise ValueError("전송할 요약 내용이 없습니다.")

    template = {
        "object_type": "text",
        "text": f"[수어 진료 요약본]\n\n{summary}",
        "link": {
            "web_url": "",
            "mobile_web_url": "",
        },
    }

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"template_object": json.dumps(template, ensure_ascii=False)}

    try:
        response = requests.post(KAKAO_MEMO_URL, headers=headers, data=data, timeout=10)
        try:
            result = response.json()
        except ValueError:
            result = response.text

        if response.status_code == 401:
            raise RuntimeError(
                "카카오 액세스 토큰이 만료되었거나 유효하지 않습니다. 다시 로그인해주세요."
            )
        if response.status_code == 403:
            raise RuntimeError(
                "나에게 전송 권한이 없습니다. 카카오 개발자 콘솔 동의항목의 talk_message 권한을 확인해주세요."
            )
        if response.status_code != 200:
            raise RuntimeError(f"카카오 API 오류 ({response.status_code}): {result}")

        print("[KAKAO] 나에게 전송 완료", flush=True)

    except requests.exceptions.Timeout as exc:
        raise RuntimeError("카카오 API 요청 시간이 초과되었습니다.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError("카카오 API 서버에 연결할 수 없습니다. 네트워크를 확인해주세요.") from exc
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"카카오 전송 실패: {exc}") from exc
