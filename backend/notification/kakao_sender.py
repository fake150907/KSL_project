from __future__ import annotations

import json

import requests

from config import Config

KAKAO_TOKEN_URL = "https://kauth.kakao.com/oauth/token"
KAKAO_MEMO_URL = "https://kapi.kakao.com/v2/api/talk/memo/default/send"


class KakaoAccessTokenError(RuntimeError):
    pass


def refresh_kakao_access_token(refresh_token: str) -> dict[str, str]:
    if not Config.KAKAO_REST_API_KEY:
        raise ValueError("KAKAO_REST_API_KEY 환경변수가 필요합니다.")
    if not refresh_token:
        raise ValueError("카카오 refresh_token이 필요합니다.")

    payload = {
        "grant_type": "refresh_token",
        "client_id": Config.KAKAO_REST_API_KEY,
        "refresh_token": refresh_token,
    }
    if Config.KAKAO_CLIENT_SECRET:
        payload["client_secret"] = Config.KAKAO_CLIENT_SECRET

    response = requests.post(
        KAKAO_TOKEN_URL,
        data=payload,
        timeout=10,
    )
    try:
        result = response.json()
    except ValueError:
        result = {"error": response.text}

    if response.status_code != 200:
        raise KakaoAccessTokenError(f"카카오 access token 재발급 실패 ({response.status_code}): {result}")

    access_token = str(result.get("access_token", "")).strip()
    if not access_token:
        raise KakaoAccessTokenError("카카오 access token 재발급 응답에 access_token이 없습니다.")

    refreshed = {"access_token": access_token}
    new_refresh_token = str(result.get("refresh_token", "")).strip()
    if new_refresh_token:
        refreshed["refresh_token"] = new_refresh_token
    return refreshed


def send_kakao_message(access_token: str, summary: str) -> None:
    if not Config.KAKAO_REST_API_KEY:
        raise ValueError("KAKAO_REST_API_KEY 환경변수가 필요합니다.")
    if not access_token:
        raise ValueError("카카오 access_token이 필요합니다.")
    if not summary:
        raise ValueError("전송할 진료 요약 내용이 없습니다.")

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
            raise KakaoAccessTokenError("카카오 access token이 만료되었거나 유효하지 않습니다.")
        if response.status_code == 403:
            raise RuntimeError("카카오 나에게 보내기 권한이 없습니다. Kakao Developers 동의항목에서 talk_message 권한을 확인해주세요.")
        if response.status_code != 200:
            raise RuntimeError(f"카카오 API 오류 ({response.status_code}): {result}")

        print("[KAKAO] 나에게 보내기 완료", flush=True)

    except requests.exceptions.Timeout as exc:
        raise RuntimeError("카카오 API 요청 시간이 초과되었습니다.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError("카카오 API 서버에 연결할 수 없습니다. 네트워크를 확인해주세요.") from exc
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"카카오 전송 실패: {exc}") from exc
