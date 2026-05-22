"""Welfare-panel data for hearing-impaired users.

When the sign-language pipeline recognises a "복지카드 분실" scenario at the
kiosk, the staff begins the card-reissue process. While the patient waits, the
UI surfaces a small slide panel of public welfare services that hearing-impaired
users can actually use today (107 손말이음센터 등). This module owns the data:
fetched once from the data.go.kr 복지서비스 API and cached in memory; if the API
key is missing or the call fails, a hard-coded fallback keeps the UX intact.
"""

from __future__ import annotations

import html
import os
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from typing import Any

import requests


WELFARE_PANEL_TRIGGER_TOKEN: str = "WORD0579"  # 복지카드
# 트리거 정책: lookup_key의 '+' 분리 토큰 중 위 토큰이 들어 있으면 패널 노출.
# 즉 WORD0579 단독뿐 아니라 SEN0322+WORD0579, SEN1817+WORD0579,
# SEN0278+WORD0579 등 "복지카드"가 들어간 모든 조합이 자동 매칭됨.


@dataclass(frozen=True)
class _PanelSeed:
    serv_id: str
    title: str
    summary: str
    agency: str
    phone: str
    website: str


_PANEL_SEEDS: tuple[_PanelSeed, ...] = (
    _PanelSeed(
        serv_id="WLF00003219",
        title="통신중계서비스 (107 손말이음센터)",
        summary="수어 통역사가 전화 통화를 실시간 중계해 드립니다.",
        agency="한국지능정보사회진흥원 손말이음센터",
        phone="107",
        website="http://107.kr",
    ),
    _PanelSeed(
        serv_id="WLF00000104",
        title="시각·청각장애인용 TV 보급",
        summary="자막·수어 방송에 최적화된 맞춤형 TV를 보급합니다.",
        agency="방송통신위원회 미디어다양성정책과",
        phone="129",
        website="https://www.bokjiro.go.kr",
    ),
    _PanelSeed(
        serv_id="WLF00003211",
        title="장애인보조기기 교부",
        summary="저소득 장애인에게 보청기·진동알람 등 보조기기를 교부합니다.",
        agency="보건복지부 장애인자립기반과",
        phone="129",
        website="https://www.bokjiro.go.kr",
    ),
)


_DETAIL_URL = (
    "http://apis.data.go.kr/B554287/NationalWelfareInformationsV001"
    "/NationalWelfaredetailedV001"
)


_cache_lock = threading.Lock()
_cache: list[dict[str, Any]] | None = None
_warmup_started = False


def _text(node: ET.Element | None, tag: str) -> str:
    if node is None:
        return ""
    child = node.find(tag)
    if child is None or not child.text:
        return ""
    return html.unescape(child.text).strip()


def _extract_apply_steps(item: ET.Element) -> list[str]:
    """Pull each <applmetList>'s servSeDetailLink (the human-readable step)."""
    steps: list[str] = []
    for entry in item.findall("applmetList"):
        link = _text(entry, "servSeDetailLink")
        if link:
            steps.append(link)
    return steps


def _fetch_one(serv_id: str, service_key: str, timeout: float = 3.0) -> dict[str, Any] | None:
    params = {
        "serviceKey": service_key,
        "servId": serv_id,
        "wlfareInfoReldBztpCd": "01",
        "callTp": "D",
    }
    try:
        resp = requests.get(_DETAIL_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except (requests.RequestException, ET.ParseError) as exc:
        print(f"[welfare_panel] fetch failed for {serv_id}: {exc}")
        return None

    if _text(root, "resultCode") not in ("", "0"):
        print(f"[welfare_panel] {serv_id} resultCode={_text(root, 'resultCode')}")
        return None

    return {
        "title_official": _text(root, "servNm"),
        "outline": _text(root, "wlfareInfoOutlCn"),
        "target": _text(root, "tgtrDtlCn"),
        "benefit": _text(root, "alwServCn"),
        "apply_steps": _extract_apply_steps(root),
        "detail_link": (
            "https://www.bokjiro.go.kr/ssis-tbu/twataa/wlfareInfo/"
            f"moveTWAT52011M.do?wlfareInfoId={serv_id}&wlfareInfoReldBztpCd=01"
        ),
    }


def _build_panel(service_key: str | None) -> list[dict[str, Any]]:
    panel: list[dict[str, Any]] = []
    for seed in _PANEL_SEEDS:
        live = _fetch_one(seed.serv_id, service_key) if service_key else None
        card = {
            **asdict(seed),
            "detail_link": (
                "https://www.bokjiro.go.kr/ssis-tbu/twataa/wlfareInfo/"
                f"moveTWAT52011M.do?wlfareInfoId={seed.serv_id}"
                "&wlfareInfoReldBztpCd=01"
            ),
            "apply_steps": [],
        }
        if live:
            card["detail_link"] = live["detail_link"] or card["detail_link"]
            card["apply_steps"] = live["apply_steps"]
            if live["outline"]:
                card["summary"] = live["outline"]
        panel.append(card)
    return panel


def _start_warmup_if_needed() -> None:
    """Kick off a background fetch of the live panel data — once per process.

    Request handlers must NEVER block on data.go.kr. Synchronous callers get
    instant seed-only data; the warmup upgrades the cache once the network call
    returns.
    """
    global _warmup_started
    if _warmup_started or _cache is not None:
        return
    with _cache_lock:
        if _warmup_started or _cache is not None:
            return
        _warmup_started = True

    def _worker() -> None:
        global _cache
        service_key = os.environ.get("PUBLIC_DATA_API_KEY")
        if not service_key:
            print("[welfare_panel] PUBLIC_DATA_API_KEY not set - seed-only")
            with _cache_lock:
                _cache = _build_panel(None)
            return
        try:
            panel = _build_panel(service_key)
            with _cache_lock:
                _cache = panel
            print("[welfare_panel] warmup done (live data)")
        except Exception as exc:
            print(f"[welfare_panel] warmup failed, falling back to seeds: {exc}")
            with _cache_lock:
                _cache = _build_panel(None)

    threading.Thread(
        target=_worker, daemon=True, name="welfare-panel-warmup"
    ).start()


def get_welfare_panel() -> list[dict[str, Any]]:
    """Return the cached panel; if warmup hasn't finished, return seed-only data.

    Never blocks on the network — the worst case is a panel without the live
    `apply_steps`, which still gives the user useful contact info (107 등).
    """
    _start_warmup_if_needed()
    if _cache is not None:
        return _cache
    return _build_panel(None)  # seed-only, no network


def panel_for_lookup_key(lookup_key: str | None) -> list[dict[str, Any]] | None:
    """Return the panel iff the matched scenario contains the 복지카드 token."""
    if not lookup_key:
        return None
    if WELFARE_PANEL_TRIGGER_TOKEN not in lookup_key.split("+"):
        return None
    return get_welfare_panel()
