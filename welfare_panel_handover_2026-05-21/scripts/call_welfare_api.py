"""Call the Bokjiro (Ministry of Health and Welfare) welfare-service search API.

Use case
--------
After the sign-language pipeline maps a user utterance to a scenario such as
"복지카드를 잃어버렸어요", we want to surface official guidance — e.g. how to
reissue a 복지카드 (welfare card). This script hits the public API at
data.go.kr ("보건복지부_복지서비스 목록/상세 조회") with a keyword search and
prints the matches, then fetches the detail record for the top hit.

Usage
-----
  # Decoded service key in env var (recommended)
  set PUBLIC_DATA_API_KEY=<your decoded key>
  python scripts/call_welfare_api.py --keyword 복지카드

  # Or pass it directly
  python scripts/call_welfare_api.py --keyword 복지카드재발급 --service-key <key>

The service key from data.go.kr comes in two forms (encoded / decoded).
Pass the *decoded* one — `requests` will URL-encode it for us.
"""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from typing import Any

import requests

LIST_URL = (
    "http://apis.data.go.kr/B554287/NationalWelfareInformationsV001"
    "/NationalWelfarelistV001"
)
DETAIL_URL = (
    "http://apis.data.go.kr/B554287/NationalWelfareInformationsV001"
    "/NationalWelfaredetailedV001"
)

# srchKeyCode values per the API spec: 001=서비스명, 002=서비스내용, 003=통합검색
SEARCH_KEY_INTEGRATED = "003"


def _xml_text(node: ET.Element | None, tag: str) -> str:
    if node is None:
        return ""
    child = node.find(tag)
    return (child.text or "").strip() if child is not None and child.text else ""


def _first_text(node: ET.Element, *tags: str) -> str:
    """Return the first non-empty text among the given tag candidates."""
    for tag in tags:
        value = _xml_text(node, tag)
        if value:
            return value
    return ""


def _dump_item(item: ET.Element) -> str:
    parts = []
    for child in list(item):
        text = (child.text or "").strip() if child.text else ""
        parts.append(f"  <{child.tag}> {text!r}")
    return "\n".join(parts)


def _check_response(root: ET.Element) -> None:
    """Raise if the API returned a non-success result code."""
    # Standard data.go.kr error envelope: <OpenAPI_ServiceResponse><cmmMsgHeader>...
    err_header = root.find(".//cmmMsgHeader")
    if err_header is not None:
        code = _xml_text(err_header, "returnReasonCode")
        msg = _xml_text(err_header, "returnAuthMsg") or _xml_text(err_header, "errMsg")
        raise RuntimeError(f"API error: code={code} msg={msg}")

    # Service-specific header
    result_code = _xml_text(root.find(".//header"), "resultCode") or _xml_text(
        root, "resultCode"
    )
    if result_code and result_code != "0":
        result_msg = _xml_text(root.find(".//header"), "resultMessage") or _xml_text(
            root, "resultMessage"
        )
        raise RuntimeError(f"API error: resultCode={result_code} msg={result_msg}")


def search_welfare_services(
    service_key: str,
    keyword: str,
    page_no: int = 1,
    num_of_rows: int = 10,
    timeout: float = 10.0,
    debug: bool = False,
) -> list[dict[str, str]]:
    """Call the list API and return parsed rows."""
    params = {
        "serviceKey": service_key,
        "callTp": "L",
        "pageNo": str(page_no),
        "numOfRows": str(num_of_rows),
        "srchKeyCode": SEARCH_KEY_INTEGRATED,
        "searchWrd": keyword,
    }
    resp = requests.get(LIST_URL, params=params, timeout=timeout)
    resp.raise_for_status()

    root = ET.fromstring(resp.content)
    _check_response(root)

    items = root.findall(".//servList")
    if debug and items:
        print("\n[DEBUG] raw tags in first <servList> item:")
        print(_dump_item(items[0]))

    rows: list[dict[str, str]] = []
    for item in items:
        rows.append(
            {
                # Tag name varies across API versions — try several candidates.
                "wlfareInfoId": _first_text(
                    item,
                    "WLFARE_INFO_ID", "wlfareInfoId",
                    "SERV_ID", "servId",
                    "WLFARE_INFO_NO",
                ),
                "servNm": _first_text(item, "servNm", "SERV_NM"),
                "servDgst": _first_text(item, "servDgst", "SERV_DGST"),
                "jurMnofNm": _first_text(item, "jurMnofNm", "JUR_MNOF_NM"),
                "jurOrgNm": _first_text(item, "jurOrgNm", "JUR_ORG_NM"),
                "bizChrDeptNm": _first_text(
                    item, "bizChrDeptNm", "BIZ_CHR_DEPT_NM"
                ),
                "servDtlLink": _first_text(item, "servDtlLink", "SERV_DTL_LINK"),
                "intrsThemaArray": _first_text(item, "intrsThemaArray"),
                "lifeArray": _first_text(item, "lifeArray"),
                "trgterIndvdlArray": _first_text(item, "trgterIndvdlArray"),
            }
        )
    return rows


_DETAIL_FIELD_MAP = {
    "servNm": ("servNm", "SERV_NM"),
    "wlfareInfoOutlCn": ("wlfareInfoOutlCn", "WLFARE_INFO_OUTL_CN"),
    "tgtrDtlCn": ("tgtrDtlCn", "TGTR_DTL_CN"),
    "slctCritCn": ("slctCritCn", "SLCT_CRIT_CN"),
    # Real tag is `alwServCn` (not `alwSrvCn`) per the live API response.
    "alwServCn": ("alwServCn", "alwSrvCn", "ALW_SRV_CN"),
    "rceptInstNm": ("rceptInstNm", "RCEPT_INST_NM"),
}

# These appear as repeated <tag>...</tag> elements with nested children.
_DETAIL_LIST_TAGS = (
    "applmetList",       # 신청방법
    "inqplCtadrList",    # 문의처
    "inqplHmpgReldList", # 관련 사이트
    "baslawList",        # 근거법령
)


def _collect_list_items(item: ET.Element, tag: str) -> list[str]:
    """For repeated container tags, flatten each instance into a readable string."""
    rendered: list[str] = []
    for node in item.findall(tag):
        # Try direct text first.
        direct = (node.text or "").strip() if node.text else ""
        children = list(node)
        if direct and not children:
            rendered.append(direct)
            continue
        # Otherwise concatenate child texts as "tag: value".
        parts: list[str] = []
        for child in children:
            child_text = (child.text or "").strip() if child.text else ""
            if child_text:
                parts.append(f"{child.tag}: {child_text}")
        if parts:
            rendered.append(" / ".join(parts))
    return [r for r in rendered if r]


def _detail_param_attempts(
    service_key: str, wlfare_info_id: str, reld_bztp_cd: str
) -> list[tuple[str, dict[str, str]]]:
    """Param sets for the detail endpoint.

    Verified working combo (with this provider's current deployment):
      servId + wlfareInfoReldBztpCd + callTp=D
    Other historic variants kept as fallbacks in case the spec shifts again.
    """
    base = {"serviceKey": service_key}
    return [
        (
            "servId + wlfareInfoReldBztpCd + callTp=D",
            {**base, "servId": wlfare_info_id,
             "wlfareInfoReldBztpCd": reld_bztp_cd, "callTp": "D"},
        ),
        (
            "servId + wlfareInfoReldBztpCd",
            {**base, "servId": wlfare_info_id,
             "wlfareInfoReldBztpCd": reld_bztp_cd},
        ),
        (
            "wlfareInfoId + wlfareInfoReldBztpCd + callTp=D",
            {**base, "wlfareInfoId": wlfare_info_id,
             "wlfareInfoReldBztpCd": reld_bztp_cd, "callTp": "D"},
        ),
    ]


def get_welfare_detail(
    service_key: str,
    wlfare_info_id: str,
    reld_bztp_cd: str = "01",
    timeout: float = 10.0,
    debug: bool = False,
) -> dict[str, str]:
    """Call the detail API, trying several documented parameter combinations.

    Returns the first successful payload. Raises RuntimeError if every
    attempt returns resultCode=40 (or another error).
    """
    last_error = "no attempts made"
    for label, params in _detail_param_attempts(
        service_key, wlfare_info_id, reld_bztp_cd
    ):
        if debug:
            redacted = {
                k: ("***" if k == "serviceKey" else v) for k, v in params.items()
            }
            print(f"\n[DEBUG] detail attempt: {label} → {redacted}")

        resp = requests.get(DETAIL_URL, params=params, timeout=timeout)
        resp.raise_for_status()

        if debug:
            print(f"[DEBUG] response (first 400 chars):\n{resp.text[:400]}")

        try:
            root = ET.fromstring(resp.content)
        except ET.ParseError as exc:
            last_error = f"XML parse error: {exc}"
            continue

        try:
            _check_response(root)
        except RuntimeError as exc:
            last_error = str(exc)
            continue

        item = (
            root.find(".//servDtlInfo")
            or root.find(".//item")
            or root.find(".//body")
            or root
        )
        if debug:
            print("[DEBUG] raw tags in detail item:")
            print(_dump_item(item))

        out: dict[str, str] = {}
        for key, candidates in _DETAIL_FIELD_MAP.items():
            out[key] = _first_text(item, *candidates)
        for list_tag in _DETAIL_LIST_TAGS:
            items_out = _collect_list_items(item, list_tag)
            out[list_tag] = "\n".join(items_out)

        if any(out.values()):
            if debug:
                print(f"[DEBUG] success with: {label}")
            return out
        last_error = "empty payload"

    raise RuntimeError(f"all detail attempts failed; last error: {last_error}")


def _print_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("(no matches)")
        return
    for i, row in enumerate(rows, 1):
        print(f"\n[{i}] {row.get('servNm', '(이름없음)')}")
        print(f"    ID         : {row.get('wlfareInfoId', '')}")
        print(f"    소관부처   : {row.get('jurMnofNm', '')}")
        print(f"    담당기관   : {row.get('jurOrgNm', '')}")
        print(f"    담당부서   : {row.get('bizChrDeptNm', '')}")
        digest = row.get("servDgst", "")
        if digest:
            print(f"    요약       : {digest}")
        for label, key in (
            ("관심주제   ", "intrsThemaArray"),
            ("생애주기   ", "lifeArray"),
            ("대상특성   ", "trgterIndvdlArray"),
        ):
            value = row.get(key, "")
            if value:
                print(f"    {label}: {value}")
        link = row.get("servDtlLink", "")
        if link:
            print(f"    상세링크   : {link}")


def _print_detail(detail: dict[str, str]) -> None:
    labels = {
        "servNm": "서비스명",
        "wlfareInfoOutlCn": "서비스 개요",
        "tgtrDtlCn": "지원대상",
        "slctCritCn": "선정기준",
        "alwServCn": "지원내용",
        "applmetList": "신청방법",
        "rceptInstNm": "접수기관",
        "inqplCtadrList": "문의처",
        "inqplHmpgReldList": "관련 사이트",
        "baslawList": "근거법령",
    }
    for key, label in labels.items():
        value = detail.get(key, "")
        if value:
            print(f"\n■ {label}\n{value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keyword", default="복지카드", help="검색어 (default: 복지카드)")
    parser.add_argument("--page-no", type=int, default=1)
    parser.add_argument("--num-rows", type=int, default=10)
    parser.add_argument(
        "--service-key",
        default=os.environ.get("PUBLIC_DATA_API_KEY"),
        help="data.go.kr 인증키 (decoded). 환경변수 PUBLIC_DATA_API_KEY로도 지정 가능.",
    )
    parser.add_argument(
        "--with-detail",
        action="store_true",
        help="상위 1건에 대해 상세조회까지 시도 (API 버전에 따라 NO DATA FOUND가 날 수 있음)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="응답 XML의 실제 태그를 덤프해 확인합니다.",
    )
    args = parser.parse_args(argv)

    if not args.service_key:
        print(
            "ERROR: 인증키가 없습니다. PUBLIC_DATA_API_KEY 환경변수를 설정하거나 "
            "--service-key 로 전달하세요.",
            file=sys.stderr,
        )
        return 2

    print(f"검색어: {args.keyword!r}  (page={args.page_no}, rows={args.num_rows})")
    rows = search_welfare_services(
        service_key=args.service_key,
        keyword=args.keyword,
        page_no=args.page_no,
        num_of_rows=args.num_rows,
        debug=args.debug,
    )
    print(f"\n총 {len(rows)}건 매칭")
    _print_rows(rows)

    if not rows or not args.with_detail:
        return 0

    top = rows[0]
    info_id = top.get("wlfareInfoId")
    if not info_id:
        print("\n(상위 결과에 wlfareInfoId가 없어 상세조회를 건너뜁니다)")
        return 0

    print(f"\n=== 상위 1건 상세조회 (wlfareInfoId={info_id}) ===")
    try:
        detail = get_welfare_detail(args.service_key, info_id, debug=args.debug)
    except RuntimeError as exc:
        print(f"상세조회 실패: {exc}")
        print(f"(웹 페이지로 확인: {top.get('servDtlLink', '')})")
        return 0
    _print_detail(detail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
