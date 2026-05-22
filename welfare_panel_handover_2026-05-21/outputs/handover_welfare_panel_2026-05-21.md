# 복지 안내 슬라이드 패널 — 통합 가이드 (2026-05-21)

수어 키오스크에서 **"복지카드를 잃어버렸어요"** 시나리오가 감지되면 우측 대화창 위에 청각장애인 지원 공공서비스 슬라이드 카드 3개를 자동 전환 표시합니다.

본 모듈은 **완전 standalone**으로 설계됐습니다. 팀원이 작업 중인 `app.py` / `useSignLanguage.ts` / `PatientKiosk.tsx` / `types.ts` 등 **기존 파일은 단 한 줄도 수정하지 않습니다**. 본 문서의 §4 통합 패치 두 곳만 적용하면 동작합니다.

---

## 1. 모듈 구성 (모두 새 파일 — 기존 파일 미수정)

### 백엔드
| 파일 | 역할 |
|---|---|
| [`backend/inference/welfare_panel.py`](../backend/inference/welfare_panel.py) | 트리거 정책, 메모리 캐시, data.go.kr fetch 워밍업 |
| [`backend/inference/__init__.py`](../backend/inference/__init__.py) | 패키지 마커 (빈 파일) |
| [`backend/welfare/__init__.py`](../backend/welfare/__init__.py) | 패키지 마커 (빈 파일) |
| [`backend/welfare/routes.py`](../backend/welfare/routes.py) | `welfare_bp` Flask Blueprint — `GET /api/welfare_panel` 제공 |

### 프론트엔드
| 파일 | 역할 |
|---|---|
| [`web/src/components/WelfarePanel.tsx`](../web/src/components/WelfarePanel.tsx) | 슬라이드 카드 컴포넌트 + `WelfarePanelItem` 타입 export |
| [`web/src/hooks/useWelfarePanel.ts`](../web/src/hooks/useWelfarePanel.ts) | `lookupKey`만 받으면 자체 fetch + state 관리하는 훅 |

### 부속
| 파일 | 역할 |
|---|---|
| [`scripts/call_welfare_api.py`](../scripts/call_welfare_api.py) | 외부 API 단독 호출/검증 스크립트 (개발용) |

---

## 2. 트리거 조건

응답에 `welfare_panel`이 채워지는 경우는 **`lookup_key`의 `+` 분리 토큰 중 `WORD0579`(복지카드)가 들어 있을 때만**입니다.

| 매칭 예시 `lookup_key` | 의미 | 패널 |
|---|---|---|
| `WORD0579` | 복지카드 (단독 인식) | ✅ |
| `SEN0322+WORD0579` | "복지카드를 잃어버렸어요" | ✅ |
| `SEN1817+WORD0579` | "복지카드를 잃어버렸어요" (변형) | ✅ |
| `SEN0278+WORD0579` | "지하철에서 복지카드를 잃어버렸어요" | ✅ |
| 시나리오 lookup에 추후 추가되는 `SEN????+WORD0579` 류 | — | ✅ (자동) |
| `SEN0354`, `WORD0602+WORD1282` 등 비복지카드 키 | — | ❌ |

정의 위치: [`backend/inference/welfare_panel.py`](../backend/inference/welfare_panel.py) — `WELFARE_PANEL_TRIGGER_TOKEN` 상수와 `panel_for_lookup_key` 함수.

---

## 3. 데이터 흐름

```
[수어 인식 / 데모 클립]
        │ (팀원 코드의 기존 파이프라인, 변경 없음)
        ▼
응답에 lookup_key 포함 (예: "SEN0322+WORD0579")
        │
        ▼
[프론트엔드] useWelfarePanel(lookupKey)
        │ lookupKey 변화 감지 → 자체 fetch
        ▼
GET /api/welfare_panel?lookup_key=SEN0322+WORD0579
        │
        ▼
[backend/welfare/routes.py] welfare_bp
        │ panel_for_lookup_key(key) 호출
        ▼
[backend/inference/welfare_panel.py]
        │ 트리거 토큰 매칭 → 메모리 캐시(or seed) 반환
        │ (백그라운드 워밍업이 data.go.kr에서 라이브 데이터 보강)
        ▼
응답 { "welfare_panel": [3 cards], "lookup_key": "..." }
        │
        ▼
[useWelfarePanel] setWelfarePanel(panel) → 컴포넌트 렌더
```

---

## 4. 통합 패치 (팀원이 추가할 부분)

### 4.1 백엔드 — `app.py`에 한 줄 추가

기존 다른 Blueprint 등록 자리 옆에 한 줄만:

```python
# 기존 코드:
# from auth.routes import auth_bp
# from notification.routes import notification_bp
# from summary.routes import summary_bp
# ...
# app.register_blueprint(auth_bp)
# app.register_blueprint(summary_bp)
# app.register_blueprint(notification_bp)

# ↓ 추가
from welfare.routes import welfare_bp
app.register_blueprint(welfare_bp)
```

그 외 `app.py`의 어떤 핸들러도 수정할 필요 없음. `/api/predict`, `/api/gloss_to_text`, `predict_dual_scenario` 응답 형식은 **그대로 유지**됩니다.

### 4.2 프론트엔드 — `PatientKiosk.tsx`에 두 곳 추가

```tsx
// (1) import 추가
import { WelfarePanel } from '../components/WelfarePanel'
import { useWelfarePanel } from '../hooks/useWelfarePanel'

// (2) 컴포넌트 안 — 기존 useSignLanguage 훅 호출 다음 줄
const { welfarePanel, dismiss } = useWelfarePanel(
  currentPrediction?.scenario?.lookup_key
)

// (3) JSX — 우측 대화창 메시지 헤더 바로 아래
{welfarePanel.length > 0 && (
  <div className="shrink-0 px-6 pt-4 pb-2 bg-slate-900">
    <WelfarePanel items={welfarePanel} onClose={dismiss} />
  </div>
)}
```

> 데모 글로스 변환 경로(`/api/gloss_to_text`)에서도 패널이 뜨게 하려면, **§4.3** 참고.

### 4.3 (선택) 데모 글로스 변환 경로에서도 트리거하려면

`useSignLanguage` 내부의 `convertDemoGlossToText`는 `/api/gloss_to_text` 응답을 메시지로만 사용하고 `lookup_key`를 외부로 노출하지 않습니다. 데모에서도 패널을 띄우려면 두 가지 옵션 중 하나:

**옵션 A — 가장 비침습적 (권장)**

데모 라벨이 시나리오 lookup hit이면 `currentPrediction`도 동일 `scenario.lookup_key`로 업데이트되도록 팀원의 `useSignLanguage`를 운영하시면 자동 동작합니다. 별도 작업 불필요.

**옵션 B — useSignLanguage가 lookup_key 노출**

훅 반환값에 `lastLookupKey: string | null` 같은 필드를 한 개 추가하고, 데모/라이브 모두 결정 시점에 그 state를 갱신. PatientKiosk에서 그 값을 `useWelfarePanel(lastLookupKey)`로 넘김. 본 모듈은 이 변경 없이도 라이브 경로에서는 정상 동작합니다.

---

## 5. 백엔드 응답 스키마

### `GET /api/welfare_panel?lookup_key=<key>`

```jsonc
// 트리거 매칭 시
{
  "lookup_key": "SEN0322+WORD0579",
  "welfare_panel": [
    {
      "serv_id": "WLF00003219",
      "title": "통신중계서비스 (107 손말이음센터)",
      "summary": "수어 통역사가 전화 통화를 실시간 중계해 드립니다.",
      "agency": "한국지능정보사회진흥원 손말이음센터",
      "phone": "107",
      "website": "http://107.kr",
      "detail_link": "https://www.bokjiro.go.kr/...",
      "apply_steps": [
        "거주지 읍/면/동 주민센터, ... '서비스 신청'",
        "..."
      ]
    },
    { /* 시각·청각장애인용 TV 보급 (WLF00000104) */ },
    { /* 장애인보조기기 교부 (WLF00003211) */ }
  ]
}

// 트리거 미매칭 (예: lookup_key=SEN0354) — 항상 200, 빈 배열
{ "lookup_key": "SEN0354", "welfare_panel": [] }

// lookup_key 누락
{ "lookup_key": "", "welfare_panel": [] }
```

**상태코드는 항상 200**입니다. 프론트는 `welfare_panel.length === 0`이면 패널을 렌더하지 않으면 됩니다.

`apply_steps`는 백엔드 부팅 직후 첫 호출 1~2건에서는 빈 배열일 수 있습니다(백그라운드 워밍업 진행 중). 워밍업 완료 후 다음 호출부터 5단계 본문이 채워집니다. 핵심 안내(제목, 전화번호, 운영기관)는 seed 데이터로 즉시 응답합니다.

---

## 6. 외부 API (data.go.kr)

### 6.1 사용 API

**한국사회보장정보원 — 중앙부처복지서비스** (`B554287/NationalWelfareInformationsV001`)

### 6.2 정답 파라미터 (디버깅 시 자주 헷갈리는 부분)

| 오퍼레이션 | 엔드포인트 | 파라미터 |
|---|---|---|
| 목록조회 | `.../NationalWelfarelistV001` | `serviceKey`, `callTp=L`, `srchKeyCode=003`, `searchWrd`, `pageNo`, `numOfRows` |
| **상세조회** | `.../NationalWelfaredetailedV001` | `serviceKey`, **`servId`**, `wlfareInfoReldBztpCd=01`, **`callTp=D`** |

⚠️ **함정**: 복지로 웹페이지 URL은 `wlfareInfoId=...`를 쓰지만 **API는 `servId`**를 받습니다. `wlfareInfoId`로 보내면 `resultCode=40 NO DATA FOUND`. 시간 잡아먹는 함정이라 주의.

### 6.3 환경변수

```powershell
# Windows 영구 등록 (한 번만)
setx PUBLIC_DATA_API_KEY "<디코딩된 서비스키>"
# 등록 후 백엔드 재시작 (새 프로세스부터 환경변수가 보임)
```

- **디코딩된 키**를 사용 (`requests`가 자동 URL 인코딩). encoded 키 넣으면 이중 인코딩되어 인증 실패.
- 미설정 상태에서도 모듈은 정상 동작하며 패널은 seed-only로 표시됩니다 (전화/사이트는 보임, `apply_steps`만 비어 있음).

### 6.4 카탈로그 한계

이 API의 카탈로그는 **사회보장 사업(혜택·급여)** 위주이고, "복지카드 발급/재발급" 같은 **민원 행정 서비스는 색인되어 있지 않습니다**. 즉, 진짜 "재발급 방법 안내"는 별도 API(정부24 민원안내)가 필요. 본 패널은 그 대신 "복지카드 소지자가 받을 수 있는 서비스" 안내로 우회한 디자인입니다.

---

## 7. 운영 / 검증

### 7.1 단독 검증 (통합 전에 확인 가능)

백엔드 켜진 상태에서:

```powershell
$body = '{}'  # 사용 안 함
Invoke-RestMethod -Uri "http://localhost:5000/api/welfare_panel?lookup_key=SEN0322%2BWORD0579" `
  | ConvertTo-Json -Depth 5
```

```bash
curl 'http://localhost:5000/api/welfare_panel?lookup_key=SEN0322%2BWORD0579'
```

응답에 `welfare_panel` 배열 3개가 들어오면 백엔드 통합 OK.

### 7.2 자주 보는 로그 패턴

| 로그 메시지 | 의미 |
|---|---|
| `[welfare_panel] PUBLIC_DATA_API_KEY not set - seed-only` | API 키 미설정. seed로만 동작 |
| `[welfare_panel] warmup done (live data)` | 백그라운드 워밍업 완료. 이후 풀 데이터 |
| `[welfare_panel] fetch failed for WLFxxxxx: <err>` | 개별 사업 fetch 실패. 다른 사업은 계속 시도 |
| `[welfare_panel] module unavailable: <err>` | welfare_panel 모듈 import 실패. 응답은 빈 배열로 정상 반환 |

### 7.3 패널이 안 뜰 때 체크리스트

1. **백엔드 응답 확인**: §7.1 단독 호출로 `welfare_panel` 배열이 채워져 오는지
2. 빈 배열로 오면 → `lookup_key`가 트리거 조건에 맞는지(`WORD0579` 토큰 포함) 확인
3. 응답엔 있는데 화면에 안 뜬다면 → 프론트에서 `currentPrediction?.scenario?.lookup_key`가 실제로 채워지는지 (브라우저 콘솔/디버거)
4. vite HMR이 hook 변경을 놓치면 → **F5 새로고침**

---

## 8. 확장 포인트

### 8.1 트리거 정책 변경

[`backend/inference/welfare_panel.py`](../backend/inference/welfare_panel.py):

```python
# 기본: WORD0579(복지카드) 토큰이 lookup_key에 들어 있으면 트리거
WELFARE_PANEL_TRIGGER_TOKEN = "WORD0579"

def panel_for_lookup_key(lookup_key):
    if not lookup_key:
        return None
    if WELFARE_PANEL_TRIGGER_TOKEN not in lookup_key.split("+"):
        return None
    return get_welfare_panel()
```

여러 트리거 단어를 동시에 인정하려면 set으로 확장:

```python
TRIGGER_TOKENS = frozenset({"WORD0579", "WORD1234"})  # 예시
...
if not (TRIGGER_TOKENS & set(lookup_key.split("+"))):
    return None
```

### 8.2 카드 추가/교체

같은 파일의 `_PANEL_SEEDS` 튜플에 `_PanelSeed(serv_id="WLFxxxxxxxx", ...)` 추가. data.go.kr에서 해당 servId의 상세조회가 가능한지 사전 검증 (`scripts/call_welfare_api.py` 활용).

### 8.3 시나리오별로 다른 카드 묶음

현재는 트리거 조건이 어떻든 같은 3개 카드. 시나리오별 큐레이션을 하려면 `panel_for_lookup_key`에서 lookup_key를 보고 다른 seed 셋을 반환하도록 분기.

### 8.4 정부24 민원안내로 확장 (재발급 절차 진짜 안내)

data.go.kr에서 **"행정안전부_민원안내 및 신청서비스 정보"** API를 추가 활용신청하면 실제 "장애인등록증 발급 신청" 행정 절차/구비서류를 받을 수 있습니다. 현 모듈 옆에 `gov24_minwon.py` 같은 모듈을 추가하고, 응답을 합쳐 반환하도록 확장 가능.

---

## 9. 알려진 한계

1. **카탈로그에 "복지카드 재발급" 없음**: 본 API는 사회보장 사업만 색인. 진짜 재발급 절차 안내는 정부24 API 필요 (§8.4).
2. **데모 글로스 변환 경로**: `useSignLanguage`가 `lookup_key`를 외부에 노출하지 않으므로 라이브 경로(`currentPrediction.scenario.lookup_key`)만 자동 트리거. 데모도 지원하려면 §4.3 옵션 B 적용.
3. **첫 호출 응답이 `apply_steps` 비어있음**: 백그라운드 워밍업이 끝나기 전이라면 정상. 다음 호출부터 채워짐.
4. **시나리오 분기 단일**: §8.3 미구현.

---

## 10. 파일 일람

### 새로 추가된 파일 (이 모듈)
- [`backend/inference/welfare_panel.py`](../backend/inference/welfare_panel.py)
- [`backend/inference/__init__.py`](../backend/inference/__init__.py)
- [`backend/welfare/__init__.py`](../backend/welfare/__init__.py)
- [`backend/welfare/routes.py`](../backend/welfare/routes.py)
- [`web/src/components/WelfarePanel.tsx`](../web/src/components/WelfarePanel.tsx)
- [`web/src/hooks/useWelfarePanel.ts`](../web/src/hooks/useWelfarePanel.ts)
- [`scripts/call_welfare_api.py`](../scripts/call_welfare_api.py)

### 팀원이 추가 수정할 파일 (§4 참고)
- `backend/app.py` — Blueprint register 한 줄
- `web/src/pages/PatientKiosk.tsx` — import 2줄 + 훅 호출 + JSX 블록 한 곳

위 외에는 **기존 어떤 파일도 수정하지 않습니다.**

---

## 부록 A. 청각장애인 관련 8개 사업 전체 패널 구성

현재 기본 패널은 3장(`WLF00003219`, `WLF00000104`, `WLF00003211`)만 들어있습니다. 청각장애인 대상 사업 8개 전체를 슬라이드로 보여주려면 [`backend/inference/welfare_panel.py`](../backend/inference/welfare_panel.py)의 `_PANEL_SEEDS` 튜플만 교체하면 됩니다.

### A.1 시드(6개 필드) vs 응답 카드(8개 필드)

응답으로 프론트에 나가는 카드 dict는 **8개 필드**입니다. 하지만 시드에 직접 박는 건 **6개**뿐이고, 나머지 **2개**는 모듈이 자동 채워줍니다.

```python
@dataclass(frozen=True)
class _PanelSeed:                  # ← 팀원이 직접 채우는 6개 필드
    serv_id: str                   # data.go.kr의 servId (이 ID로 상세조회 → 라이브 데이터 보강)
    title: str                     # 사용자에게 보일 카드 제목 (워밍업 후에도 유지)
    summary: str                   # 한 줄 요약 — 워밍업 후 응답의 wlfareInfoOutlCn으로 덮어씀
    agency: str                    # 운영 부처/기관 (워밍업 후에도 유지)
    phone: str                     # 큰 글씨로 노출되는 대표 전화번호 (워밍업 후에도 유지)
    website: str                   # 보유는 하지만 현재 컴포넌트는 detail_link만 노출
```

`_build_panel()`이 시드를 dict로 변환하면서 다음 두 필드를 **자동으로 추가**:

```python
card = {
    **asdict(seed),                # 위 6개 필드
    "detail_link": (               # ← serv_id로부터 자동 생성 (복지로 상세페이지 URL)
        "https://www.bokjiro.go.kr/ssis-tbu/twataa/wlfareInfo/"
        f"moveTWAT52011M.do?wlfareInfoId={seed.serv_id}"
        "&wlfareInfoReldBztpCd=01"
    ),
    "apply_steps": [],             # ← 처음엔 빈 배열, 라이브 워밍업 후 5단계로 채워짐
}
```

### A.1.1 각 필드의 출처와 라이브 갱신 여부

| 응답 필드 | 출처 | 라이브 워밍업 후 |
|---|---|---|
| `serv_id` | 시드 그대로 | 유지 |
| `title` | 시드 그대로 | 유지 |
| `summary` | 시드 (fallback) | **응답의 `wlfareInfoOutlCn`(서비스 개요)로 덮어씀** |
| `agency` | 시드 그대로 | 유지 |
| `phone` | 시드 그대로 | 유지 (응답엔 여러 개라 시드값이 안정적) |
| `website` | 시드 그대로 | 유지 (응답엔 여러 개라 시드값이 안정적) |
| `detail_link` | `serv_id`로부터 자동 생성 (복지로 URL) | 응답에 다른 detail_link 있으면 덮어씀, 없으면 자동 생성값 유지 |
| `apply_steps` | `[]` (빈 배열) | **응답의 `<applmetList>`에서 추출한 5단계로 채워짐** |

따라서 팀원이 신경 쓸 건 **6개 시드 필드만**입니다. `detail_link`와 `apply_steps`는 모듈이 알아서 처리합니다.

### A.2 8개 사업 정보 (라이브 호출로 수집한 데이터)

청각장애인 우선 노출 순서로 정리. 각 phone은 응답의 `<inqplCtadrList>` 중 **실제 신청·문의에 가장 유용한 한 곳**을 골랐습니다(여러 개 있을 경우 사용자 케이스에 맞춰 선택).

| 순 | ID | 사업명 | 소관부처 | 추천 전화 | 추천 사이트 |
|---|---|---|---|---|---|
| 1 | WLF00003219 | 통신중계서비스 제공 | 과학기술정보통신부 통신경쟁정책과 | **107** | http://107.kr |
| 2 | WLF00000104 | 시각·청각장애인용 TV 보급사업 | 방송통신위원회 미디어다양성정책과 | **1688-4596** (시청자미디어재단) | https://tv.kcmf.or.kr |
| 3 | WLF00001130 | 선천성 난청검사 및 보청기 지원 | 보건복지부 출산정책과 | **129** (보건복지상담센터) | http://www.e-health.go.kr |
| 4 | WLF00003211 | 장애인보조기기 교부 | 보건복지부 장애인자립기반과 | **1670-5529** (중앙보조기기센터) | http://knat.go.kr |
| 5 | WLF00001136 | 보험급여(건강보험 장애인보조기기) | 보건복지부 보험급여과 | **1577-1000** (국민건강보험공단) | http://www.nhis.or.kr/ |
| 6 | WLF00001062 | 정보통신보조기기 보급 | 과학기술정보통신부 디지털포용정책팀 | **1588-2670** (한국지능정보사회진흥원) | http://www.at4u.or.kr |
| 7 | WLF00004637 | TV수신료 면제 | 방송통신위원회 방송정책기획과 | **1588-1801** (KBS 수신료콜센터) | http://www.kcc.go.kr |
| 8 | WLF00001153 | 장애인보조견전문훈련기관지원 | 보건복지부 장애인정책과 | **031-691-7782** | (응답에 없음 — 복지로 상세링크로 대체) |

> ⚠️ WLF00001153은 응답의 `<inqplHmpgReldList>`가 비어 있어 별도 website가 없습니다. `_PanelSeed`의 `website`는 `https://www.bokjiro.go.kr` 같은 일반 복지로 메인을 넣고, 사용자는 카드의 "복지로에서 자세히 보기" 링크로 이동.

### A.3 그대로 복사·붙여넣기 가능한 `_PANEL_SEEDS`

```python
_PANEL_SEEDS: tuple[_PanelSeed, ...] = (
    _PanelSeed(
        serv_id="WLF00003219",
        title="통신중계서비스 (107 손말이음센터)",
        summary="수어 통역사가 전화 통화를 실시간 중계해 드립니다.",
        agency="과학기술정보통신부 통신경쟁정책과",
        phone="107",
        website="http://107.kr",
    ),
    _PanelSeed(
        serv_id="WLF00000104",
        title="시각·청각장애인용 TV 보급사업",
        summary="자막·수어 방송에 최적화된 맞춤형 TV를 보급합니다.",
        agency="방송통신위원회 미디어다양성정책과",
        phone="1688-4596",
        website="https://tv.kcmf.or.kr",
    ),
    _PanelSeed(
        serv_id="WLF00001130",
        title="선천성 난청검사 및 보청기 지원",
        summary="영유아 선천성 난청 조기진단과 보청기 구입비를 지원합니다.",
        agency="보건복지부 출산정책과",
        phone="129",
        website="http://www.e-health.go.kr",
    ),
    _PanelSeed(
        serv_id="WLF00003211",
        title="장애인보조기기 교부",
        summary="저소득 장애인에게 보청기·진동알람 등 보조기기를 교부합니다.",
        agency="보건복지부 장애인자립기반과",
        phone="1670-5529",
        website="http://knat.go.kr",
    ),
    _PanelSeed(
        serv_id="WLF00001136",
        title="건강보험 장애인보조기기 급여",
        summary="등록 장애인의 보조기기 구입비 일부를 건강보험으로 지원합니다.",
        agency="보건복지부 보험급여과",
        phone="1577-1000",
        website="http://www.nhis.or.kr/",
    ),
    _PanelSeed(
        serv_id="WLF00001062",
        title="정보통신보조기기 보급",
        summary="장애인의 정보화 역량 강화를 위한 정보통신 보조기기를 보급합니다.",
        agency="과학기술정보통신부 디지털포용정책팀",
        phone="1588-2670",
        website="http://www.at4u.or.kr",
    ),
    _PanelSeed(
        serv_id="WLF00004637",
        title="TV수신료 면제",
        summary="장애인·국가유공자 등 취약계층의 TV수신료를 면제합니다.",
        agency="방송통신위원회 방송정책기획과",
        phone="1588-1801",
        website="http://www.kcc.go.kr",
    ),
    _PanelSeed(
        serv_id="WLF00001153",
        title="장애인보조견 전문훈련기관 지원",
        summary="장애인보조견을 통해 청각장애인의 소리 인지 등을 보조합니다.",
        agency="보건복지부 장애인정책과",
        phone="031-691-7782",
        website="https://www.bokjiro.go.kr",
    ),
)
```

### A.4 8개 노출 시 UX 영향

- 슬라이드 자동 전환 간격은 기본 7초. 8장이면 한 바퀴 도는 데 **약 56초**. 진료 처리 대기시간 1~2분 가정 시 사용자가 모든 카드를 한 번씩 보게 됩니다.
- 더 빠르게 보고 싶으면 `<WelfarePanel items={...} intervalMs={5000} />`로 5초 전환 등 prop 조정.
- 사용자가 ✕ 버튼으로 닫으면 같은 진료 세션에서는 다시 안 뜸 (`useWelfarePanel`의 dismissedKeysRef).

### A.5 라이브 데이터 보강 검증

8개로 늘리면 백그라운드 워밍업이 data.go.kr에 8회 호출합니다. timeout(`_fetch_one`의 3초)을 한 번 지연될 수 있으니, 첫 호출 응답에서 `apply_steps`가 비어 있어도 정상. 워밍업 완료 메시지(`[welfare_panel] warmup done (live data)`)를 backend.stdout.log에서 확인.

검증 스크립트:

```powershell
$ids = "WLF00003219","WLF00000104","WLF00001130","WLF00003211","WLF00001136","WLF00001062","WLF00004637","WLF00001153"
foreach ($id in $ids) {
  python scripts/call_welfare_api.py --keyword $id --num-rows 1 --with-detail --no-detail-on-fail
  # 또는 직접:
  # Invoke-RestMethod -Uri "http://localhost:5000/api/welfare_panel?lookup_key=WORD0579" 후 응답 카드 개수 확인
}
```

또는 단순히 `/api/welfare_panel?lookup_key=WORD0579` 한 번 호출 후 응답의 `welfare_panel.length`가 **8**인지 확인.

### A.6 시드 추가/변경 시 점검 포인트

1. **`serv_id` 정확성**: data.go.kr에서 실제로 존재하는 servId인지 [`scripts/call_welfare_api.py`](../scripts/call_welfare_api.py)로 검증
2. **카드 정렬**: `_PANEL_SEEDS` 튜플 순서가 그대로 슬라이드 순서. 가장 핵심적인 사업을 1번 인덱스에 배치 (현재: WLF00003219 통신중계 = 수어 사용자 최우선)
3. **phone 표시 너비**: `WelfarePanel.tsx`의 phone 배지는 한 줄 표시. 매우 긴 번호(예: `031-691-7782`)는 잘 보이지만 더 길어지면 CSS 손볼 필요
4. **website**: 현재 컴포넌트는 카드 하단 "복지로에서 자세히 보기"만 노출하므로 `website` 필드는 미사용. 추후 카드에 별도 링크를 추가할 때 사용 예정 (예: 운영기관 사이트 직링크)
