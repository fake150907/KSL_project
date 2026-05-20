# 🤟 수어 통역 시스템 (Sign Language Interpretation System)

민원인(수어 사용자)와 상담원 간의 실시간 소통을 돕는 **AI 기반 수어 통역 웹 애플리케이션**입니다.  
민원인는 키오스크에서 카메라로 수어를 입력하고, 상담원는 대시보드에서 번역된 텍스트와 음성 인식으로 소통합니다.

---

## 📁 프로젝트 구조

```
src/
├── App.tsx                    # 앱 루트 — 라우팅 및 전역 상태 관리
├── main.tsx                   # React 진입점
├── index.css                  # 전역 스타일
├── types.ts                   # 공용 TypeScript 타입 정의
├── socket.ts                  # Socket.IO 클라이언트 설정
│
├── pages/
│   ├── AgentLaunchScreen.tsx # 상담원 대기 화면 (민원인 도착 알림 수신)
│   ├── AgentDashboard.tsx    # 상담원 상담 대시보드
│   ├── KioskLaunchScreen.tsx  # 민원인 키오스크 시작 화면
│   └── CitizenKiosk.tsx       # 민원인 수어 입력 화면
│
├── hooks/
│   ├── useSignLanguage.ts     # 수어 인식 핵심 로직
│   └── useSpeechRecognition.ts# 음성 인식 핵심 로직
│
└── components/
    ├── ChatMessage.tsx         # 채팅 말풍선 컴포넌트
    ├── VideoFeed.tsx           # 카메라 영상 + 랜드마크 오버레이
    └── SignLanguageStream.tsx  # 수어+음성 통합 스트림 뷰 (데모용)
```

---

## ⚙️ 핵심 아키텍처

### 데이터 흐름

```
[민원인 키오스크]                        [상담원 대시보드]
  카메라 영상 캡처                         음성 인식 (Web Speech API)
      ↓                                        ↓
  /api/predict (백엔드 AI)              ChatMessage (agent)
      ↓                                        ↓
  수어 번역 텍스트                    ←── BroadcastChannel ───→
      ↓                                        ↓
  ChatMessage (citizen)              공유 messages 배열 (App.tsx)
                                          ↓
                                    localStorage 영속화
```

### 탭 간 통신 (BroadcastChannel)

두 개의 채널을 사용합니다.

| 채널 이름 | 용도 |
|---|---|
| `sign-lang-chat` | 채팅 메시지 동기화, 세션 시작/종료 이벤트 |
| `citizen-session-notify` | 민원인 도착 알림, 상담원 입장 확인 신호 |

---

## 📄 파일별 상세 설명

---

### `types.ts` — 공용 타입 정의

앱 전체에서 사용하는 세 가지 핵심 데이터 구조를 정의합니다.

```ts
// AI 모델이 반환하는 수어 예측 결과
interface Prediction {
  label: string | null        // 인식된 수어 단어 (예: '우유', '아프다')
  confidence: number          // 확신도 (0.0 ~ 1.0)
  has_hand?: boolean          // 손이 화면에 있는지 여부
  window_filled?: boolean     // 분석에 필요한 프레임이 충분히 쌓였는지 여부
  top_predictions?: Array<{ label: string; confidence: number }> // 상위 예측 목록
  // ...기타 프레임 추적 관련 필드
}

// 민원인↔상담원 채팅 메시지
interface ChatMessage {
  id: string
  sender: 'citizen' | 'agent'
  text: string
  timestamp: Date
  label?: string  // 말풍선 위 라벨 (예: '수어 번역')
}

// 상담원의 상담 메모
interface AgentNote {
  id: string
  text: string
  tag: '증상' | '관찰' | '처방'
  timestamp: Date
}
```

---

### `App.tsx` — 앱 루트 & 전역 상태 관리

**역할:** 라우팅, 전역 메시지 상태, 탭 간 통신, 세션 생명주기 전체를 담당합니다.

**핵심 기능:**

1. **라우팅** — React Router로 4개의 화면을 연결합니다.

   | URL | 컴포넌트 | 설명 |
   |---|---|---|
   | `/` | `AgentLaunchScreen` | 상담원 대기 화면 |
   | `/agent` | `AgentDashboard` | 상담원 상담 대시보드 |
   | `/kiosk` | `KioskLaunchScreen` | 민원인 시작 화면 |
   | `/kiosk/session` | `CitizenKiosk` | 민원인 수어 입력 화면 |

2. **메시지 브로드캐스팅** — `handleNewMessage`가 호출되면 화면에 표시하는 동시에 `BroadcastChannel`로 다른 탭에 전송하고, `localStorage`에 영속화합니다.

3. **중복 방지** — `seenIds` (Set)로 동일 메시지가 두 번 렌더링되는 것을 막습니다.

4. **세션 관리** — `handleSessionEnd` / `handleSessionReset`으로 상담 시작과 종료를 제어합니다.

```ts
// 핵심 상태
const [messages, setMessages] = useState<ChatMessage[]>([])
const [sessionEnded, setSessionEnded] = useState(false)
const seenIds = useRef<Set<string>>(new Set())
const channelRef = useRef<BroadcastChannel | null>(null)
```

---

### `main.tsx` — React 진입점

`ReactDOM.createRoot`로 React 앱을 마운트합니다. `BrowserRouter`로 라우팅 컨텍스트를 제공합니다.

```ts
ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
)
```

---

### `index.css` — 전역 스타일

Tailwind CSS를 로드하고, `html / body / #root`를 `height: 100%`로 설정해 전체 화면 레이아웃을 보장합니다. 기본 배경색은 `#0b1120` (짙은 네이비)입니다.

---

### `socket.ts` — Socket.IO 클라이언트

`http://localhost:5000` 백엔드 서버와의 실시간 WebSocket 연결을 설정합니다. 자동 연결 및 재연결 옵션이 활성화되어 있습니다. (현재 앱의 주요 통신은 BroadcastChannel을 사용하며, 이 파일은 향후 확장을 위해 준비된 모듈입니다.)

---

### `hooks/useSignLanguage.ts` — 수어 인식 핵심 로직

**역할:** 카메라 영상을 일정 주기로 캡처 → 백엔드 AI API로 전송 → 수어 예측 결과를 받아 채팅 메시지로 변환합니다.

#### 설정 파라미터 (SignLanguageConfig)

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `modelType` | `'cnn_gru'` | AI 모델 종류 (CNN+GRU 또는 LSTM) |
| `confidenceThreshold` | `0.50` | 이 값 이상일 때만 단어로 인정 (낮을수록 민감) |
| `windowSize` | `16` | 한 번에 분석하는 프레임 수 |
| `captureIntervalMs` | `60` | 프레임 캡처 주기 (밀리초) |
| `maxMissingFrames` | `3` | 손이 사라져도 유지하는 최대 프레임 수 |

#### 핵심 알고리즘: Peak Detection (최고점 추출)

단순히 인식 결과를 실시간으로 출력하면 같은 단어가 여러 번 반복되는 문제가 있습니다. 이를 해결하기 위해 **"손이 올라갔다가 내려오는 순간"** 에 가장 높은 신뢰도의 단어 하나만 채팅에 전송합니다.

```
손이 화면에 등장 (has_hand = true)
    → 인식 결과가 들어올 때마다 최고 신뢰도 단어를 peakGestureRef에 갱신

손이 화면에서 사라짐 (has_hand = false)
    → peakGestureRef에 저장된 단어를 채팅에 딱 한 번 전송
    → peakGestureRef를 null로 초기화
```

#### 랜드마크 시각화 (drawLandmarks)

백엔드가 반환한 손/포즈 관절 좌표를 `<canvas>` 위에 직접 그립니다.

| 색상 | 의미 |
|---|---|
| 🔵 파랑 (`#38bdf8`) | 상체 포즈 |
| 🟢 초록 (`#22c55e`) | 왼손 |
| 🟠 주황 (`#f97316`) | 오른손 |

#### 반환값

```ts
return {
  videoRef, canvasRef, landmarkCanvasRef,  // DOM 참조
  isRunning, isDemoMode, currentPrediction, // 상태
  videoDevices, selectedDeviceId, setSelectedDeviceId, // 카메라 선택
  startCamera, stopCamera, getPredictionStatus,         // 제어 함수
}
```

---

### `hooks/useSpeechRecognition.ts` — 음성 인식 로직

**역할:** 상담원의 음성을 텍스트로 변환해 채팅 메시지로 전송하고, 실시간 음성 레벨 시각화 데이터를 제공합니다.

#### 핵심 기능

1. **음성 인식** — 브라우저의 Web Speech API(`SpeechRecognition`)를 사용합니다. 언어는 `ko-KR`(한국어)로 고정됩니다. `continuous: true`로 끊기지 않고 계속 인식합니다. 인식이 끝나면 자동으로 재시작해 끊김 없는 경험을 제공합니다.

2. **음성 레벨 시각화** — Web Audio API로 마이크 입력을 분석해 48개 주파수 버킷의 레벨 배열(`voiceLevels`)을 실시간으로 반환합니다. 이 데이터로 UI에서 이퀄라이저 효과를 구현합니다.

```ts
// 음성 레벨 계산 핵심 로직
const bucketSize = Math.floor(data.length / BUCKET_COUNT)  // 48개 버킷
const nextLevels = Array.from({ length: BUCKET_COUNT }, (_, i) => {
  const avg = data.slice(i * bucketSize, (i+1) * bucketSize).reduce(...) / bucketSize
  return Math.max(VOICE_THRESHOLD_MIN, Math.min(1, avg / MAX_VOLUME))
})
```

#### 반환값

```ts
return {
  isActive,    // 현재 음성 인식 중인지 여부
  voiceLevels, // 48개 float 배열 (0.12 ~ 1.0), 이퀄라이저 UI용
  start,       // 음성 인식 + 시각화 시작
  stop,        // 음성 인식 + 시각화 중지
}
```

---

### `pages/KioskLaunchScreen.tsx` — 민원인 시작 화면

**역할:** 민원인이 상담을 시작하기 전에 보는 첫 화면입니다.

**흐름:**

1. 민원인이 **"상담 시작하기"** 버튼을 누름
2. `citizen-session-notify` 채널로 `{ type: 'citizen_arrived' }` 메시지 브로드캐스팅
3. 민원인 화면은 **"상담원을 기다리는 중"** 대기 화면으로 전환
4. 상담원이 대시보드에서 입장을 확인하면 `{ type: 'agent_ready' }` 신호 수신
5. 자동으로 `/kiosk/session` (수어 입력 화면)으로 이동

---

### `pages/AgentLaunchScreen.tsx` — 상담원 대기 화면

**역할:** 상담원이 상담을 시작하기 전 민원인 도착 알림을 기다리는 화면입니다.

**흐름:**

1. `citizen-session-notify` 채널을 구독하며 대기
2. `{ type: 'citizen_arrived' }` 수신 시 → **"민원인 도착 알림"** 카드로 UI 전환 + 펄스 애니메이션
3. 상담원이 **"상담실 입장하기"** 버튼을 누름
4. `handleSessionReset()` 호출 → 이전 상담 기록 초기화
5. `{ type: 'agent_ready' }` 브로드캐스팅 후 `/agent` 대시보드로 이동

**주의:** 입장 버튼을 누를 때 이전 세션을 자동으로 초기화(`onSessionReset`)하므로 이전 민원인의 대화 내역이 섞이지 않습니다.

---

### `pages/CitizenKiosk.tsx` — 민원인 수어 입력 화면

**역할:** 민원인이 수어로 증상을 전달하는 메인 화면입니다.

**핵심 구성:**
- `useSignLanguage` 훅으로 카메라 + AI 인식 연결
- `VideoFeed` 컴포넌트로 카메라 영상 + 랜드마크 오버레이 표시
- `ChatMessage` 컴포넌트로 번역된 수어 및 상담원 메시지 표시
- 세션 종료 시 (`sessionEnded = true`) 상담 종료 화면 표시

**Props:**

```ts
interface CitizenKioskProps {
  messages: ChatMessage[]         // 전체 대화 내역 (App.tsx에서 내려옴)
  onNewMessage: (msg) => void     // 새 메시지 전송 콜백
  onSessionReset: () => void      // 세션 초기화 콜백
  sessionEnded: boolean           // 상담 종료 여부
}
```

---

### `pages/AgentDashboard.tsx` — 상담원 상담 대시보드

**역할:** 상담원이 민원인의 수어 번역 내용을 확인하고, 음성으로 답변하며 상담 메모를 작성하는 화면입니다.

**핵심 구성:**
- `useSpeechRecognition` 훅으로 상담원 음성 인식
- `useSignLanguage` 훅으로 민원인 카메라 피드 미리보기 (소형)
- `ChatMessage` 컴포넌트로 양방향 대화 표시
- 상담 메모 작성 (`AgentNote` — 증상/관찰/처방 태그)
- **"상담 끝내기"** 버튼으로 세션 종료

**Props:**

```ts
interface AgentDashboardProps {
  messages: ChatMessage[]
  onNewMessage: (msg) => void
  onSessionEnd: () => void        // 상담 종료 콜백
  onSessionReset: () => void
}
```

---

### `components/VideoFeed.tsx` — 카메라 영상 컴포넌트

**역할:** 카메라 영상 스트림과 AI 랜드마크 오버레이를 렌더링합니다.

**특이사항:**
- 영상에 `transform: scaleX(-1)` 적용 → 거울 모드 (카메라 특성상 좌우 반전이 자연스러움)
- 랜드마크 캔버스도 동일하게 반전해 영상과 정확히 일치시킴
- `compact` prop으로 크기를 조절할 수 있음 (상담원 대시보드 소형 뷰에 사용)
- `isRunning`이 `false`이면 빈 아이콘 플레이스홀더를 표시

```tsx
<video style={{ transform: 'scaleX(-1)' }} ... />
<canvas style={{ transform: 'scaleX(-1)' }} ... />  {/* 랜드마크 */}
```

---

### `components/ChatMessage.tsx` — 채팅 말풍선 컴포넌트

**역할:** 단일 채팅 메시지를 말풍선 UI로 렌더링합니다.

**디자인 규칙:**

| 구분 | 정렬 | 색상 |
|---|---|---|
| `sender: 'agent'` (상담원) | 오른쪽 | 파란색 (`rgba(37,99,235,0.75)`) |
| `sender: 'citizen'` (민원인) | 왼쪽 | 초록색 (`#22c55e`) |

말풍선 위에 `label` (예: `'수어 번역'`, `'상담원'`)을 작게 표시합니다. 시간은 `HH:MM` 형식 (한국어)으로 표시합니다.

---

### `components/SignLanguageStream.tsx` — 통합 스트림 뷰 (데모용)

**역할:** 수어 인식 + 음성 인식 + 채팅을 한 화면에 보여주는 독립 데모 컴포넌트입니다. `useSignLanguage`와 `useSpeechRecognition`을 동시에 사용하는 통합 예시 역할을 합니다.

---

### `SignLanguageApp.tsx` — 단일 탭 통합 앱 (레거시)

**역할:** BroadcastChannel 없이 단일 탭 내에서 역할(`launch` → `citizen` / `agent`)을 선택하는 초기 구조입니다. 현재는 `App.tsx`의 라우팅 기반 구조로 대체되었으며, 참고용으로 보존되어 있습니다.

---

## 🔌 백엔드 API 연동

### POST `/api/predict`

수어 인식의 핵심 엔드포인트입니다. `useSignLanguage` 훅이 60ms마다 호출합니다.

**요청 (multipart/form-data):**

| 필드 | 타입 | 설명 |
|---|---|---|
| `frame` | Blob (JPEG) | 캡처된 카메라 프레임 이미지 |
| `frame_id` | string | 프레임 순서 번호 (오래된 응답 무시용) |
| `model_type` | string | `'sequence'` (CNN+GRU) 또는 `'lstm'` |
| `client_id` | string | 브라우저 탭 고유 ID |
| `confidence_threshold` | string | 최소 신뢰도 (기본 `'0.50'`) |
| `window_size` | string | 분석 윈도우 크기 (기본 `'16'`) |

**응답 (JSON):**

```json
{
  "frame_id": 42,
  "prediction": {
    "label": "우유",
    "confidence": 0.87,
    "has_hand": true,
    "window_filled": true,
    "window_progress": 16,
    "window_size": 16,
    "landmarks": {
      "pose": [[x, y, z], ...],
      "left_hand": [[x, y, z], ...],
      "right_hand": [[x, y, z], ...]
    },
    "top_predictions": [
      { "label": "우유", "confidence": 0.87 },
      { "label": "물", "confidence": 0.08 }
    ]
  }
}
```

### POST `/api/messages`

채팅 메시지를 백엔드 DB에 저장합니다. 실패해도 앱 동작에는 영향 없습니다.

---

## 🚀 시작하기

```bash
# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
```

브라우저에서 두 개의 탭을 열어 사용합니다:

- **탭 1 (상담원):** `http://localhost:5173/` → 상담원 대기 화면
- **탭 2 (민원인):** `http://localhost:5173/kiosk` → 민원인 키오스크 화면

민원인이 "상담 시작하기"를 누르면 상담원 탭에 알림이 표시되고, 상담원이 "상담실 입장하기"를 누르면 양쪽 탭이 동시에 상담 화면으로 전환됩니다.

---

## 🛠️ 기술 스택

| 분류 | 기술 |
|---|---|
| UI 프레임워크 | React 18 + TypeScript |
| 라우팅 | React Router v6 |
| 스타일링 | Tailwind CSS |
| 탭 간 통신 | BroadcastChannel API |
| 음성 인식 | Web Speech API |
| 카메라 | MediaDevices API |
| 오디오 시각화 | Web Audio API |
| 실시간 소켓 | Socket.IO (socket.ts) |
| AI 모델 통신 | REST API (`/api/predict`) |