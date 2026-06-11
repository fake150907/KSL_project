# 🌷 피어나 (Pierna)

> **농인의 목소리가 민원 현장에서 다시 피어나도록**
> 주민센터 창구용 AI 양방향 수어 통역 보조 서비스
> 🏆 2026 국민행복 서비스 발굴·창업경진대회 출품작 · Team **오동가영**

<p align="center">
  <img src="https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=white" />
  <img src="https://img.shields.io/badge/TypeScript-5-3178C6?logo=typescript&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-2.3-000000?logo=flask&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/MediaPipe-0.10-00897B?logo=google&logoColor=white" />
  <img src="https://img.shields.io/badge/PostgreSQL-Supabase-4169E1?logo=postgresql&logoColor=white" />
  <img src="https://img.shields.io/badge/Claude_API-Anthropic-D97757" />
</p>

---

## 📖 소개

**피어나(Pierna)** 는 주민센터 창구에서 AI가 **수어와 음성을 실시간으로 상호 통역**하여, 농인이 통역사 동행 없이 직접 민원을 처리할 수 있도록 지원하는 서비스입니다.

농인이 수어로 요청을 전달하면 자연어 문장으로 변환해 직원에게 보여주고, 직원의 음성은 STT를 통해 자막으로 농인에게 전달됩니다. 공무원은 기존 업무 절차를 그대로 수행하고 **AI는 의사소통 통역만 전담**하는 구조로, 공공기관 도입 부담을 최소화했습니다.

> 💬 *"복지카드를 잃어버렸습니다. 재발급 받고 싶습니다."*
> 창구에서 몇 분이면 끝나는 민원이, 수어를 쓰는 농인에게는 통역사 없이는 어렵습니다. 피어나는 그 벽을 허뭅니다.

---

## 🎯 배경

| 지표 | 수치 | 출처 |
|------|------|------|
| 전국 등록 청각장애인 | 약 **44만 명** | 보건복지부, 2024 |
| 수어 주 사용 농인 | 약 **13만 명** | 국립국어원, 2023 |
| 공공기관 이용 시 수어통역 필요 응답 | **62.9%** | 국립국어원, 2023 |
| 한국수어를 농인의 언어로 인식 | **90.8%** | 국립국어원, 2023 |

현행 지원 체계(손말이음센터·출장 통역)는 예약·대기·비용 부담으로 **즉시 대응이 어렵습니다.** 피어나는 창구에 상시 비치된 태블릿으로 **별도 절차 없이 즉시** 대면 상담을 지원합니다.

---

## ✨ 핵심 기능

### 1️⃣ 농인 → 직원 · 수어 인식 및 자연어 변환
- 📹 카메라로 수어를 인식하고 자연스러운 한국어 문장으로 변환
- 🧠 AIHub 한국 수어 데이터셋(WORD 3,000 / SENTENCE 2,000 클래스) 기반 학습
- ✅ 단어 인식률 **97%** · 문장 인식률 **88%**

### 2️⃣ 직원 → 농인 · 음성 실시간 시각화
- 🎙️ 직원 음성을 STT로 실시간 자막 변환
- 💬 필담 없이 자연스러운 양방향 대화, 창구 대화 흐름 중단 없이 지원

### 3️⃣ 대기 중 맞춤형 복지 정보 안내
- 🗂️ 공공데이터포털 중앙부처 복지서비스 API 연동
- 🔗 청각장애인 대상 복지 정보를 카드로 표시 → '복지로' 상세 페이지 연결

### 4️⃣ 상담 종료 후 처리
- 📝 Claude API로 상담 핵심 요약 → 카카오 알림톡 발송
- 🔐 이름·전화번호만 수집, **AES-256 암호화 저장 + 로컬 마스킹**, 개인정보보호법 준수

---

## 🏗️ 시스템 아키텍처

```text
┌─────────────────────┐       ┌──────────────────────┐       ┌──────────────────────────┐
│   농인용 태블릿 (Kiosk) │  ⇄   │   피어나 백엔드 (Flask)   │  ⇄   │  모델 서버 (MediaPipe+분류기) │
│  · 수어 입력 카메라      │       │  · 추론 라우팅 / 세션 관리 │       │  · 프레임 → 랜드마크 추출      │
│  · 자막·보조 수어 표시   │       │  · Claude 자연어 변환     │       │  · CNN-GRU 수어 분류         │
└─────────────────────┘       │  · 요약 / 알림 트리거     │       └──────────────────────────┘
┌─────────────────────┐       └──────────────────────┘
│  상담 직원용 PC 화면    │  ⇄   Socket.IO 실시간 시그널링 (WebRTC 영상/음성)
│  · 인식 텍스트 표시     │
│  · 음성 입력 (STT)     │              ▼
└─────────────────────┘   외부 연동 · Claude API · Web Speech STT · 카카오 알림톡 · 복지로 API
```

---

## 🛠️ 기술 스택

### Frontend
| 기술 | 용도 |
|------|------|
| React 18 + TypeScript | 키오스크 / 상담원 듀얼 화면 UI |
| Vite | 빌드 · 개발 서버 |
| TailwindCSS | 스타일링 |
| Socket.IO Client · WebRTC | 실시간 영상·신호 통신 |
| Web Speech API | 직원 음성 STT |

### Backend
| 기술 | 용도 |
|------|------|
| Flask | REST API 서버 |
| Socket.IO (Node.js) | 실시간 시그널링 서버 |
| PostgreSQL (Supabase) | 민원 세션 저장 |
| cryptography (AES-256-GCM) | 개인정보 암호화 |
| Anthropic Claude API | 수어 글로스 → 자연어 변환, 상담 요약 |
| 카카오 알림톡 API | 상담 요약 발송 |

### AI / Model
| 기술 | 용도 |
|------|------|
| PyTorch | 모델 학습 / 추론 |
| MediaPipe Holistic | 손·상체 keypoint 추출 |
| 1D-CNN + BiGRU + Attention Pooling | 수어 시퀀스 분류 |

---

## 🧠 모델 성능

수어 동작은 시간 흐름에 따라 의미가 달라지므로, **1D-CNN + BiGRU + Attention Pooling** 구조로 단어·문장 단위 시퀀스를 분류합니다.

| 모델 | 대상 | 정확도 (Top-1) |
|------|------|---------------|
| 🟢 **WORD** (단어) | unseen 전문 시연자 | **97.53%** |
| 🔵 **SENTENCE** (문장) | unseen 전문 시연자 | **88.54%** |

### 학습 안정성 (과적합 없음 ✅)
| 모델 | Best Epoch | train-val gap | val Top-1 |
|------|-----------|---------------|-----------|
| WORD | 49/50 | **-0.083** (정상) | 96.28% |
| SENTENCE | 21/50 | **-0.133** (정상) | 92.78% |

> train loss와 val loss가 함께 수렴 → 단순 암기가 아닌 **일반화 가능한 패턴 학습**을 확인했습니다.

### 실환경 적응 (Stage 2 Fine-tune)
청인 직원 촬영 영상 615개를 30% 추가 학습하여 다양한 사용자 환경 적응력을 강화했습니다.

| 모델 | Stage 1 → Stage 2 (Top-1) | macro_f1 |
|------|---------------------------|----------|
| WORD | 40% → **60%** | 22.05 → 59.97 |
| SENTENCE | 13.33% → **61.82%** | — |

### 실제 민원 시나리오 검증
'복지카드 분실 재발급' 11턴 시나리오 기준 — **Top-5 정확도 100%, E2E 점수 0.9045** (운영 기준 0.85 초과 ✅)

---

## 📂 프로젝트 구조

```text
KSL_project/
├── backend/                # Flask 백엔드
│   ├── app.py              # 앱 엔트리 · 블루프린트 등록
│   ├── db.py               # PostgreSQL 연결 · 마스킹 · AES-256 암호화
│   ├── inference/          # 모델 로드 / 추론 라우트
│   ├── session/            # 민원 세션 관리
│   ├── summary/            # Claude 상담 요약
│   ├── notification/       # 카카오 알림톡
│   └── welfare/            # 복지 정보 API
├── web/                    # React + TS 프론트엔드
│   └── src/
│       ├── pages/          # 키오스크 / 상담원 화면
│       ├── components/     # 한글 키보드 · 마스킹 · UI
│       └── hooks/          # 수어 인식 · STT · WebRTC
├── server/                 # Socket.IO 시그널링 서버
└── src/                    # 모델 학습 / 데이터 파이프라인
    ├── models/             # CNN-GRU 모델 정의
    └── data/               # MediaPipe 전처리 · 키포인트 유틸
```

---

## 🚀 실행 방법

### 1. 백엔드
```bash
cd backend
pip install -r requirements.txt
# .env 파일 생성 후 DATABASE_URL, DB_ENCRYPTION_KEY, ANTHROPIC_API_KEY 설정
python app.py
```

### 2. 시그널링 서버
```bash
cd server
npm install
node server.js
```

### 3. 프론트엔드
```bash
cd web
npm install
npm run dev
```

### 🔑 환경 변수 (`backend/.env`)
```env
DATABASE_URL=postgresql://...        # PostgreSQL (Supabase)
DB_ENCRYPTION_KEY=...                # AES-256 키 (64자리 hex)
ANTHROPIC_API_KEY=...                # Claude API
KAKAO_REST_API_KEY=...               # 카카오 알림톡
```

---

## 🔐 개인정보 보호

피어나는 **개인정보 최소 수집 원칙**을 따릅니다.

| 항목 | 처리 방식 |
|------|----------|
| 이름 | 마스킹(`김*수`) 후 AES-256 암호화 저장 |
| 전화번호 | 중간 4자리 마스킹 후 AES-256 암호화 저장 |
| 민감정보(주민번호 등) | 입력 패턴 감지 시 차단 + 알림 |
| 수어 영상 | **저장하지 않음** — 실시간 좌표 추출 후 즉시 삭제 |

> AI는 통역만 수행하며, 본인확인·발급 등 행정 처리와 의사결정은 담당 공무원이 수행합니다.

---

## 👥 팀 오동가영

| 역할 | 담당 | 주요 업무 |
|------|------|----------|
| 🎯 **기획 · 팀장** | **윤가연** | 서비스 기획, 사용자 시나리오 설계, 프로젝트 총괄 |
| 🎨 **프론트엔드** | **최영수** | 키오스크·상담원 UI 구현, 수어 인식 실환경 검증 |
| ⚙️ **백엔드** | **김동완** | Flask 서버, Claude·STT·알림톡 연동, AES-256 암호화·보안 설계 |
| 🧠 **AI 모델** | **권오경** | 수어 데이터 전처리, CNN-GRU/BiGRU 모델 학습 및 고도화 |

---

## 📊 활용 데이터

- **AIHub 한국 수어 데이터셋** (한국지능정보사회진흥원, NIA) — WORD 3,000 / SENTENCE 2,000 클래스, 약 536,000 영상 클립
- **자체 촬영 수어 영상** 615개 — 창구 웹캠 환경 보강
- **공공데이터포털 중앙부처 복지서비스 API** (data.go.kr) — 청각장애인 복지 정보 안내

---

<p align="center">
  <b>🌷 피어나</b> — 말 대신 손끝으로, 농인의 표현이 민원센터 안에서 피어납니다.
</p>
