# 새 PC 프로젝트 시작 가이드

**작성일**: 2026-04-26
**상태**: 이관 완료, 웹 앱 개발 시작 준비

---

## 🎯 현재 상황 요약

### ✅ 지난주 (이전 PC) 완료 사항

#### 1. 라벨 선정 (팀 논의 결과)
- **1723개** 전체 라벨 분석
- **876개** → 1차 생활형 구분
- **239개** → 2차 감정/상태/행동 중심 구분
- **8개** → MVP 최종 선택

**라벨 선정 기준** (감정/상태/행동만 포함):
```
✅ 감정 (Emotion): 기쁘다, 슬프다, 화나다, 안타깝다
✅ 상태 (State): 아프다, 배고프다, 졸리다, 피곤하다
✅ 행동 (Action): 먹다, 걷다, 자다, 때리다, 가다

❌ 제외: 사람, 사물, 장소, 시간, 추상, 기관
💡 핵심: 동사 + 형용사만 (명사는 제외)
```

**확대 전략** (팀 합의):
```
Phase 1: MVP 8개 모델 학습 + 웹 UI 구현
Phase 2: 웹 UI/데모 VISIBILITY 확인
         ✅ 웹 앱이 제대로 작동 + 실시간 인식 안정적 
            → 239개로 확대 GO!
         ❌ 웹 앱 불안정 또는 실시간 인식 어려움
            → 8개 유지/보완
```

#### 2. 데이터 확대 (REAL01~05)
- 이전: 300개 샘플 (REAL01만)
- 현재: **375개 샘플** (REAL01~05 모두) - 25% 증가
- 포함 파일:
  - `data/raw/selected_keypoints_01~05/` (5개 디렉토리)
  - `data/sample_subset_manifest.csv` (375행)
  - `data/processed/sign_word_subset.npz` (전처리 완료)

#### 3. 모델 재학습
- **Baseline**: `outputs/checkpoints/baseline.joblib`
  - Validation 정확도: 74.67% (이전 78.33%)
  - 감소 이유: 샘플 다양성 증가로 일반화 학습
- **Sequence (LSTM)**: `outputs/checkpoints/sequence_model.pt`
  - 재학습 완료

---

## 🚀 새 PC에서 해야 할 일 (우선순위 순)

### 1주차: 웹 앱 기초 + MVP 검증

#### [월] 이관 상태 검증
```powershell
# 1. 기본 파일 확인
ls data/sample_subset_manifest.csv
ls outputs/checkpoints/baseline.joblib
ls data/processed/sign_word_subset.npz

# 2. Manifest 검증
python scripts/validate_subset_manifest.py --manifest data/sample_subset_manifest.csv --check_paths

# 3. 모델 로드 테스트
python -c "import joblib; model = joblib.load('outputs/checkpoints/baseline.joblib'); print('Model loaded OK')"
```

**목표**: 모든 파일이 정상적으로 로드되는지 확인

#### [화~목] Flask 백엔드 구현
```powershell
# 1. Flask 패키지 설치
cd backend
C:\Users\Samsung\anaconda3\envs\gesture-test\python.exe -m pip install Flask Flask-CORS mediapipe numpy opencv-python

# 2. app.py 구현
# - MediaPipe Holistic 초기화
# - /api/predict 엔드포인트 구현
# - baseline/sequence 모델 로드
# - CORS 설정

# 3. Flask 테스트
C:\Users\Samsung\anaconda3\envs\gesture-test\python.exe app.py
# 접속: http://localhost:5000/api/health
```

**필수 구현**:
```python
@app.route('/api/predict', methods=['POST'])
def predict():
    # frame (base64 또는 이미지) 수신
    # → MediaPipe 랜드마크 추출
    # → 모델 입력 변환
    # → baseline/sequence 추론
    # → {label, confidence, timestamp} 반환
    pass
```

#### [금] React 연동 첫 테스트
```powershell
# 1. React 시작
cd web
npm install  # 최초 1회만
npm run dev
# 접속: http://localhost:3000

# 2. Flask API 호출 테스트
# - 프레임 캡처 → Flask로 전송 → 결과 수신
# - 신뢰도 필터링 (confidence >= 0.5)
# - 결과 표시 확인
```

**목표**: Flask + React 기본 연동 확인

---

### 2주차: 웹 앱 안정화 + UI 최적화

#### 웹 앱 성능 개선
- [ ] 프레임 해상도 최적화 (640x480 권장)
- [ ] 전송 주기 조정 (300ms 권장)
- [ ] 메모리 누수 확인
- [ ] 오류 처리 추가

#### UI/UX 개선
- [ ] 예측 신뢰도 시각화
- [ ] 오류 상황 처리
- [ ] 모바일 반응형 테스트

#### 결정: 웹 앱 vs Streamlit
```
웹 앱 80%+ 완성 → 웹 앱으로 발표
웹 앱 50~80% → 마무리 작업으로 충분
웹 앱 50% 미만 → Streamlit으로 전환
```

---

### 3주차: 발표 준비 (UI 선택 후)

#### 발표 자료 준비
- [ ] 슬라이드 작성 (프로젝트 개요, 기술 스택, 결과)
- [ ] 각 라벨별 샘플 영상 준비
- [ ] 시연 시나리오 작성
- [ ] Q&A 예상 질문 정리

#### 최종 테스트
- [ ] 웹 앱 또는 Streamlit 안정성 확인
- [ ] 각 라벨별 인식률 측정
- [ ] 실패 대비 백업 영상 준비
- [ ] 리허설

---

## 📋 핵심 파일 위치

| 파일 | 용도 |
|------|------|
| `report/새PC_프로젝트_이관_지시서.md` | 이관 상태 및 데이터 설명 |
| `data/sample_subset_manifest.csv` | 375개 샘플 목록 |
| `outputs/checkpoints/baseline.joblib` | 학습된 모델 |
| `backend/app.py` | Flask 백엔드 (구현 필요) |
| `web/src/` | React 프론트엔드 |
| `config/default.yaml` | 기본 설정 |

---

## 🔧 필수 환경 설정

### Conda 환경
```powershell
conda activate gesture-test  # 이미 생성됨
python --version  # Python 3.10.x 확인
```

### Flask 필수 패키지
```powershell
cd backend
pip install Flask Flask-CORS mediapipe numpy opencv-python joblib torch torchvision
```

### React 필수 패키지
```powershell
cd web
npm install  # node_modules 생성 (최초 1회)
npm run dev  # 개발 서버 시작
```

---

## 📊 현재 성능 기준

| 항목 | 현재값 | 목표값 |
|------|-------|-------|
| Baseline 정확도 | 74.67% | 80%+ (확대 후) |
| 샘플 수 | 375개 | 1000+ (전체 확대) |
| 실시간 FPS | 미측정 | 30+ |
| 예측 지연 | 미측정 | 300ms 이하 |
| 라벨 수 | 8개 | 20~239개 (선택사항) |

---

## ✅ 체크리스트 (이 주)

### [필수]
- [ ] Manifest 및 모델 검증 완료
- [ ] Flask /api/predict 엔드포인트 구현
- [ ] React + Flask 기본 연동
- [ ] **웹 UI/데모 VISIBILITY 확인** (4월 30일)
  - 안정적이면 → 239개 확대 GO!
  - 불안정하면 → 8개 유지로 진행

### [선택]
- [ ] 라벨 239개 확대 준비 (VISIBILITY 확인 후)
- [ ] 팀원 참여 일정 잡기 (239개 확대 시)

---

## 💬 기억할 핵심

1. **라벨 확대는 선택사항**
   - 8개에서 충분한 성과 확인 후 결정
   - 무조건 239개까지 갈 필요 없음

2. **웹 앱이 이상적이지만 Streamlit도 좋음**
   - 웹 앱: 전문성 강조, 실제 제품처럼 보임
   - Streamlit: 빠른 구현, 신뢰할 수 있는 기술 시연
   - 둘 중 더 완성된 것으로 발표

3. **실시간 성능이 핵심**
   - 300ms 이하 지연 필수
   - 신뢰도 필터링 (confidence >= 0.5)
   - 8~10개 라벨은 안정적 인식 필수

---

**다음 단계**: Flask 백엔드 `/api/predict` 엔드포인트 구현 시작
**예상 소요 시간**: 2~3일
**발표 예정일**: (미정 - 팀 공지 대기)
