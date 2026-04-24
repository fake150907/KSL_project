# Sign Language Interpreter - React Web UI

청각장애인과 일반인 간의 실시간 대화를 위한 수어 인식 웹 애플리케이션입니다.

## 구조

- **web/**: React 프론트엔드 (Vite + TypeScript)
- **backend/**: Flask 백엔드 (Python 모델 추론)

## 설치

### 프론트엔드 (React)

```bash
cd web
npm install
npm run dev
```

- 접속: http://localhost:3000

### 백엔드 (Flask)

```bash
# gesture-test 환경 사용
C:\Users\Samsung\anaconda3\envs\gesture-test\python.exe -m pip install -r backend/requirements.txt

# Flask 앱 실행
C:\Users\Samsung\anaconda3\envs\gesture-test\python.exe backend/app.py
```

- 접속: http://localhost:5000/api/health

## 동작 흐름

1. **React (localhost:3000)**
   - 카메라 스트림 받기
   - 프레임을 Flask로 전송 (300ms 간격)

2. **Flask (localhost:5000)**
   - 프레임 수신
   - MediaPipe + ML 모델로 추론
   - 예측 결과 반환 (라벨, 신뢰도)

3. **React UI 업데이트**
   - 카메라 오버레이에 예측 표시
   - 텍스트 박스에 자동 입력

## 현재 상태

- ✓ React 프로젝트 기본 구조
- ✓ 카메라 스트림 UI
- ✓ 청각장애인 / 일반인 레이아웃
- ⏳ Flask 백엔드 (더미 예측만 구현)
- ⏳ MediaPipe + ML 모델 통합
- ⏳ 실시간 추론 로직

## 다음 단계

1. Flask에 MediaPipe 손 인식 추가
2. 기존 ML 모델 (baseline.joblib, sequence_model.pt) 통합
3. 예측 결과 리팩토링 (신뢰도 기준 필터링)
4. 메모리/성능 최적화
