# 양방향 수어 통역 보조 시스템 - 전체 TODO 리스트 (완료 상태 포함)

**최종 업데이트**: 2026-04-26
**작성자**: 팀장 (Claude Code)

---

## 📌 범례

| 기호 | 의미 |
|------|------|
| ✅ | 완료 |
| ⏳ | 진행 중 |
| ❌ | 미완료 (필수) |
| 💡 | 미완료 (선택/심화) |

---

## Phase 1: 프로젝트 설정 및 환경 준비

### 환경 구축
- [x] ✅ Python 3.10 기반 conda 환경 생성 (gesture-test)
- [x] ✅ requirements.txt 패키지 설치
- [x] ✅ 추가 패키지 설치 (scikit-learn, joblib, matplotlib, pandas, numpy, PyYAML, streamlit)
- [x] ✅ AIHub API Key 설정 (환경변수: AIHUB_APIKEY)
- [x] ✅ 프로젝트 경로 통일 (C:\github\ai-project-01)

### 데이터 디렉토리 설정
- [x] ✅ data/raw/ 디렉토리 구조 확인
  - [x] ✅ data/raw/Training/REAL/WORD/real_word_keypoint/
  - [x] ✅ data/raw/Training/REAL/WORD/real_word_morpheme/
  - [x] ✅ data/raw/Validation/REAL/WORD/real_word_keypoint/
  - [x] ✅ data/raw/Validation/REAL/WORD/real_word_morpheme/
- [x] ✅ data/processed/ 디렉토리 생성
- [x] ✅ data/raw/selected_keypoints_01/ 디렉토리 생성

### 설정 파일
- [x] ✅ config/default.yaml 기본 설정 검증
- [x] ✅ 시퀀스 길이 설정: 32프레임
- [x] ✅ Keypoint feature 설정: pose + left hand + right hand (얼굴 제외)

---

## Phase 2: 데이터 분석 및 라벨 선정

### AIHub 데이터 다운로드
- [x] ✅ 01_real_word_morpheme.zip 다운로드 및 추출
  - 파일: data/raw/Training/REAL/WORD/real_word_morpheme/
  - 샘플 수: 9,079개
- [x] ✅ 01_real_word_keypoint.zip 다운로드 및 추출
  - 파일: data/raw/Training/REAL/WORD/real_word_keypoint/
  - 크기: 11GB (압축), 약 44GB (해제 후)
- ❌ 02~16번 keypoint zip 다운로드 (팀원 역할)
  - 예상 크기: 174GB (압축), 520~700GB (해제 후)

### 라벨 분석
- [x] ✅ Extract_labels.py 실행으로 고유 라벨 추출
  - 고유 라벨: 1,723개 확인
  - 샘플 수: 9,079개
- [x] ✅ label_candidates.csv 생성
- [x] ✅ 생활형 8개 라벨 선택 완료
  - 가다, 감사, 괜찮다, 배고프다, 병원, 아프다, 우유, 자다
- [x] ✅ selected_labels_small.json 생성
- [x] ✅ selected_lifestyle_candidates.csv 생성

### Target Sample 추출
- [x] ✅ export_selected_targets.py 실행
  - 생성 파일: data/selected_label_targets.csv
  - 총 샘플: 75개
    - 가다: 20개
    - 감사: 5개
    - 괜찮다: 5개
    - 배고프다: 10개
    - 병원: 5개
    - 아프다: 5개
    - 우유: 10개
    - 자다: 15개
- [x] ✅ Target 매칭 기준 확인
  - WORD번호 + 카메라각도(F/L/R/U/D) 기준
  - REAL번호는 무시 (같은 WORD+각도면 같은 동작으로 간주)

---

## Phase 3: Keypoint Subset 추출 및 전처리

### 01번 Keypoint 처리 (완료)
- [x] ✅ extract_keypoint_subset_from_zip.py 실행 (01번)
  - 입력: 01_real_word_keypoint.zip
  - 출력: data/raw/selected_keypoints_01/
  - Manifest: data/sample_subset_manifest_01.csv
- [x] ✅ validate_subset_manifest.py로 검증
  - 결과: [OK] Manifest validation passed.
  - 샘플 수: 60개 (target 75개 중 실제 확보)
- [x] ✅ subset manifest 열 확인
  - sample_id, label, angle, morpheme_path, start, end, duration, split, keypoint_path, is_dummy

### 02~16번 Keypoint 처리 (진행 중)
- ⏳ 팀원 A: 02, 03, 04번 처리 중
- ⏳ 팀원 B: 05, 06, 07번 처리 중
- ⏳ 팀원 C: 08, 09, 10번 처리 중
- ⏳ 팀원 D: 11, 12, 13번 처리 중
- ⏳ 팀원 E: 14, 15, 16번 처리 중

**필수 명령어**:
```powershell
python -m src.data.extract_keypoint_subset_from_zip `
  --zip_path "data/raw/aihub_downloads/word_keypoint_XX/004.수어영상/1.Training/라벨링데이터/REAL/WORD/XX_real_word_keypoint.zip" `
  --output_dir "data/raw/worker_X_selected_keypoints_XX" `
  --manifest_output "data/worker_X_sample_subset_manifest_XX.csv"

python scripts/validate_subset_manifest.py --manifest data/worker_X_sample_subset_manifest_XX.csv
```

### Manifest 병합 (미완료)
- ❌ 팀원별 manifest 수집 대기
  - 팀원 A: worker_A_selected_keypoints/, worker_A_sample_subset_manifest_merged.csv
  - 팀원 B: worker_B_selected_keypoints/, worker_B_sample_subset_manifest_merged.csv
  - 팀원 C: worker_C_selected_keypoints/, worker_C_sample_subset_manifest_merged.csv
  - 팀원 D: worker_D_selected_keypoints/, worker_D_sample_subset_manifest_merged.csv
  - 팀원 E: worker_E_selected_keypoints/, worker_E_sample_subset_manifest_merged.csv
- ❌ merge_subset_manifests.py로 최종 병합
  ```powershell
  python scripts/merge_subset_manifests.py `
    --inputs data/worker_A_sample_subset_manifest_merged.csv `
             data/worker_B_sample_subset_manifest_merged.csv `
             ... `
    --output data/sample_subset_manifest_final.csv
  ```
- ❌ 중복 sample_id 제거 확인
- ❌ 최종 validate_subset_manifest.py 검증

### Keypoint 전처리 (부분 완료)
- [x] ✅ preprocess_keypoints.py 초기 실행 완료 (01번 기준)
  - 생성 파일: data/processed/sign_word_subset.npz
  - Shape: (300, 32, 201)
  - Train: 240, Validation: 60
- ❌ 확장 데이터 기반 전처리 재실행
  - 확장 manifest 기준으로 재생성
  - Shape 및 샘플 수 확인
  ```powershell
  python -m src.data.preprocess_keypoints --config config/default.yaml --subset_only
  ```

---

## Phase 4: 모델 학습 및 평가

### Baseline 모델 (RandomForest)
- [x] ✅ baseline 모델 초기 학습 완료
  - Checkpoint: outputs/checkpoints/baseline.joblib
  - Validation 정확도: 78.33%
  - 샘플 수: 300개 기준
- [x] ✅ baseline_metrics.json 생성
- [x] ✅ confusion_matrix.png 생성
- ❌ 확장 데이터 기반 재학습
  ```powershell
  python -m src.models.train_baseline --config config/default.yaml
  ```
- ❌ 새 모델과 기존 모델 성능 비교 분석

### Sequence 모델 (LSTM/GRU)
- [x] ✅ LSTM 모델 초기 학습 완료
  - Checkpoint: outputs/checkpoints/sequence_model.pt
  - 기본 학습 경로 확인
- [x] ✅ sequence_metrics.json 생성
- ❌ 확장 데이터 기반 재학습
  ```powershell
  python -m src.models.train_sequence --config config/default.yaml --epochs 10 --batch_size 16
  ```
- ❌ GRU 모델도 학습 (비교용)
- ❌ Baseline vs Sequence vs GRU 성능 비교

### 모델 평가
- [x] ✅ 기본 성능 지표 수집 (01번 기준)
- ❌ 확장 데이터 기반 모든 모델 재평가
- ❌ Confusion matrix 최신화
- ❌ 클래스별 성능 분석
  - 각 라벨별 정확도
  - 가장 성능 높은/낮은 라벨 파악
- ❌ 성능 보고서 (performance_summary.md) 최종 작성

---

## Phase 5: 실시간 추론 및 데모

### CLI 기반 실시간 추론
- [x] ✅ realtime_sign_inference.py 기본 경로 확인
- [x] ✅ MediaPipe Holistic 랜드마크 추출 동작
- [x] ✅ 웹캠 입력 기본 처리
- ❌ 실시간 인식 안정성 개선
  - 프레임 수집 안정화
  - 윈도우 크기 최적화
  - Frame stride 조정
- ❌ 실시간 예측 신뢰도 필터링
  - 낮은 신뢰도 예측 제외
  - 반복 예측 안정화
- ❌ CLI 실행 명령어 확인
  ```powershell
  python -m src.inference.realtime_sign_inference --model baseline --speak
  ```

### Streamlit UI (부분 완료)
- [x] ✅ Streamlit UI 기본 구현 (app.py)
- [x] ✅ 웹캠 실시간 카메라 입력
- [x] ✅ 예측 라벨 오버레이 표시 (불완전)
- [x] ✅ 수어 텍스트 박스
- [x] ✅ 음성 텍스트 박스
- [x] ✅ STT/TTS 버튼 UI
- [x] ✅ 오프라인 샘플 점검 패널
- ❌ **실시간 결과 자동 반영 안정화 (핵심 이슈)**
  - 현재: 카메라 오버레이 예측 일부 동작, 텍스트 자동입력 불안정
  - 필요한 작업:
    - Frame 버퍼링 로직 개선
    - Streamlit session state 관리
    - 자동 입력 트리거 방식 재설계
    - 휴대폰 웹캠 프레임 드롭 처리
- ❌ 발표용 시연 시나리오 작성
  - 각 라벨별 샘플 영상 준비
  - 실시간 인식 데모 흐름도 작성

**Streamlit 실행 명령**:
```powershell
# 로컬 접속
python -m streamlit run src/ui/app.py

# 휴대폰 접속 (권장)
C:\Users\Samsung\anaconda3\envs\gesture-test\python.exe -m streamlit run src/ui/app.py `
  --server.address=0.0.0.0 --server.port=8501
```

**접속 주소**:
- PC: http://localhost:8501
- 휴대폰: http://192.168.0.108:8501

---

## Phase 6: React + Flask 웹 UI (추가 개발)

### 현재 상태
- [x] ✅ React 프로젝트 기본 구조 완성
  - Package: web/, node_modules 제외 (이관 시)
- [x] ✅ 카메라 스트림 UI 구현
- [x] ✅ 청각장애인 / 일반인 레이아웃
- ⏳ Flask 백엔드 (더미 예측만 구현됨)
- ❌ **MediaPipe + ML 모델 통합 (진행 필요)**

### Flask 백엔드 개발 (미완료)
- ❌ Flask 앱 설정 (app.py 확인 및 개선)
- ❌ MediaPipe Holistic 통합
  - 프레임 수신 → landmark 추출 → 모델 추론
- ❌ 기존 ML 모델 로드 (baseline.joblib, sequence_model.pt)
- ❌ /api/predict 엔드포인트 구현
  - 입력: 프레임 (이미지 바이너리)
  - 출력: {label, confidence, timestamp}
- ❌ /api/health 엔드포인트 확인
- ❌ 메모리/성능 최적화
  - 프레임 해상도 조정
  - 배치 처리 vs 실시간 처리

**Flask 실행**:
```powershell
cd backend
C:\Users\Samsung\anaconda3\envs\gesture-test\python.exe -m pip install -r requirements.txt
C:\Users\Samsung\anaconda3\envs\gesture-test\python.exe app.py
```

### React 프론트엔드 개선
- ❌ Flask 연동 테스트 (HTTP POST /api/predict)
- ❌ 예측 결과 신뢰도 기준 필터링
  - 낮은 신뢰도(< 0.5) 제외
  - 신뢰도 시각화 (프로그레스바/게이지)
- ❌ 결과 텍스트 자동 입력
  - 수어 → 텍스트 (Flask 예측)
  - 음성 → 텍스트 (STT, 아직 미구현)
- ❌ TTS 버튼 연동 (별도 API 필요)
- ❌ 모바일 반응형 UI 개선

---

## Phase 7: STT/TTS 서비스 (선택)

### STT (Speech-to-Text)
- ❌ Google Cloud Speech-to-Text 연동
  - Credentials 설정 (서비스 계정 JSON)
  - 음성 입력 수집
  - 비동기 처리 및 결과 반환
- 💡 (선택) 로컬 STT 대체 (예: Vosk, PocketSphinx)
  - 네트워크 독립성
  - 낮은 지연시간

### TTS (Text-to-Speech)
- ❌ Google Cloud Text-to-Speech 연동
  - 한국어 음성 생성
  - 음성 파일 저장 또는 스트리밍
- 💡 (선택) 로컬 TTS 대체 (예: pyttsx3, gTTS)
  - 오프라인 동작

### 서비스 통합
- ❌ stt_service.py 완성
- ❌ tts_service.py 완성
- ❌ UI에 STT/TTS 버튼 연동
- ❌ 오류 처리 및 폴백

---

## Phase 8: 최적화 및 최종 테스트

### 성능 최적화
- ❌ 모델 추론 속도 측정 (FPS 확인)
  - 목표: 실시간 처리 (30FPS 이상)
- ❌ 메모리 사용량 최적화
  - 대용량 manifest 처리
  - Keypoint 로딩 효율화
- ❌ 네트워크 처리 최적화 (Flask)
  - 프레임 압축 (JPEG)
  - 배치 처리

### 오류 처리
- ❌ 웹캠 오류 처리
- ❌ 모델 로드 실패 처리
- ❌ 네트워크 연결 오류 처리
- ❌ API 타임아웃 처리

### 최종 테스트
- ❌ 로컬 PC에서 전체 end-to-end 테스트
  - 웹캠 입력 → 예측 → 결과 표시
- ❌ 휴대폰 웹캠 (비디오 재생) 테스트
  - 네트워크 안정성 확인
  - 프레임 드롭 상황 확인
- ❌ 각 라벨별 인식 성공률 테스트
- ❌ 크로스 플랫폼 테스트 (다양한 브라우저)

---

## Phase 9: 발표 준비

### 발표 범위 확정
- ❌ **핵심 시연 항목 명확히**
  - ✅ 수화 인식 (8개 라벨)
  - ✅ 텍스트 변환
  - ✅ STT (음성 입력 → 텍스트)
  - ❓ TTS (텍스트 → 음성) - 시간 남으면
- ❌ **제외 항목 명확히**
  - ❌ 전체 한국수어 번역 (범위 초과)
  - ❌ 문장 단위 번역
  - ❌ 모바일 네이티브 앱

### 발표 자료 준비
- ❌ 프로젝트 개요 슬라이드
  - 배경: 청각장애인 소통 장벽
  - 목표: 양방향 통역 보조 시스템
  - 범위: MVP (8개 라벨)
- ❌ 기술 스택 소개
  - Python, MediaPipe, OpenCV
  - Flask, React
  - Google Cloud STT/TTS (또는 로컬 대체)
- ❌ 데이터 파이프라인 설명
  - AIHub 데이터 → Keypoint 추출 → 모델 학습
  - 샘플 수: 300+ (확장 후)
- ❌ 모델 성능 결과
  - Baseline 정확도: 78.33% → (재학습 후 업데이트)
  - Confusion matrix 시각화
  - 각 라벨별 성능
- ❌ 실시간 데모 흐름도
  - 웹캠 → MediaPipe → 모델 → 결과 표시
- ❌ 발표 시연 시나리오
  - 각 라벨별 샘플 영상 구성
  - 실제 웹캠 데모 순서
  - 트러블 슈팅 계획 (유튜브 영상 백업 등)

### 발표 영상/데모 준비
- ❌ 각 라벨별 샘플 수어 영상 준비
  - 가다, 감사, 괜찮다, 배고프다, 병원, 아프다, 우유, 자다
- ❌ 실시간 인식 데모 리허설
  - 적정 거리/조명/배경 확인
  - 성공률 측정
- ❌ 실패 대비 백업 영상 준비
- ❌ 최종 슬라이드 및 발표 대본 작성

### 팀원 교육
- ❌ 모든 팀원에게 최종 시스템 설명
- ❌ 시연 담당자 정하기
- ❌ 시연 순서 및 타이밍 연습
- ❌ Q&A 예상 질문 정리

---

## 🔍 알려진 문제점 (Known Issues)

### 실시간 인식 (우선순위: 높음)
| 문제 | 현재 상태 | 원인 | 해결 방안 |
|------|---------|------|---------|
| 텍스트 자동 입력 불안정 | 재현 가능 | Streamlit session state 처리 미흡 | UI 재설계 또는 AJAX 기반 전환 |
| 휴대폰 카메라 프레임 드롭 | 재현 가능 | 네트워크 지연 + 화면 갱신 | 프레임 skip 로직 추가 |
| 낮은 신뢰도 예측 노이즈 | 재현 가능 | 모델 신뢰도 필터링 없음 | confidence >= 0.5 필터 추가 |

### 데이터 처리 (우선순위: 높음)
| 문제 | 현재 상태 | 원인 | 해결 방안 |
|------|---------|------|---------|
| 02~16번 keypoint 매칭 실패 | 미테스트 | REAL 번호 다름 | WORD번호 + 각도 기준 매칭 필수 |
| 라벨 불균형 (가다: 20, 감사: 5) | 확인됨 | 원본 데이터 불균형 | 데이터 증강 또는 가중치 조정 |

### 모델 성능 (우선순위: 중간)
| 문제 | 현재 상태 | 원인 | 해결 방안 |
|------|---------|------|---------|
| 78.33% 정확도 (낮음) | 확인됨 | 샘플 수 부족 + 클래스 불균형 | 데이터 확장 + 모델 재학습 |
| Sequence 모델 vs Baseline | 미비교 | 성능 비교 미실시 | 확장 데이터 기준 비교 필요 |

### Web UI (우선순위: 낮음)
| 문제 | 현재 상태 | 원인 | 해결 방안 |
|------|---------|------|---------|
| Flask 백엔드 더미 예측 | 진행 중 | 모델 통합 미완료 | MediaPipe + ML 모델 연동 |
| React 컴포넌트 최적화 | 기본 구현 | 성능 테스트 미실시 | 렌더링 최적화 (useMemo 등) |

---

## ⚠️ 중요 확인 사항

### 데이터 매칭 규칙 (필수!)
```
❌ 잘못된 방식 (완전일치 비교):
  NIA_SL_WORD0943_REAL01_F (morpheme)
  NIA_SL_WORD0943_REAL02_F (keypoint)
  → "REAL 번호 다르니까 다른 데이터" (X)

✅ 올바른 방식 (WORD번호 + 각도):
  WORD0943_F (같음)
  → "같은 단어, 같은 각도니까 같은 동작" (O)
  → REAL01 morpheme의 label, start, end 사용
```

### 환경 변수 확인
```powershell
# 반드시 gesture-test 환경 사용
conda activate gesture-test

# API Key 설정
$env:AIHUB_APIKEY="YOUR_KEY_HERE"

# Python 경로 확인
python --version  # Python 3.10.x 여야 함
```

### 저장공간 확인
```powershell
# 디스크 여유 공간 확인 (최소 100GB 이상)
Get-Volume

# 필요한 디렉토리 용량
# data/raw/aihub_downloads/: 195GB (압축)
# data/raw/selected_keypoints_*/: 600~800GB (해제 후)
```

---

## 📊 주간 진행 상황 체크리스트

### 1주차 (데이터 다운로드 및 추출)
- ⏳ 팀원 A~E: keypoint zip 다운로드 중
- ⏳ 팀원 A~E: subset 추출 및 manifest 생성 중
- ⏳ 팀원 A~E: 검증 완료 (각 3개 zip)

**완료 기준**: 모든 팀원이 worker_X_sample_subset_manifest_merged.csv 생성 및 검증 통과

### 2주차 (병합 및 모델 재학습)
- ❌ 팀장: 최종 manifest 병합
- ❌ 팀장: 전처리 재실행
- ❌ 팀장: Baseline 재학습
- ❌ 팀장: Sequence 재학습
- ❌ 팀장: 성능 비교 분석

**완료 기준**: 
- 최종 manifest 검증 통과
- 새 모델 checkpoint 생성
- performance_summary.md 작성

### 3주차 (최적화 및 발표 준비)
- ❌ 팀장: Streamlit UI 안정화
- ❌ 팀장: Flask + React 통합 (optional)
- ❌ 팀장: 발표 자료 준비
- ❌ 팀장: 최종 리허설
- ❌ 팀장: 발표 실행

**완료 기준**:
- 실시간 인식 안정적 동작
- 발표 슬라이드 완성
- 데모 성공률 80% 이상

---

## 🎯 Success Metrics

### 데이터 확장
- [ ] 총 샘플 수: 300개 → 1,000개 이상 (목표)
- [ ] 라벨별 최소 샘플: 5개 → 30개 이상
- [ ] 클래스 불균형 개선: Gini 계수 감소

### 모델 성능
- [ ] Baseline 정확도: 78.33% → 85% 이상 (목표)
- [ ] Validation 정확도: 안정적 추이 (감소 추세 X)
- [ ] Inference 속도: 100ms 이하

### 실시간 인식
- [ ] 웹캠 FPS: 30FPS 이상 (안정적)
- [ ] 예측 반영 지연: 500ms 이하
- [ ] 신뢰도 필터링 후 오음성: 10% 이하

### 발표
- [ ] 데모 성공률: 80% 이상
- [ ] 8개 라벨 모두 시연 가능
- [ ] 청중 이해도: 설문 평균 4.0/5.0 이상

---

## 📝 문서 참고

| 문서 | 용도 |
|------|------|
| 새PC_프로젝트_이관_지시서.md | 새 환경 설정 및 초기 검증 |
| 작업자_역할분담_가이드.md | 팀원별 데이터 처리 방법 |
| 다음작업_TODO.md | 우선순위별 작업 항목 |
| 작업절차.md | 진행된 작업 흐름 (참고용) |
| 기획서_양방향 수어 통역 보조 시스템.md | 프로젝트 전체 범위 |
| 프로젝트_로드맵_및_팀원분담.md | 이번 분기 로드맵 및 팀원 분담 |
| 전체_TODO_리스트_완료상태.md | **이 문서** |

---

**마지막 업데이트**: 2026-04-26
**다음 검토**: 매주 금요일 오후 5시
**팀장**: (프로젝트 리더)
