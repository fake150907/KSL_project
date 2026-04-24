# Web Demo 작업 히스토리

작성일: 2026-04-27  
프로젝트 루트: `c:\github\ai-project-01`

이 문서는 새 세션에서 바로 이어서 작업할 수 있도록, React/Flask 수어 인식 웹 데모의 현재 상태와 다음 확인 포인트를 정리한 인수인계 문서다.

## 현재 목표

- `web` 폴더의 React UX에서 실시간 카메라 영상을 사용해 수어 인식 결과를 시연한다.
- 내장 카메라 또는 Windows에서 인식되는 휴대폰/가상 카메라를 선택할 수 있게 한다.
- 카메라 영상 위에 MediaPipe 랜드마크를 그려서 손/팔 추적 상태를 확인한다.
- 현재 최고 정확도 모델인 CNN-GRU sequence 모델을 기본 모델로 사용한다.
- 실시간 오인식 원인을 보기 위해 1순위 예측뿐 아니라 top-3 후보를 화면과 Flask 로그에 남긴다.

## 실행 환경

- Conda 환경: `gesture-test`
- Python: `C:\Users\shemanul\anaconda3\envs\gesture-test\python.exe`
- GPU: NVIDIA GeForce RTX 5080
- PyTorch: `torch 2.11.0+cu128`
- CUDA 사용 가능 상태로 확인됨
- React/Vite 앱 위치: `web`
- Flask 백엔드: `backend/app.py`

## 주요 모델 상태

현재 웹 데모의 기본 sequence 모델은 아래 체크포인트를 사용한다.

- 활성 체크포인트: `outputs/checkpoints/sequence_model.pt`
- 모델 구조: CNN-GRU
- 검증 정확도: 89.44%
- best epoch: 80
- 백업 체크포인트: `outputs/checkpoints/sequence_model_cnn_gru_c64_h64_d03_80e_best.pt`
- 메트릭 파일: `outputs/reports/sequence_metrics.json`

비교 모델:

| 모델 | 검증 정확도 |
|---|---:|
| RandomForest baseline | 70.00% |
| GRU sequence | 85.56% |
| BiGRU + Attention | 86.11% |
| Transformer Encoder | 87.78% |
| CNN-GRU | 89.44% |

웹 UI 모델 선택에는 현재 다음처럼 표시된다.

- `RandomForest (70.00%)`
- `CNN-GRU (89.44%)`

## 백엔드 상태

파일: `backend/app.py`

주요 내용:

- Flask API 서버로 `/api/health`, `/api/predict` 제공
- `outputs/checkpoints/baseline.joblib`에서 baseline 모델 로드
- `outputs/checkpoints/sequence_model.pt`에서 sequence 모델 로드
- `src.models.model_sequence.build_sequence_model`을 사용해 CNN-GRU/GRU/BiGRU/Transformer 계열 체크포인트를 로드할 수 있게 되어 있음
- MediaPipe Holistic으로 pose/hand landmark 추출
- live frame을 `sequence_to_tensor`로 변환해 모델 추론
- 모델 입력 feature 수와 live tensor feature 수가 다를 경우 `align_tensor_features`로 맞춤
- 예측 결과에 `top_predictions`를 추가함
- Flask 로그에도 top-3 후보가 `top=` 형태로 출력되도록 함
- 서버 실행 설정:
  - host: `127.0.0.1`
  - port: `5000`
  - debug: `False`
  - reloader: `False`
  - threaded: `True`

백엔드 검증:

```powershell
C:\Users\shemanul\anaconda3\envs\gesture-test\python.exe -m py_compile backend/app.py
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5000/api/health
```

마지막 확인 당시 `/api/health`는 `{"status":"ok"}`로 정상 응답했다.

## 프론트엔드 상태

주요 파일:

- `web/src/components/SignLanguageStream.tsx`
- `web/src/styles/SignLanguageStream.css`

현재 구현된 기능:

- 카메라 시작/중지
- Windows에서 인식되는 카메라 목록 선택
- 휴대폰 카메라도 Iriun/DroidCam 등으로 Windows 카메라 장치로 잡히면 선택 가능
- 음성 인식 시작/중지
- 모델 선택: baseline / sequence
- 신뢰도 임계값 조정
- sequence window size 조정
- 안정화 최소 개수 조정
- 중복 방지 쿨다운 조정
- 수어 인식 결과를 대화창에 자동 추가
- 카메라 영상 위에 랜드마크 overlay 표시
- 예측 overlay에 top-3 후보 표시

랜드마크 색상:

- pose: 파란색 `#38bdf8`
- left hand: 초록색 `#22c55e`
- right hand: 빨간색 `#ef4444`

프론트 검증:

```powershell
cd web
npm.cmd run build
```

마지막 확인 당시 TypeScript/Vite production build가 성공했다.

## 실행 방법

백엔드 실행:

```powershell
$env:MPLCONFIGDIR='C:\github\ai-project-01\.matplotlib'
C:\Users\shemanul\anaconda3\envs\gesture-test\python.exe backend/app.py
```

프론트 개발 서버 실행:

```powershell
cd web
npm.cmd run dev -- --host 127.0.0.1
```

브라우저 접속:

```text
http://127.0.0.1:3001
```

API proxy 확인:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:3001/api/health
```

마지막 확인 당시 React proxy를 통한 `/api/health`도 정상 응답했다.

## 로그 위치

- Flask stdout: `outputs/logs/flask_web_demo.out.log`
- Flask stderr/access log: `outputs/logs/flask_web_demo.err.log`
- Vite stdout: `outputs/logs/vite_web_demo.out.log`
- Vite stderr: `outputs/logs/vite_web_demo.err.log`

예측 디버깅 시 볼 것:

- 화면 prediction overlay의 1순위 예측
- 화면 prediction overlay 아래의 top-3 후보
- Flask 로그의 `Prediction: ... top=[...]`

## 최근 관찰한 문제

사용자가 `감사하다` 동작을 했고, 손 랜드마크도 감사 동작처럼 잘 찍혔지만 인식 결과가 이상하다고 보고했다.

현재 판단:

- 랜드마크 검출 실패보다는 live 카메라 입력 분포와 AIHub 학습 데이터 분포가 다른 문제일 가능성이 크다.
- 학습 데이터는 AIHub keypoint clip이고, 실시간 데모는 MediaPipe로 직접 추출한 좌표를 사용한다.
- 거리, 카메라 각도, 좌우 반전, 동작 시작/끝 프레임, 속도, 손 위치가 달라지면 모델이 다른 라벨로 판단할 수 있다.
- 현재 모델은 isolated clip 기준으로 학습되었고, live continuous motion은 중간 동작/준비 동작/복귀 동작이 섞인다.
- 그래서 화면에 top-3 후보를 추가해 `감사`가 2~3순위라도 나오는지 확인하도록 했다.

## 2026-04-27 진단 및 1차 수정

YouTube 클립 2개를 받아 현재 CNN-GRU 모델로 평가했다.

- 감사: `https://youtu.be/T9FCgQp8V98?si=U-gPF9mdNsR_XbHh`
- 괜찮다: `https://youtube.com/shorts/HfqEWZ1ssQk?si=eYjiZHxBqM6eoE4c`

초기 평가에서는 `감사`, `괜찮다` 모두 거의 모든 window가 `병원`으로 예측됐다. 원인을 추적한 결과, AIHub 학습 데이터는 2D keypoint의 3번째 채널이 confidence인데 live/video MediaPipe 변환은 3번째 채널에 z-depth를 넣고 있었다.

수정 파일:

- `src/data/keypoint_utils.py`

수정 내용:

- pose landmark: `[x, y, visibility]`
- hand landmark: `[x, y, 1.0]`

수정 후 YouTube 재평가:

| 기대 라벨 | 방향 | 결과 |
|---|---|---|
| 감사 | 일반 | 47개 window 중 41개가 `감사`, 평균 1순위 `감사` 64.93% |
| 감사 | 좌우반전 | `배고프다`로 악화 |
| 괜찮다 | 일반 | 아직 `병원` 쏠림 남음 |
| 괜찮다 | 좌우반전 | `감사`로 오인식 |

결론:

- `병원` 전체 쏠림의 큰 원인 중 하나는 3번째 채널 불일치였다.
- `감사`는 이 수정으로 상당히 회복됐다.
- `괜찮다`는 별도 원인이 남아 있어 실제 수어 구간 crop, pose/hand 분포 비교, live 샘플 fine-tuning이 필요하다.

상세 보고서:

- `report/youtube_clip_eval.md`

## 사용자가 준 참고 영상

실시간 인식 튜닝 시 아래 영상을 참고해서 실제 동작 형태와 모델 예측을 비교한다.

- 감사: `https://youtu.be/T9FCgQp8V98?si=U-gPF9mdNsR_XbHh`
- 괜찮다: `https://youtube.com/shorts/HfqEWZ1ssQk?si=eYjiZHxBqM6eoE4c`

주의:

- 영상 링크는 동작 참고용이다.
- 저작권 문제 때문에 영상 내용을 그대로 저장하거나 복제하지 말고, 사용자가 직접 보며 동작 비교에만 사용한다.

## 다음 작업 제안

1. 웹 데모에서 `감사` 동작을 여러 번 수행하고 top-3 후보를 기록한다.
2. `감사`가 top-3 안에 자주 들어오면 후처리, 임계값, window size, 안정화 로직을 조정한다.
3. `감사`가 top-3에도 거의 안 들어오면 live 카메라 데이터와 AIHub 데이터 간 feature normalization 차이를 점검한다.
4. 좌우 반전 여부를 확인한다. 화면은 mirror 표시이지만 모델 입력까지 의도치 않게 좌우가 바뀌었는지 확인해야 한다.
5. `감사`, `괜찮다`처럼 헷갈리는 라벨을 골라 live 카메라 샘플을 소량 수집하고 fine-tuning 또는 calibration 세트를 만든다.
6. window 안에 준비/복귀 프레임이 많이 섞이지 않도록 “동작 시작 버튼” 또는 “최근 N프레임 중 motion peak 중심 crop” 방식을 검토한다.

## 새 세션에서 바로 확인할 명령

```powershell
cd c:\github\ai-project-01
C:\Users\shemanul\anaconda3\envs\gesture-test\python.exe -m py_compile backend/app.py
cd web
npm.cmd run build
```

서버가 떠 있는지 확인:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:5000/api/health
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:3001/api/health
```

## 주의할 점

- `report/streamlit_실행_메모.md_이건 작업하지마` 파일은 사용자가 작업하지 말라고 표시해둔 파일이다.
- AIHub API key 같은 민감한 값은 문서에 남기지 않는다.
- 현재 작업의 핵심은 모델 구조 변경보다 live demo 입력과 학습 데이터 입력의 차이를 줄이는 것이다.
- 정확도 개선 기록은 `report/정확도_history.md`에 계속 누적한다.
