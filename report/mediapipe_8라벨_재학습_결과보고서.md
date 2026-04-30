# MediaPipe 8라벨 재학습 결과 보고서

작성일: 2026-04-27

## 1. 작업 목적

이번 작업의 목적은 기존 AIHub OpenPose 계열 keypoint 학습 데이터와 웹캠 MediaPipe 입력 사이의 좌표계 불일치를 줄이는 것이다.

기존 모델은 AIHub가 제공한 OpenPose 계열 keypoint를 학습했고, React/Flask 웹캠 시연은 MediaPipe Holistic으로 실시간 landmark를 추출한다. 이 차이 때문에 같은 수어 동작도 모델 입장에서는 다른 입력 분포로 보일 수 있다.

따라서 이번에는 AIHub 원본 수어 동영상을 다시 MediaPipe Holistic으로 전처리하고, 그 결과로 sequence 모델을 재학습했다.

## 2. 대상 라벨

전체 AIHub 라벨을 사용하지 않고, 아래 8개 라벨만 사용했다.

| 라벨 | 샘플 수 |
|---|---:|
| 가다 | 20 |
| 감사 | 5 |
| 괜찮다 | 5 |
| 배고프다 | 10 |
| 병원 | 5 |
| 아프다 | 5 |
| 우유 | 10 |
| 자다 | 15 |

총 샘플 수는 75개다.

## 3. 데이터 준비

AIHub 원본 WORD 동영상 archive에서 필요한 mp4만 추출했다.

사용한 원천 영상 archive:

| archive | AIHub file key | 이번 작업에서 사용한 샘플 |
|---|---:|---:|
| `01_real_word_video.zip` | `39546` | 5 |
| `02_real_word_video.zip` | `39547` | 70 |

추출된 원본 영상 위치:

```text
data/raw/videos/
```

전처리용 manifest:

```text
data/mediapipe_video_manifest.csv
```

manifest 컬럼:

```csv
sample_id,label,video_path,start,end,duration,split
```

split 구성:

| split | 샘플 수 |
|---|---:|
| train | 60 |
| validation | 15 |

## 4. MediaPipe 전처리

실행한 전처리 명령:

```powershell
C:\Users\fake1\anaconda3\envs\gesture-test\python.exe -m src.data.preprocess_mediapipe_videos `
  --config config/mediapipe.yaml `
  --manifest data/mediapipe_video_manifest.csv `
  --output data/processed/sign_word_mediapipe_subset.npz `
  --sequence_length 32 `
  --frame_step 1
```

생성된 산출물:

| 파일 | 내용 |
|---|---|
| `data/processed/sign_word_mediapipe_subset.npz` | 학습용 MediaPipe tensor |
| `data/processed/sign_word_mediapipe_subset.meta.json` | 전처리 metadata |

전처리 결과:

| 항목 | 값 |
|---|---:|
| shape | `(75, 32, 201)` |
| sequence length | 32 |
| feature count | 201 |
| failed rows | 0 |

feature layout:

```text
MediaPipe fixed slots: pose25 + left_hand21 + right_hand21, each [x, y, confidence]
```

즉 각 frame은 다음 구조다.

```text
pose 25 points * 3 = 75
left hand 21 points * 3 = 63
right hand 21 points * 3 = 63
total = 201
```

## 5. 학습 결과

모델:

```text
CNN-GRU sequence model
```

학습 명령:

```powershell
C:\Users\fake1\anaconda3\envs\gesture-test\python.exe -m src.models.train_sequence `
  --config config/mediapipe.yaml `
  --model_type cnn_gru `
  --epochs 80 `
  --batch_size 64
```

학습 환경:

| 항목 | 값 |
|---|---|
| Python | 3.10.20 |
| device | CPU |
| MediaPipe | 0.10.13 |
| OpenCV | 4.11.0 |
| NumPy | 1.26.4 |

최종 학습 산출물:

| 파일 | 내용 |
|---|---|
| `outputs/checkpoints/sequence_model.pt` | 새 MediaPipe 재학습 checkpoint |
| `outputs/reports/sequence_metrics.json` | 학습 metric |

기존 checkpoint는 아래에 백업했다.

```text
outputs/checkpoints/backup/sequence_model_before_mediapipe_20260427_151658.pt
outputs/reports/backup/sequence_metrics_before_mediapipe_20260427_151658.json
```

학습 결과:

| 항목 | 값 |
|---|---:|
| best epoch | 42 |
| best validation accuracy | 73.33% |
| final reported accuracy | 73.33% |

## 6. 라벨별 검증 결과

validation set은 15개로 작기 때문에 아래 수치는 참고용으로 봐야 한다.

| 라벨 | precision | recall | f1-score | support |
|---|---:|---:|---:|---:|
| 가다 | 0.800 | 1.000 | 0.889 | 4 |
| 감사 | 1.000 | 1.000 | 1.000 | 1 |
| 괜찮다 | 0.000 | 0.000 | 0.000 | 1 |
| 배고프다 | 0.667 | 1.000 | 0.800 | 2 |
| 병원 | 0.000 | 0.000 | 0.000 | 1 |
| 아프다 | 0.000 | 0.000 | 0.000 | 1 |
| 우유 | 0.500 | 0.500 | 0.500 | 2 |
| 자다 | 0.750 | 1.000 | 0.857 | 3 |

요약:

| 평균 | precision | recall | f1-score |
|---|---:|---:|---:|
| macro avg | 0.465 | 0.563 | 0.506 |
| weighted avg | 0.586 | 0.733 | 0.648 |

## 7. 웹 시연 상태

React/Vite와 Flask backend는 실행 확인했다.

| 항목 | 상태 |
|---|---|
| Flask backend | 정상 실행 |
| React/Vite frontend | 정상 실행 |
| `/api/health` | 정상 응답 |
| sequence model load | 정상 |
| baseline model | `baseline.joblib` 없음 |

접속 주소:

```text
http://127.0.0.1:3001
```

백엔드 health:

```text
http://127.0.0.1:5000/api/health
```

프론트 proxy health:

```text
http://127.0.0.1:3001/api/health
```

추가로, Vite 앱에 `index.html`이 빠져 있어 production build가 실패했으므로 아래 파일을 새로 추가했다.

```text
web/index.html
```

이후 `npm run build`는 성공했다.

## 8. 웹캠 테스트 관찰

Flask 로그 기준으로 손 검출 자체는 간헐적으로 들어왔다.

다만 로그에서는 `window=32/32`까지 안정적으로 채워진 예측보다, 손 검출이 끊기면서 window가 초기화되는 흐름이 많이 보였다.

대표 패턴:

```text
has_hand=True, window=1/32
has_hand=False, miss=...
window reset
```

이 상태에서는 모델 정확도보다 먼저 웹캠 입력 안정성이 결과에 큰 영향을 준다. 즉 손이 계속 검출되어 32프레임 window가 충분히 채워져야 sequence 모델이 의미 있는 예측을 낼 수 있다.

## 9. 해석

이번 실험은 학습 데이터와 웹캠 입력을 모두 MediaPipe 기준으로 맞췄다는 점에서 중요한 진전이다.

좋아진 점:

- OpenPose keypoint 기반 학습과 MediaPipe 웹캠 입력 사이의 구조적 불일치를 줄였다.
- MediaPipe 전처리 산출물이 기존 학습 코드와 호환되는 `(75, 32, 201)` NPZ로 생성됐다.
- 새 sequence checkpoint가 Flask backend에서 정상 로드됐다.
- 웹 프론트 빌드와 proxy health까지 확인했다.

아직 조심해야 할 점:

- validation 샘플이 15개뿐이라 73.33% 정확도는 안정적인 일반화 성능으로 보기 어렵다.
- `감사`, `괜찮다`, `병원`, `아프다`처럼 support가 1개인 라벨은 라벨별 점수가 크게 흔들릴 수 있다.
- 실제 웹캠에서는 32프레임 window가 안정적으로 채워지는지 먼저 확인해야 한다.
- 현재 PC 학습은 CPU로 진행했다.

## 10. 다음 작업 제안

1. 웹캠에서 손이 계속 잡히도록 카메라 거리와 각도를 조정한다.
2. 화면 overlay에서 hand landmark가 끊기지 않는지 확인한다.
3. 같은 동작을 할 때 `window=32/32`가 자주 도달하는지 Flask 로그로 확인한다.
4. 실제 예측이 나오면 top-3 후보를 라벨별로 기록한다.
5. 특히 `괜찮다`, `병원`, `아프다`는 추가 샘플 확보가 필요하다.
6. 데이터가 더 확보되면 8개 라벨별 최소 샘플 수를 균형 있게 맞춘 뒤 재학습한다.

## 10-1. label8_clip 추가 평가

`data/raw/label8_clip`에 있는 13개 mp4를 현재 MediaPipe 재학습 checkpoint로 추가 평가했다.

실행 스크립트:

```text
scripts/evaluate_local_mediapipe_videos.py
```

결과 파일:

```text
outputs/local_label8_mediapipe_eval/local_label8_mediapipe_eval_summary.csv
outputs/local_label8_mediapipe_eval/local_label8_mediapipe_eval.json
```

평가 결과:

| 지표 | 결과 |
|---|---:|
| final accuracy | 0.00% |
| average accuracy | 0.00% |
| majority accuracy | 0.00% |

이 결과는 1안 자체가 무의미하다는 뜻이 아니라, AIHub validation과 별도 클립 사이의 도메인 차이가 아직 크다는 의미로 보는 것이 맞다. 현재 별도 클립에서는 대부분 `가다`로 쏠렸고, 일부 긴 클립에서는 `감사`가 average/majority로 나왔다.

따라서 다음 개선은 모델 구조 변경보다 데이터 보강이 우선이다. 특히 `label8_clip` 또는 직접 웹캠으로 촬영한 MediaPipe 기준 샘플을 학습 데이터에 추가하고, 라벨별 샘플 수를 균형 있게 맞춘 뒤 같은 클립으로 재평가해야 한다.

상세 비교 문서:

```text
report/1안_2안_비교_및_다음작업.md
```

## 11. 결론

이번 작업으로 8개 라벨에 대해 AIHub 원본 동영상을 MediaPipe Holistic 기준으로 다시 전처리하고, 그 데이터로 CNN-GRU sequence 모델을 재학습하는 흐름을 끝까지 확인했다.

현재 결과는 “웹캠 입력과 학습 입력의 좌표계 통일”이라는 목표는 달성했지만, 데이터 수와 웹캠 window 안정성 면에서는 아직 보완이 필요하다.

따라서 다음 판단 기준은 단순 validation accuracy보다 실제 웹캠에서 다음이 개선되는지다.

```text
1. 병원 쏠림이 줄어드는가
2. 감사/가다/자다 같은 동작이 top-3 안에 안정적으로 들어오는가
3. 손 검출이 32프레임 window를 충분히 채우는가
```
