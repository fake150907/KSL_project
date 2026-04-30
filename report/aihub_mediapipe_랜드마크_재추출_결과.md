# AIHub 8라벨 MediaPipe 랜드마크 재추출 결과

작성일: 2026-04-27

## 1. 작업 목적

AIHub 원본 수어 동영상 8개 라벨만 대상으로 MediaPipe Holistic landmark를 다시 추출했다.

대상 라벨:

```text
가다
감사
괜찮다
배고프다
병원
아프다
우유
자다
```

이번 작업은 기존 산출물을 덮어쓰지 않고, 새 파일명으로 재추출 결과를 따로 남겼다.

## 2. 실행 입력

사용 manifest:

```text
data/mediapipe_video_manifest.csv
```

원본 동영상 위치:

```text
data/raw/videos/
```

manifest에 포함된 샘플 수:

| 항목 | 값 |
|---|---:|
| 전체 샘플 수 | 75 |
| train | 60 |
| validation | 15 |
| 라벨 수 | 8 |

## 3. 실행 명령

```powershell
C:\Users\fake1\anaconda3\envs\gesture-test\python.exe -m src.data.preprocess_mediapipe_videos `
  --config config/mediapipe.yaml `
  --manifest data/mediapipe_video_manifest.csv `
  --output data/processed/sign_word_mediapipe_aihub_rerun_20260427.npz `
  --sequence_length 32 `
  --frame_step 1
```

## 4. 생성 산출물

| 파일 | 내용 |
|---|---|
| `data/processed/sign_word_mediapipe_aihub_rerun_20260427.npz` | 재추출된 학습/검증 tensor |
| `data/processed/sign_word_mediapipe_aihub_rerun_20260427.meta.json` | 샘플별 추출 통계 |
| `outputs/reports/aihub_rerun_mediapipe_eval_20260427.json` | 현재 checkpoint 기준 validation 재평가 결과 |

## 5. 랜드마크 재추출 결과

| 항목 | 결과 |
|---|---:|
| shape | `(75, 32, 201)` |
| sequence length | 32 |
| feature count | 201 |
| failed rows | 0 |

feature layout:

```text
MediaPipe fixed slots: pose25 + left_hand21 + right_hand21, each [x,y,confidence]
```

## 6. 라벨별 추출 통계

| 라벨 | 샘플 수 | processed frames | hand frames | pose frames | 평균 processed | 평균 hand |
|---|---:|---:|---:|---:|---:|---:|
| 가다 | 20 | 595 | 595 | 595 | 29.75 | 29.75 |
| 감사 | 5 | 185 | 185 | 185 | 37.00 | 37.00 |
| 괜찮다 | 5 | 140 | 140 | 140 | 28.00 | 28.00 |
| 배고프다 | 10 | 375 | 375 | 375 | 37.50 | 37.50 |
| 병원 | 5 | 135 | 135 | 135 | 27.00 | 27.00 |
| 아프다 | 5 | 160 | 160 | 160 | 32.00 | 32.00 |
| 우유 | 10 | 440 | 440 | 440 | 44.00 | 44.00 |
| 자다 | 15 | 400 | 400 | 400 | 26.67 | 26.67 |

모든 샘플에서 pose와 hand landmark가 추출됐다. 따라서 이번 재추출 자체는 실패 없이 정상 완료된 것으로 볼 수 있다.

## 7. 현재 checkpoint 기준 재평가

재추출한 NPZ를 현재 1안 MediaPipe 재학습 checkpoint에 넣어 validation split을 다시 평가했다.

사용 checkpoint:

```text
outputs/checkpoints/sequence_model.pt
```

결과:

| 항목 | 값 |
|---|---:|
| validation 샘플 수 | 15 |
| accuracy | 73.33% |

validation 정답과 예측:

| 순서 | 정답 | 예측 | 결과 |
|---:|---|---|---|
| 1 | 가다 | 가다 | 성공 |
| 2 | 가다 | 가다 | 성공 |
| 3 | 가다 | 가다 | 성공 |
| 4 | 가다 | 가다 | 성공 |
| 5 | 감사 | 감사 | 성공 |
| 6 | 괜찮다 | 자다 | 실패 |
| 7 | 배고프다 | 배고프다 | 성공 |
| 8 | 배고프다 | 배고프다 | 성공 |
| 9 | 병원 | 가다 | 실패 |
| 10 | 아프다 | 우유 | 실패 |
| 11 | 우유 | 우유 | 성공 |
| 12 | 우유 | 배고프다 | 실패 |
| 13 | 자다 | 자다 | 성공 |
| 14 | 자다 | 자다 | 성공 |
| 15 | 자다 | 자다 | 성공 |

## 8. 해석

AIHub 원본 mp4 기준 MediaPipe landmark 재추출은 정상적으로 완료됐다. shape, 라벨 목록, 실패 건수 모두 기존 1안 산출물과 일치한다.

현재 checkpoint로 재평가했을 때 validation accuracy도 `73.33%`로 유지됐다. 즉 AIHub 내부 validation 기준에서는 재추출 결과가 안정적으로 재현된 것이다.

다만 이전 `label8_clip` 평가에서는 13개 클립 기준 0%가 나왔으므로, 현재 문제는 AIHub landmark 추출 실패가 아니라 외부/편집/웹캠 클립과 AIHub 학습 데이터 사이의 도메인 차이에 가깝다.

## 9. 다음 판단

현재 우선순위는 아래와 같다.

1. AIHub 기준 파이프라인은 정상으로 보고 유지한다.
2. `label8_clip` 또는 직접 웹캠 촬영 클립을 MediaPipe 기준 학습 데이터에 추가한다.
3. 라벨별 샘플 수를 균형 있게 맞춘다.
4. 특히 `괜찮다`, `병원`, `아프다`, `감사` 샘플을 늘린다.
5. 재학습 후 같은 `label8_clip`으로 다시 평가한다.

한 줄 결론:

```text
AIHub 동영상 -> MediaPipe landmark 재추출은 성공.
AIHub validation 결과도 73.33%로 재현.
남은 문제는 외부/웹캠 클립 일반화 성능이다.
```
