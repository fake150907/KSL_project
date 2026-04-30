# YouTube 클립 예측 평가

작성일: 2026-04-27

## 목적

라이브 카메라에서 `감사` 동작이 이상하게 예측되는 원인이 카메라 자체 문제인지, 아니면 현재 웹 데모의 MediaPipe 입력과 AIHub 학습 데이터 사이의 전처리/분포 차이인지 확인하기 위해 사용자가 제공한 YouTube 클립 2개를 현재 CNN-GRU 모델로 평가했다.

## 평가 대상

- 감사: `https://youtu.be/T9FCgQp8V98?si=U-gPF9mdNsR_XbHh`
- 괜찮다: `https://youtube.com/shorts/HfqEWZ1ssQk?si=eYjiZHxBqM6eoE4c`

다운로드된 영상:

- `outputs/video_eval/videos/감사.mp4`
- `outputs/video_eval/videos/괜찮다.mp4`

## 평가 방법

- 스크립트: `scripts/evaluate_video_urls.py`
- 모델: `outputs/checkpoints/sequence_model.pt`
- 모델 구조: CNN-GRU
- sequence length: 32
- 프레임 처리: 모든 프레임 사용 (`--frame-step 1`)
- 랜드마크 추출: MediaPipe Holistic
- 추론 방식: 32프레임 sliding window별 top-k 예측
- 추가 확인: 좌우반전 입력도 별도 평가 (`--mirror`)

## 일반 방향 결과

| 기대 라벨 | hand frames | windows | 최종 1순위 | 최종 신뢰도 | 평균 1순위 | 평균 신뢰도 | 다수결 1순위 |
|---|---:|---:|---|---:|---|---:|---|
| 감사 | 78 | 47 | 병원 | 85.19% | 병원 | 82.42% | 병원 100% |
| 괜찮다 | 907 | 876 | 병원 | 89.60% | 병원 | 90.60% | 병원 100% |

일반 방향 상세 결과:

- `outputs/video_eval/youtube_eval.json`
- `outputs/video_eval/youtube_eval_summary.csv`

## 좌우반전 결과

| 기대 라벨 | hand frames | windows | 최종 1순위 | 최종 신뢰도 | 평균 1순위 | 평균 신뢰도 | 다수결 1순위 |
|---|---:|---:|---|---:|---|---:|---|
| 감사 | 79 | 48 | 병원 | 81.38% | 병원 | 82.15% | 병원 100% |
| 괜찮다 | 902 | 871 | 병원 | 91.96% | 병원 | 92.13% | 병원 100% |

좌우반전 상세 결과:

- `outputs/video_eval/youtube_eval_mirror.json`
- `outputs/video_eval/youtube_eval_mirror_summary.csv`

## 참고: 학습 데이터 기준 클래스별 성능

현재 `data/processed/sign_word_subset.npz` 전체에 대해 활성 CNN-GRU 모델을 다시 추론했을 때:

| 라벨 | 정답/전체 | 정확도 |
|---|---:|---:|
| 가다 | 211/240 | 87.92% |
| 감사 | 59/60 | 98.33% |
| 괜찮다 | 57/60 | 95.00% |
| 배고프다 | 108/120 | 90.00% |
| 병원 | 60/60 | 100.00% |
| 아프다 | 58/60 | 96.67% |
| 우유 | 106/120 | 88.33% |
| 자다 | 179/180 | 99.44% |

즉 모델이 `감사`, `괜찮다` 라벨 자체를 전혀 못 배운 상태는 아니다.

## 판단

현재 결과는 단순한 라이브 카메라 장치 문제라기보다, **AIHub keypoint JSON으로 학습한 입력 분포와 MediaPipe로 새로 추출한 live/video 입력 분포가 다르기 때문에 생기는 문제**로 보는 것이 타당하다.

근거:

- 유튜브 영상에서도 웹캠과 같은 MediaPipe 추출 파이프라인을 쓰면 `감사`, `괜찮다`가 모두 `병원`으로 강하게 쏠린다.
- 좌우반전을 해도 결과가 바뀌지 않아 단순 mirror 문제 가능성은 낮다.
- 학습 데이터에 대해서는 `감사`, `괜찮다` 정확도가 높다.
- 따라서 모델 구조보다 입력 정규화, landmark 구성, frame window 구성, AIHub와 MediaPipe 좌표 체계 차이를 먼저 봐야 한다.

## 1차 수정: MediaPipe 3번째 채널 보정

2026-04-27에 `src/data/keypoint_utils.py`의 `mediapipe_landmarks_to_frame()`을 수정했다.

원인:

- AIHub 학습 데이터의 2D keypoint는 `[x, y, confidence]` 형태다.
- 기존 live/video MediaPipe 변환은 `[x, y, z]` 형태로 넣고 있었다.
- `sequence_to_tensor()`의 정규화는 x/y만 정규화하고 3번째 채널은 그대로 유지한다.
- 따라서 모델은 학습 때 보던 confidence 채널 대신 MediaPipe z-depth 채널을 입력받고 있었고, 이 때문에 live/video 입력이 `병원`으로 강하게 쏠린 것으로 보인다.

수정:

- pose: `[lm.x, lm.y, lm.visibility]`
- hand: `[lm.x, lm.y, 1.0]`

## 1차 수정 후 재평가

결과 위치:

- 일반 방향: `outputs/video_eval_confidence_fix/youtube_eval.json`
- 일반 방향 요약: `outputs/video_eval_confidence_fix/youtube_eval_summary.csv`
- 좌우반전: `outputs/video_eval_confidence_fix/youtube_eval_mirror.json`
- 좌우반전 요약: `outputs/video_eval_confidence_fix/youtube_eval_mirror_summary.csv`

### 일반 방향

| 기대 라벨 | hand frames | windows | 최종 1순위 | 최종 신뢰도 | 평균 1순위 | 평균 신뢰도 | 다수결 1순위 |
|---|---:|---:|---|---:|---|---:|---|
| 감사 | 78 | 47 | 감사 | 52.21% | 감사 | 64.93% | 감사 87.23% |
| 괜찮다 | 907 | 876 | 병원 | 99.32% | 병원 | 62.87% | 병원 64.73% |

### 좌우반전

| 기대 라벨 | hand frames | windows | 최종 1순위 | 최종 신뢰도 | 평균 1순위 | 평균 신뢰도 | 다수결 1순위 |
|---|---:|---:|---|---:|---|---:|---|
| 감사 | 79 | 48 | 배고프다 | 73.97% | 배고프다 | 58.36% | 배고프다 66.67% |
| 괜찮다 | 902 | 871 | 감사 | 82.90% | 감사 | 56.25% | 감사 65.33% |

해석:

- `감사` 일반 방향은 수정 후 정상에 가깝게 복구됐다.
- 따라서 `병원` 쏠림의 큰 원인 중 하나는 3번째 채널 불일치가 맞다.
- `괜찮다`는 아직 정상 복구되지 않았다. 이 문제는 3번째 채널만으로는 해결되지 않으며, 동작 구간, 라벨 자체의 영상 차이, point 순서/정규화, 학습 데이터와 외부 영상의 자세 차이를 추가로 봐야 한다.
- 좌우반전은 오히려 악화되므로 현재 YouTube `감사` 기준으로는 일반 방향이 더 적합하다.

## 다음 작업

1. 웹 데모를 재시작한 뒤 `감사`가 실제 카메라에서도 복구되는지 확인한다.
2. `괜찮다` 영상의 전체 35.7초 중 실제 수어 구간만 잘라서 다시 평가한다.
3. AIHub `괜찮다`와 YouTube `괜찮다`의 pose/hand 위치 분포를 비교한다.
4. `괜찮다`가 계속 틀리면 live/video 샘플을 추가해 calibration/fine-tuning을 진행한다.
5. 이후 `감사`, `괜찮다`, `병원` 3개 라벨부터 작은 진단 세트로 안정화한다.
