# MediaPipe 기준 재학습 검토

작성일: 2026-04-27

## 용어 정리

현재 소스 keypoint 데이터는 **OpenPose 계열 형식**으로 보는 것이 맞다.

현재 AIHub keypoint JSON은 landmark 추출 결과의 필드명과 포인트 구성이 OpenPose 스키마에 가깝다.

실제 JSON에서 확인되는 대표 필드:

```text
version: 1.3
people
people.pose_keypoints_2d
people.hand_left_keypoints_2d
people.hand_right_keypoints_2d
people.face_keypoints_2d
people.pose_keypoints_3d
people.hand_left_keypoints_3d
people.hand_right_keypoints_3d
```

특히 현재 모델 입력 feature 수가 `201`인 것도 OpenPose 계열 구조와 정확히 맞는다.

```text
pose_keypoints_2d       = 25 points * [x, y, confidence] = 75
hand_left_keypoints_2d  = 21 points * [x, y, confidence] = 63
hand_right_keypoints_2d = 21 points * [x, y, confidence] = 63

total = 75 + 63 + 63 = 201
```

정확한 문제 정의:

> 현재 학습 데이터는 AIHub 제공 **OpenPose BODY_25 + 양손 keypoint** 기반이고, 웹캠 입력은 **MediaPipe Holistic landmark** 기반이다. 두 좌표계와 포인트 순서가 다르기 때문에 같은 수어 동작도 모델 입장에서는 다른 분포로 보일 수 있다.

## 결론

첫 번째 케이스인 `MediaPipe로 원본 동영상을 다시 landmark 추출해서 학습`하는 방향은 현재 문제를 확인하기에 가장 정석적인 방법이다.

현재 모델은 AIHub 제공 OpenPose 계열 keypoint를 학습했고, 웹캠 시연은 MediaPipe landmark를 입력한다. 이 둘은 포인트 개수, 포인트 순서, 좌표 의미, confidence 처리 방식이 다르다. 그래서 웹캠에서 `병원` 쏠림이 생기는 주요 원인 후보가 된다.

MediaPipe 재학습은 학습 데이터와 실시간 입력을 같은 추출기로 맞추는 방식이다. 따라서 라이브 카메라 문제가 큰지, 좌표계 불일치가 큰지 분리해서 판단할 수 있다.

## 왜 OpenPose와 MediaPipe가 충돌할 수 있나

OpenPose BODY_25와 MediaPipe Pose는 모두 사람 몸 landmark를 제공하지만, 같은 인덱스가 같은 신체 부위를 의미하지 않는다.

예를 들어 현재 AIHub keypoint는 OpenPose 쪽 관례에 따라 `pose_keypoints_2d` 25개를 사용한다. 웹캠 입력은 MediaPipe에서 나온 pose landmark 중 앞 25개를 가져와 임시로 맞추고 있다. 하지만 “앞 25개”가 OpenPose BODY_25의 25개와 의미상 1:1로 일치하지 않는다.

또한 손 keypoint도 다음 차이가 있다.

- OpenPose: `hand_left_keypoints_2d`, `hand_right_keypoints_2d`가 `[x, y, confidence]` 배열이다.
- MediaPipe: landmark 객체에 `x, y, z`가 있고, hand landmark는 pose처럼 명확한 visibility가 항상 있는 구조가 아니다.

이 때문에 MediaPipe의 `z`를 그대로 세 번째 값으로 넣으면 OpenPose의 `confidence` 위치에 전혀 다른 의미의 값이 들어간다. 이 문제는 이전에 `z` 대신 confidence 성격의 값으로 바꾸며 일부 개선을 확인했다.

하지만 세 번째 값만 맞춘다고 좌표계 전체가 같아지는 것은 아니다. 근본적으로는 학습과 실시간 입력을 같은 추출기로 맞추는 것이 더 안정적이다.

## 현재 확인된 제약

- 현재 기본 manifest인 `data/sample_subset_manifest.csv`는 `keypoint_path` 중심이다.
- MediaPipe 재학습에는 원본 동영상 경로가 필요하다.
- 일부 스크립트에는 `video_path` 컬럼을 만드는 흐름이 있지만, 현재 학습용 master manifest는 원본 동영상만으로 바로 재학습할 수 있는 상태가 아니다.

따라서 팀원 작업의 첫 산출물은 `video_path`가 포함된 manifest여야 한다.

이번 실험의 라벨 범위는 현재 MVP 모델과 동일한 8개 라벨로 제한한다.

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

전체 AIHub 라벨을 대상으로 하지 않는다. 8개 라벨 밖의 데이터가 섞이면 기존 정확도 history, 웹캠 시연 결과, 현재 checkpoint와 비교하기 어려워진다.

## 새로 추가한 파일

- `src/data/preprocess_mediapipe_videos.py`
  - 원본 동영상에서 MediaPipe Holistic landmark를 추출한다.
  - `pose25 + left_hand21 + right_hand21` 고정 슬롯 구조로 저장한다.
  - 결과는 기존 `src.models.train_sequence`이 읽을 수 있는 `.npz` 형식이다.

- `config/mediapipe.yaml`
  - MediaPipe 재학습용 설정 파일이다.
  - 기본 출력은 `data/processed/sign_word_mediapipe_subset.npz`다.

- `data/mediapipe_video_manifest_template.csv`
  - 팀원이 채워야 할 manifest 예시다.

## 입력 manifest 형식

필수 컬럼:

| 컬럼 | 의미 |
|---|---|
| `sample_id` | 샘플 ID |
| `label` | 정답 라벨 |
| `video_path` | 원본 동영상 경로 |

권장 컬럼:

| 컬럼 | 의미 |
|---|---|
| `start` | 수어 동작 시작 시점 |
| `end` | 수어 동작 종료 시점 |
| `duration` | 전체 영상 길이 |
| `split` | `train` 또는 `validation` |

`start/end/duration`이 있으면 원본 영상 전체가 아니라 실제 수어 구간만 잘라서 landmark를 추출한다.

## 실행 예시

```powershell
conda activate gesture-test

python -m src.data.preprocess_mediapipe_videos `
  --config config/mediapipe.yaml `
  --manifest data/mediapipe_video_manifest.csv `
  --output data/processed/sign_word_mediapipe_subset.npz `
  --sequence_length 32 `
  --frame_step 1
```

학습:

```powershell
python -m src.models.train_sequence `
  --config config/mediapipe.yaml `
  --model_type cnn_gru `
  --epochs 80 `
  --batch_size 64
```

CPU 팀원은 같은 명령을 사용하되 시간이 더 오래 걸릴 수 있다. GPU 팀원은 PyTorch가 CUDA를 인식하면 학습 스크립트가 자동으로 `cuda`를 사용한다.

## 판단 기준

MediaPipe 재학습 모델로 웹캠 시연했을 때 다음을 본다.

1. `병원` 쏠림이 줄어드는가
2. YouTube 클립 평가와 웹캠 평가의 차이가 줄어드는가
3. `감사`, `자다`, `가다`처럼 사용자가 직접 테스트한 동작의 top prediction이 현실적인 후보로 이동하는가

이 결과가 좋아지면 좌표계 불일치가 주요 원인이었다고 볼 수 있다. 결과가 여전히 나쁘면 데이터 수, 라벨 경계, 동작 속도, 실시간 window 설계 쪽을 추가로 봐야 한다.

## 두 가지 해결 방향의 위치

현재 검토 중인 두 가지 케이스는 역할이 다르다.

1. **MediaPipe 기준 재학습**
   - 원본 동영상에서 MediaPipe landmark를 다시 추출한다.
   - 학습과 웹캠 입력의 좌표계를 통일한다.
   - 가장 정석적인 해결 방향이다.

2. **MediaPipe landmark를 OpenPose BODY_25 순서에 맞게 재매핑**
   - 원본 동영상 없이 빠르게 호환성을 시험할 수 있다.
   - 다만 MediaPipe와 OpenPose는 애초에 감지 모델과 포인트 정의가 다르므로 완전한 변환은 어렵다.
   - 빠른 실험 또는 임시 호환 계층으로 보는 것이 적절하다.

개인적인 우선순위는 1번이 먼저다. 1번 결과가 좋아지면 좌표계 불일치가 주된 원인임을 확인할 수 있고, 이후 2번은 원본 영상 확보가 어려운 경우를 위한 보조 전략으로 가져가면 된다.
