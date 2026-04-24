# 2안 OpenPose 호환 재매핑 연구결과

작성일: 2026-04-27

## 1. 왜 이 실험을 했나

현재 모델은 AIHub에서 제공한 OpenPose 계열 keypoint로 학습되어 있다.

그런데 웹캠이나 YouTube 클립을 테스트할 때는 MediaPipe Holistic으로 landmark를 뽑고 있다. 학습할 때 본 좌표 체계와 실제 입력 좌표 체계가 다르면, 사람이 보기에는 같은 수어라도 모델 입장에서는 전혀 다른 모양으로 보일 수 있다.

그래서 이번에는 MediaPipe landmark를 기존 모델이 기대하는 **OpenPose BODY_25 + 양손 keypoint** 구조에 최대한 맞춰 넣어봤다. 이게 우리가 말한 2안이다.

## 2. 출발점

기존 모델이 기대하는 입력은 아래 구조다.

```text
OpenPose BODY_25 pose 25 points * [x, y, confidence] = 75
left hand 21 points * [x, y, confidence] = 63
right hand 21 points * [x, y, confidence] = 63
total = 201 features
```

문제는 MediaPipe pose의 앞 25개를 그대로 쓰면 OpenPose BODY_25의 순서와 맞지 않는다는 점이었다. 예를 들어 OpenPose의 1번은 Neck에 가까운 위치인데, MediaPipe의 1번은 얼굴 쪽 landmark다. 이런 식으로 포인트 의미가 밀리면 모델이 엉뚱한 라벨을 내놓을 가능성이 높다.

이전 웹캠 테스트에서 `병원` 쏠림과 여러 라벨 혼동이 계속 보였기 때문에, 우선 좌표 순서라도 맞춰보자는 생각으로 진행했다.

## 3. 바꾼 내용

수정한 파일은 아래와 같다.

- `src/data/keypoint_utils.py`
- `src/data/preprocess_mediapipe_videos.py`
- `backend/app.py`
- `web/src/components/SignLanguageStream.tsx`
- `scripts/evaluate_video_urls.py`
- `scripts/evaluate_local_videos.py`

주요 변경 사항:

- `mediapipe_landmarks_to_frame(results, layout="openpose_compat")`를 추가했다.
- 실시간 입력의 기본 layout은 `openpose_compat`로 두었다.
- MediaPipe pose landmark를 OpenPose BODY_25 의미 순서에 맞춰 재배열했다.
- MediaPipe 기준 재학습용 스크립트는 `mediapipe_fixed`를 명시해서, 이번 2안과 섞이지 않게 했다.
- 웹 화면 설정에 `랜드마크 매핑` 선택을 추가했다.
  - `OpenPose 호환`
  - `MediaPipe 원본`
- 로컬 mp4 클립을 바로 평가할 수 있도록 `scripts/evaluate_local_videos.py`를 추가했다.

## 4. MediaPipe Pose를 OpenPose BODY_25로 어떻게 맞췄나

MediaPipe Pose는 총 33개 landmark를 제공한다. 기존 모델은 OpenPose BODY_25의 25개 pose 슬롯을 기대하므로, MediaPipe의 33개 중 필요한 것만 골라서 25개 슬롯에 맞췄다.

매핑에 사용한 MediaPipe Pose landmark는 다음과 같다.

| MediaPipe Pose | OpenPose BODY_25로 사용한 위치 | 비고 |
|---|---|---|
| nose | Nose | 직접 매핑 |
| left_shoulder + right_shoulder | Neck | 두 어깨 평균 |
| right_shoulder | RShoulder | 직접 매핑 |
| right_elbow | RElbow | 직접 매핑 |
| right_wrist | RWrist | 직접 매핑 |
| left_shoulder | LShoulder | 직접 매핑 |
| left_elbow | LElbow | 직접 매핑 |
| left_wrist | LWrist | 직접 매핑 |
| left_hip + right_hip | MidHip | 두 골반 평균 |
| right_hip | RHip | 직접 매핑 |
| right_knee | RKnee | 직접 매핑 |
| right_ankle | RAnkle | 직접 매핑 |
| left_hip | LHip | 직접 매핑 |
| left_knee | LKnee | 직접 매핑 |
| left_ankle | LAnkle | 직접 매핑 |
| right_eye | REye | 직접 매핑 |
| left_eye | LEye | 직접 매핑 |
| right_ear | REar | 직접 매핑 |
| left_ear | LEar | 직접 매핑 |
| left_foot_index | LBigToe, LSmallToe | 발가락 2점을 하나로 대체 |
| left_heel | LHeel | 직접 매핑 |
| right_foot_index | RBigToe, RSmallToe | 발가락 2점을 하나로 대체 |
| right_heel | RHeel | 직접 매핑 |

아래 MediaPipe Pose landmark는 이번 변환에서 쓰지 않았다.

| MediaPipe Pose | 쓰지 않은 이유 |
|---|---|
| left_eye_inner, left_eye_outer | OpenPose BODY_25에 대응 위치가 없음 |
| right_eye_inner, right_eye_outer | OpenPose BODY_25에 대응 위치가 없음 |
| mouth_left, mouth_right | OpenPose BODY_25에 대응 위치가 없음 |
| left_pinky, right_pinky | 손가락은 별도의 hand landmark 21개를 사용 |
| left_index, right_index | 손가락은 별도의 hand landmark 21개를 사용 |
| left_thumb, right_thumb | 손가락은 별도의 hand landmark 21개를 사용 |

정리하면 이렇다.

```text
MediaPipe Pose 원본: 33개
OpenPose BODY_25 출력 슬롯: 25개
직접 또는 평균으로 사용한 MediaPipe Pose landmark: 21개
미매핑 MediaPipe Pose landmark: 12개
```

양손은 MediaPipe Pose의 손가락 포인트를 쓰지 않았다. 대신 MediaPipe Holistic의 `left_hand_landmarks`, `right_hand_landmarks` 21개씩을 OpenPose hand 21개 슬롯에 붙였다.

## 5. 평가한 영상

평가는 `sample_video` 폴더의 `숫자_영문명.mp4` 파일만 대상으로 했다. 총 16개다.

`sample_video` 폴더에는 파일이 두 종류 있다.

| 파일명 형식 | 의미 |
|---|---|
| 한글 파일명 mp4 | YouTube에서 받은 원본 클립 |
| 숫자_영문명 mp4 | 설명 구간을 제거하고 수어 동작 부분만 편집한 클립 |

최종 평가는 설명 구간을 줄인 영문 파일명 클립만 사용했다. 즉 원본 영상 전체가 아니라, 최대한 수어 동작만 남긴 상태에서 본 결과다.

실행 명령:

```powershell
python scripts/evaluate_local_videos.py `
  --video-dir sample_video `
  --output-dir outputs/local_video_eval_all_english `
  --pattern '[0-9]_*.mp4' `
  --frame-step 1 `
  --landmark-layout openpose_compat
```

결과 파일:

- `outputs/local_video_eval_all_english/local_video_eval_summary_openpose_compat.csv`
- `outputs/local_video_eval_all_english/local_video_eval_openpose_compat.json`

평가는 세 가지 방식으로 봤다.

| 지표 | 의미 |
|---|---|
| `final` | 마지막 32프레임 window의 예측 |
| `average` | 전체 window의 확률 평균 |
| `majority` | 전체 window에서 1등으로 나온 라벨의 다수결 |

수어는 동작이 끝난 뒤 손을 내리거나 멈추는 자세가 들어가면 마지막 window가 쉽게 흔들린다. 그래서 `final` 하나만 보는 것보다 `average`나 `majority`도 같이 보는 것이 낫다.

## 6. 전체 결과

16개 영문 클립 기준 결과는 아래와 같다.

```text
final accuracy    = 4/16 = 25.00%
average accuracy  = 5/16 = 31.25%
majority accuracy = 5/16 = 31.25%
```

기대했던 것만큼 올라가지는 않았다. 그래도 어떤 라벨은 분명히 가능성이 보였고, 어떤 라벨은 재매핑만으로는 어렵다는 것도 확인했다.

## 7. 라벨별로 본 결과

### 가다

| 파일 | final | average | majority | 판단 |
|---|---|---|---|---|
| `1_go.mp4` | 감사 | 감사 | 배고프다 | 실패 |
| `1_go2.mp4` | 가다 | 가다 | 가다 | 성공 |

`가다`는 모델이 아예 못 잡는 라벨은 아니었다. 클립에 따라 결과가 크게 갈렸다. 손 이동 방향, 촬영 각도, 동작 구간이 꽤 영향을 주는 것으로 보인다.

### 감사

| 파일 | final | average | majority | 판단 |
|---|---|---|---|---|
| `2_thanks.mp4` | 감사 | 감사 | 감사 | 성공 |
| `2_thanks2.mp4` | 감사 | 병원 | 병원 | 부분 성공 |

짧고 명확한 `감사` 클립은 매우 잘 맞았다. 다만 긴 클립에서는 중간 window들이 `병원` 쪽으로 많이 갔다. 준비 자세나 정지 구간이 섞이면 흔들릴 수 있다.

### 괜찮다

| 파일 | final | average | majority | 판단 |
|---|---|---|---|---|
| `3_okay.mp4` | 가다 | 가다 | 가다 | 실패 |
| `3_okay2.mp4` | 가다 | 가다 | 가다 | 실패 |
| `3_okay3.mp4` | 가다 | 가다 | 가다 | 실패 |
| `3_okay4.mp4` | 자다 | 가다 | 가다 | 실패 |
| `3_okay5.mp4` | 가다 | 가다 | 가다 | 실패 |

`괜찮다`는 여러 클립을 바꿔도 거의 계속 `가다`로 나왔다. 이 정도면 단순히 클립을 잘못 고른 문제라고 보기는 어렵다. 현재 OpenPose 호환 재매핑과 기존 모델 조합에서는 `괜찮다` 특징이 `가다`와 강하게 겹치는 것으로 보인다.

### 배고프다

| 파일 | final | average | majority | 판단 |
|---|---|---|---|---|
| `4_hungry.mp4` | 배고프다 | 배고프다 | 배고프다 | 성공 |

`배고프다`는 현재 방식에서도 안정적으로 맞았다.

### 병원

| 파일 | final | average | majority | 판단 |
|---|---|---|---|---|
| `5_hospital.mp4` | 가다 | 감사 | 감사 | 실패 |
| `5_hospital2.mp4` | 가다 | 감사 | 감사 | 실패 |
| `5_hospital3.mp4` | 감사 | 감사 | 감사 | 실패 |

`병원`은 여러 클립에서 계속 `감사`나 `가다`로 갔다. 재매핑 전에는 다른 입력에서 `병원` 쏠림이 문제였는데, 이번에는 실제 병원 동작이 오히려 `감사/가다`로 흐르는 문제가 보였다. 이 라벨은 재매핑만으로 해결하기 어렵다.

### 아프다

| 파일 | final | average | majority | 판단 |
|---|---|---|---|---|
| `6_sick.mp4` | 자다 | 자다 | 자다 | 실패 |
| `6_sick2.mp4` | 가다 | 가다 | 가다 | 실패 |
| `6_sick3.mp4` | 가다 | 가다 | 가다 | 실패 |

`아프다`도 안정적으로 잡히지 않았다. 손이 얼굴이나 몸 근처로 가는 동작들이 서로 섞이는 듯하고, 현재 모델에서는 `자다` 또는 `가다`로 많이 흘렀다.

### 우유

| 파일 | final | average | majority | 판단 |
|---|---|---|---|---|
| `7_milk.mp4` | 감사 | 우유 | 우유 | 부분 성공 |

`우유`는 마지막 window만 보면 틀리지만, 전체 평균과 다수결로 보면 맞았다. 마지막 자세 때문에 final이 뒤집힌 사례로 보면 된다.

### 자다

| 파일 | final | average | majority | 판단 |
|---|---|---|---|---|
| `8_sleep.mp4` | 가다 | 자다 | 자다 | 부분 성공 |

`자다`도 `우유`와 비슷했다. final은 틀렸지만 average와 majority 기준으로는 맞았다.

## 8. 정리

상대적으로 가능성이 보인 라벨:

```text
감사
배고프다
일부 가다
우유
자다
```

계속 어려웠던 라벨:

```text
괜찮다
병원
아프다
```

특히 `괜찮다`, `병원`, `아프다`는 설명 구간을 제거한 영문 클립에서도 계속 다른 라벨로 분류됐다. 그래서 원인을 단순히 “영상이 길어서”, “설명 구간이 섞여서”라고만 보기는 어렵다.

내 판단으로는 기존 OpenPose 기반 모델과 MediaPipe 입력 사이의 landmark 분포 차이가 여전히 크게 남아 있다.

## 9. 결론

2안은 해볼 만한 실험이었다. 좌표 순서를 아무렇게나 쓰는 것보다는 OpenPose 의미 순서에 맞추는 편이 확실히 더 타당하다. `감사`, `배고프다`, 일부 `가다`처럼 가능성이 보인 라벨도 있었다.

하지만 최종 해결책으로 보기에는 부족하다. 전체 정확도도 낮고, 무엇보다 `괜찮다`, `병원`, `아프다`가 여러 클립에서 반복적으로 실패했다.

결론은 아래처럼 정리할 수 있다.

```text
2안은 빠른 호환 실험으로는 의미가 있었다.
하지만 이 방식만으로는 한계가 분명하다.
다음 핵심 작업은 1안, 즉 MediaPipe 기준 재학습이다.
```

## 10. 다음 작업: 1안 MediaPipe 기준 재학습

1안은 원본 동영상에서 MediaPipe landmark를 다시 뽑고, 그 데이터로 모델을 새로 학습하는 방식이다.

목표는 간단하다.

```text
학습 데이터: 원본 동영상 -> MediaPipe landmark
실시간 입력: 웹캠 -> MediaPipe landmark
```

학습과 예측을 같은 landmark 체계로 맞추자는 것이다. 지금 2안에서 보인 한계를 생각하면, 이쪽이 더 근본적인 해결 방향이다.

팀원에게 전달할 메시지도 명확하다.

```text
2안 재매핑 실험은 여기서 마무리한다.
괜찮다/병원/아프다에서 한계가 분명히 보였다.
이제 핵심은 1안 MediaPipe 기준 재학습이다.
AIHub 원본 동영상에서 MediaPipe landmark를 추출해 재학습하는 작업이 다음 희망이다.
```

## 11. 참고 파일

- `report/openpose_compat_remap_eval.md`
- `report/다음작업_TODO.md`
- `src/data/keypoint_utils.py`
- `scripts/evaluate_local_videos.py`
- `outputs/local_video_eval_all_english/local_video_eval_summary_openpose_compat.csv`
- `outputs/local_video_eval_all_english/local_video_eval_openpose_compat.json`
