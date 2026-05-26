# 모델 사양서 (WEB 담당자용)

**대상**: 한국 수어 인식 v2 모델 (WORD + SENTENCE)
**작성일**: 2026-05-23

---

## 1. 두 모델 비교 — 가장 중요

| 항목 | WORD v2 | SENTENCE v2 |
|---|---|---|
| **체크포인트 파일** | `models/word_stage2.pt` | `models/sentence_stage2_reproc.pt` |
| **모델 클래스** | `CNNGRUAttn` | **`CNNGRUAttnV2`** (다름!) |
| **Dataset 클래스** | `NPZKeypointDataset` | **`NPZKeypointDatasetV2`** |
| **입력 채널** | xyz **3-channel** | **xyzc 4-channel** ⚠ |
| **Keypoint 수** | 75 | 75 |
| **frame당 dim** | 225 (= 75 × 3) | **300** (= 75 × 4) ⚠ |
| **Sequence length** | 가변 | **T_max=128 padding** ⚠ |
| **valid_lengths 인자** | 불필요 | **필수** ⚠ |
| **클래스 수** | 3000 | 2000 |
| **forward 호출** | `model(x)` | **`model(x, valid_lengths)`** ⚠ |
| **출력** | logits (B, 3000) | logits (B, 2000) |
| **운영 Temperature** | 0.8627 | 0.9156 |
| **운영 검증 정확도** | val_top1 96.55% (REAL18 97.53%) | val_top1 92.26% (REAL18 88.54%) |

⚠ = v1과 다른 부분, WEB 코드 수정 필요

---

## 2. 입력 데이터 형식

### 2.1 MediaPipe Holistic 추출 설정 (양쪽 공통)
```python
import mediapipe as mp
holistic = mp.solutions.holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,       # 정확도/속도 균형
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
```

### 2.2 Keypoint 75개 구성 (양쪽 공통)

소스 확정: [src/data/keypoint_utils.py](https://) `mediapipe_landmarks_to_frame()` 의 `layout="mediapipe_xyzc"` (SENTENCE) / `"mediapipe_xyz"` (WORD) 분기.

| 부위 | 개수 | 출처 |
|---|---:|---|
| **Pose (전신)** | **33** | `pose_landmarks.landmark[:33]` — MediaPipe Pose 전체 33개 (얼굴 5 + 상체 + 팔 + 하체 포함) |
| **Left Hand** | 21 | `left_hand_landmarks.landmark[:21]` |
| **Right Hand** | 21 | `right_hand_landmarks.landmark[:21]` |
| **합계** | **75** | (face_landmarks 모듈은 사용 안 함) |

> ⚠ **MediaPipe Holistic 호출 시 `refine_face_landmarks=False`로 face 모듈 비활성화** 가능 (성능 ↑). 위 코드에 명시.
>
> 누락된 landmark (예: hand가 미감지) → 해당 행을 0으로 채움. SENTENCE의 4번째 채널(c)도 0 (visibility=0 = 못 봄 신호).

### 2.3 Keypoint 채널 구성

#### WORD (xyz 3-channel, 225-dim/frame)
```
frame_t = [x_0, y_0, z_0, x_1, y_1, z_1, ..., x_74, y_74, z_74]  # 225 dim
```

#### SENTENCE (xyzc 4-channel, 300-dim/frame)
```
frame_t = [x_0, y_0, z_0, c_0, x_1, y_1, z_1, c_1, ..., x_74, y_74, z_74, c_74]  # 300 dim
```
- `c` = confidence (MediaPipe visibility 값, 0~1)
- 누락 키포인트는 x=y=z=0, c=0

### 2.4 Sequence 정규화

#### WORD
- 가변 길이 그대로 입력 가능 (단일 동작 영상 1초 내외, ~30 frame)
- min frame 권장: 8 frame 이상

#### SENTENCE
- **T_max = 128 frame** 고정 (모자라면 0 padding, 넘으면 잘라야 함)
- **valid_lengths** = 실제 유효 frame 수 (1차원 tensor, batch당 1개 정수)
- 영상 시간: full_span_bounds 적용 (시작·종료 morpheme 모두 포함)

---

## 3. 모델 입출력 텐서 shape

### WORD
```python
# 입력
x = torch.Tensor of shape (B, T, 225)  # B=batch, T=frame 수 (가변)

# forward
logits = model(x)                       # shape (B, 3000)

# 후처리 (top5 + temperature)
T = 0.8627
probs = torch.softmax(logits / T, dim=1)
top5 = probs.topk(5, dim=1)              # values (B,5), indices (B,5)
```

### SENTENCE
```python
# 입력
x = torch.Tensor of shape (B, 128, 300)         # T_max=128 padded
valid_lengths = torch.IntTensor of shape (B,)   # 각 sample의 실제 frame 수

# forward
logits = model(x, valid_lengths)                # shape (B, 2000)

# 후처리
T = 0.9156
probs = torch.softmax(logits / T, dim=1)
top5 = probs.topk(5, dim=1)
```

---

## 4. Output → 사람이 읽을 수 있는 라벨 변환

### WORD
```python
import json
label_map = json.load(open('labels/label_map_word.json'))
# label_map 구조: {"0": "WORD0001", "1": "WORD0002", ...} 또는 list

predicted_class_id = top5.indices[0, 0].item()
word_id = label_map[str(predicted_class_id)]  # 예: "WORD0579"
```

### SENTENCE
```python
label_map = json.load(open('labels/label_map_sentence.json'))
predicted_class_id = top5.indices[0, 0].item()
sen_id = label_map[str(predicted_class_id)]   # 예: "SEN0322"

# (선택) gloss 표시
gloss_vocab = json.load(open('lookup/gloss_vocab_sentence.json'))
gloss_seq = gloss_vocab.get(sen_id)            # 예: ["복지카드", "잃다"]
```

### Lookup Table (운영 시나리오용)
```python
lookup = json.load(open('lookup/lookup_table.json'))
# lookup_table[sen_id] = {"label": "잃어버리다 (복지카드/물건 합성)", "weight": 1.5}
```

---

## 5. 모델 메타 정보 (체크포인트 안에 포함)

체크포인트는 다음을 포함:
```python
state = torch.load('models/word_stage2.pt', map_location='cpu')
# state["model_state"]    -> 가중치
# state["model_config"]   -> 모델 생성 인자 (CNNGRUAttn 직접 init 가능)
# state["metrics"]        -> 학습 시 최고 성능 (참고용)
```

SENTENCE도 동일 구조, 단 `model_config["variant"] == "v2"` 키가 있으면 `CNNGRUAttnV2` 생성 시 제거 후 사용.

---

## 6. 운영 검증 결과 요약

| 모델 | AIHub val_top1 | REAL18 unseen | 시연자 격차 | ECE (REAL17→T 적용) |
|---|---:|---:|---:|---|
| WORD v2 | 96.55% | 97.53% | +1.98%p (강한 일반화) | 0.1100 → 0.0319 |
| SENTENCE v2 | 92.26% | 88.54% | -7.44%p (문장 unseen 어려움) | 0.0699 → 0.0242 |

운영 성공 기준 (v7 §E.3):
- 시나리오 E2E REAL18 0.9045 (목표 0.85+) ✅
- Critical Word Recall 1.0 ✅
- Lookup hit rate 1.0 ✅

상세는 본 ZIP과 별도로 `reports_final_sentence/` 보고서 참고.

---

## 7. 의존성 (PyTorch 환경)

| 패키지 | 버전 |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.0+ (학습 환경: 2.11.0+cu128) |
| numpy | 1.20+ |
| mediapipe | 0.10+ (양쪽 모델 입력 keypoint 추출) |

GPU 권장 (CPU 가능, 추론 속도 1/10 수준).
