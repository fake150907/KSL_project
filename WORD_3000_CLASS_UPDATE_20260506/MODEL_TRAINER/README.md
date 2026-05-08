# MODEL_TRAINER

모델 학습 담당자용 3000개 WORD ID 기준 안내입니다.

## 학습 담당자가 받는 입력

| 파일 | 용도 |
|---|---|
| 통합 NPZ | 학습 입력 데이터. `X`, `y`, `splits`, `sample_ids`, `labels` 포함 |
| `label_map_word_id_3000.json` | `WORD ID -> class id` 기준표 |
| `idx_to_label_3000.json` | 예측 결과 class id를 `WORD ID`와 한국어 라벨로 복원 |
| `word_id_label_master_3000.csv` | 사람이 확인하는 전체 WORD ID 기준표 |

## 학습 코드는 무엇을 바꿔야 하나

전처리/취합된 NPZ의 `y`는 이미 `WORD ID` 기준 `0~2999`로 들어갑니다. 학습자는 `y`를 다시 만들 필요가 없습니다.

대신 모델의 출력 클래스 수를 3000으로 바꿔야 합니다.

| 수정 지점 | 기존 8개 MVP | 3000개 전체 |
|---|---|---|
| 클래스 수 | `8` | `3000` |
| config | `max_classes: 8` | `max_classes: 3000` |
| 모델 마지막 출력층 | `Linear(..., 8)` | `Linear(..., 3000)` |
| 결과 해석 | 한국어 label 기준 | WORD ID 기준 |

## 권장 방식

가능하면 클래스 수를 하드코딩하지 말고 NPZ의 `labels` 길이에서 읽어 주세요.

```python
data = np.load(npz_path, allow_pickle=True)
num_classes = len(data["labels"])
```

3000개 전체 NPZ라면:

```python
num_classes == 3000
```

PyTorch 모델 예시:

```python
self.classifier = nn.Linear(hidden_size, num_classes)
```

하드코딩이 필요하다면:

```python
NUM_CLASSES = 3000
```

## 예측 결과 해석

모델 출력은 숫자 class id입니다.

```python
pred_id = int(logits.argmax(dim=1).item())
```

이 숫자를 `idx_to_label_3000.json`으로 해석합니다.

```python
import json

with open("idx_to_label_3000.json", "r", encoding="utf-8-sig") as f:
    idx_to_label = json.load(f)

result = idx_to_label[str(pred_id)]
```

예시:

```json
{
  "word_id": "WORD1381",
  "word_no": 1381,
  "label": "괜찮다"
}
```

## 주의할 점

한국어 라벨명만 보고 class id를 만들면 안 됩니다. 전체 3000개에서는 같은 한국어 라벨이 여러 `WORD ID`에 중복으로 존재할 수 있습니다.

학습 기준은 항상 `WORD ID`입니다.

