# WEB Model Handoff

## 전달 파일

```text
sequence_model_lstm_FILE01_03-FILE10_12_valacc0.8095.pt
sequence_model_cnn_gru_FILE01_03-FILE10_12_valacc0.8571.pt
model_labels.json
model_readme.md
```

추천 모델은 CNN-GRU입니다.

```text
sequence_model_cnn_gru_FILE01_03-FILE10_12_valacc0.8571.pt
```

LSTM 모델은 비교/백업용으로 함께 전달합니다.

## 학습 데이터

학습에 사용한 데이터:

```text
team_handover_outputs/FILE01_03-FILE10_12/mediapipe_npz_FILE01_03-FILE10_12.npz
```

구성:

```text
FILE01~FILE03
FILE10~FILE12
총 109개 샘플
8개 MVP 라벨
```

`data/collected/File04_File06`는 점검했지만 이번 모델 학습에는 넣지 않았습니다. 일부 manifest `sample_id`와 NPZ `sample_ids` 표기가 달라 통합 보류했습니다.

## 입력 형식

단일 샘플:

```text
shape: (32, 225)
```

배치 입력:

```text
shape: (N, 32, 225)
```

특징:

```text
MediaPipe xyz
pose33 + left_hand21 + right_hand21 = 75 landmarks
75 landmarks * 3 xyz = 225 features per frame
normalization: shoulder-center + shoulder-width scale
```

## 라벨 순서

모델 출력 index는 아래 순서와 정확히 매칭해야 합니다.

```json
["가다", "감사", "괜찮다", "배고프다", "병원", "아프다", "우유", "자다"]
```

예:

```text
pred_idx = 0 -> 가다
pred_idx = 7 -> 자다
```

## 학습 방식

학습 코드:

```text
team_handover_2026-04-29/INTEGRATION_MASTER/src/models/train_sequence.py
team_handover_2026-04-29/INTEGRATION_MASTER/src/models/model_sequence.py
```

중요 실행 조건:

```text
작업 위치: team_handover_2026-04-29/INTEGRATION_MASTER
PYTHONPATH: team_handover_2026-04-29/INTEGRATION_MASTER
```

repo root에서 `python -m src.models.train_sequence`로 실행하면 root `src`가 로드될 수 있어 결과가 달라질 수 있습니다.

공통 설정:

```text
epochs: 50
batch_size: 16
learning_rate: 0.001
hidden_size: 64
num_layers: 1
dropout: 0.3
sequence_length: 32
feature_count: 225
validation split: 학습 NPZ 내부 label-wise holdout
```

LSTM:

```text
model_type: rnn
rnn_type: lstm
best_epoch: 15
internal_holdout_validation_accuracy: 0.8095
```

CNN-GRU:

```text
model_type: cnn_gru
rnn_type: gru
conv_channels: 64
best_epoch: 32
internal_holdout_validation_accuracy: 0.8571
```

## 실제 AIHub VALIDATION 비디오 평가

학습에 사용하지 않은 AIHub VALIDATION 비디오에서 8개 MVP 단어 전체를 추출해 별도 평가했습니다.

검증 데이터:

```text
team_handover_outputs/VALIDATION_MVP_FULL/mediapipe_npz_VALIDATION_MVP_FULL.npz
samples: 150
shape: (150, 32, 225)
failed_samples: 0
```

라벨 분포:

```text
가다 40
감사 10
괜찮다 10
배고프다 20
병원 10
아프다 10
우유 20
자다 30
```

외부 validation 결과:

```text
LSTM    : 70 / 150 correct, accuracy 0.4667
CNN-GRU : 82 / 150 correct, accuracy 0.5467
```

상세 결과:

```text
team_handover_outputs/VALIDATION_EVAL_FULL/validation_eval_summary.md
team_handover_outputs/VALIDATION_EVAL_FULL/lstm_FILE01_03-FILE10_12_on_validation_full_metrics.json
team_handover_outputs/VALIDATION_EVAL_FULL/cnn_gru_FILE01_03-FILE10_12_on_validation_full_metrics.json
```

## 로딩 메모

`.pt` checkpoint 안에는 아래 값들이 들어 있습니다.

```text
state_dict
labels
input_size
config
model_type
best_epoch
best_accuracy
```

로드할 때는 같은 architecture로 모델을 만든 뒤 `state_dict`를 넣어야 합니다.

```text
input_size: 225
num_classes: 8
hidden_size: 64
num_layers: 1
dropout: 0.3
```

CNN-GRU는 추가로:

```text
conv_channels: 64
```

## 설명 시 주의

내부 validation accuracy는 109개 학습 파일 묶음 안에서 나눈 holdout 점수입니다. 실제 AIHub VALIDATION 비디오 150개 평가 결과는 별도 외부 검증 점수이며, 발표나 강사 점검에서는 두 값을 구분해서 설명하는 것이 좋습니다.
