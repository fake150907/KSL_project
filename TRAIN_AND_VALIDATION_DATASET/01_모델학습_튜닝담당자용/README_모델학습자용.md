# 모델학습/튜닝 담당자용 README

이 폴더는 새 모델을 만들거나 하이퍼파라미터를 튜닝하는 팀원이 사용하는 전달본입니다.

## 폴더 구조

```text
01_모델학습_튜닝담당자용/
  train/       학습용 FILE01~16 통합 데이터셋
  validation/  외부 검증용 VALIDATION MVP 데이터셋
  configs/     LSTM, CNN-GRU 학습 설정 YAML
  metrics/     기존 학습/검증 결과 JSON
```

## 새 모델 학습 시 필수 파일

CNN-GRU 학습 최소 필수 파일:

```text
train/mediapipe_npz_FILE01_16_FULL.npz
train/label_map_FILE01_16_FULL.json
configs/mediapipe_FILE01_16_FULL_cnn_gru.yaml
```

LSTM 학습 최소 필수 파일:

```text
train/mediapipe_npz_FILE01_16_FULL.npz
train/label_map_FILE01_16_FULL.json
configs/mediapipe_FILE01_16_FULL_lstm.yaml
```

`label_map_FILE01_16_FULL.json`은 필수입니다. 모델의 출력 index가 어떤 단어인지 정하는 기준이라서, 이 순서가 바뀌면 예측 결과 단어가 틀어집니다.

## 참고 파일

| 위치 | 용도 |
|---|---|
| `train/shard_manifest_FILE01_16_FULL.csv` | 각 샘플의 sample_id, word_id, label, split, source_file_tag 확인용 |
| `train/preprocess_report_FILE01_16_FULL.md` | 전처리 결과 요약, shape, split, 라벨 분포 확인용 |
| `validation/mediapipe_npz_VALIDATION_MVP_FULL.npz` | 학습에 섞지 않는 외부 검증용 NPZ |
| `validation/shard_manifest_VALIDATION_MVP_FULL.csv` | validation 샘플 목록 확인용 |
| `validation/validation_eval_summary.md` | 기존 모델의 외부 validation 평가 요약 |
| `metrics/*.json` | 기존 LSTM/CNN-GRU 학습 로그와 validation 결과 |

## 학습 데이터 요약

- 학습 데이터셋: FILE01~16 통합본
- 샘플 수: 484개
- 입력 shape: `(484, 32, 225)`
- sequence length: 32프레임
- feature 수: 225개
- 라벨 수: 8개
- 라벨 순서: `가다`, `감사`, `괜찮다`, `배고프다`, `병원`, `아프다`, `우유`, `자다`

## MediaPipe 좌표 사용 범위

이번 모델은 원본 동영상이 아니라 MediaPipe로 추출한 좌표를 학습합니다.

사용한 좌표:

```text
pose 33개 + left hand 21개 + right hand 21개
각 landmark의 x/y/z 3개 값 사용
(33 + 21 + 21) * 3 = 225
```

사용하지 않은 좌표/정보:

- face mesh 468개
- iris 좌표
- segmentation mask
- 원본 RGB 프레임
- pose visibility/presence confidence

즉 이번 모델은 얼굴 표정보다는 상체 자세, 팔 움직임, 양손/손가락 움직임을 중심으로 학습한 모델입니다.

## 최신 모델 정확도

| 모델 | 내부 holdout 최고 정확도 | best epoch | 외부 VALIDATION 정확도 | 비고 |
|---|---:|---:|---:|---|
| LSTM | 92.68% | 47 | 62.67% | 비교 모델 |
| CNN-GRU | 100.00% | 34 | 79.33% | 현재 권장 모델 |

CNN-GRU의 내부 holdout 100.00%는 과적합 가능성이 있으므로, 실사용 성능 설명에는 외부 VALIDATION 정확도 79.33%를 더 중요하게 보는 것이 안전합니다.
