# 정확도 History

이 문서는 현재 8개 라벨 MVP 데이터셋 확장 과정에서 확인한 baseline 정확도 변화를 기록한다.

## 기준

- 라벨 수: 8개
- 라벨: 가다, 감사, 괜찮다, 배고프다, 병원, 아프다, 우유, 자다
- 입력 형태: keypoint sequence `(sequence_length=32, feature_dim=201)`
- 검증 split: 현재 manifest 기준 `U` angle이 validation, `D/F/L/R` angle이 train
- baseline 모델: RandomForest

## RandomForest Baseline

| 단계 | 사용 REAL 범위 | 샘플 수 | Train | Validation | Accuracy |
|---|---:|---:|---:|---:|---:|
| 초기 이관 상태 | REAL01~REAL03 | 225 | 180 | 45 | 77.78% |
| 04번 추가 후 | REAL01~REAL04 | 300 | 240 | 60 | 78.33% |
| 05번 추가 후 | REAL01~REAL05 | 375 | 300 | 75 | 74.67% |
| 06번 추가 후 | REAL01~REAL06 | 450 | 360 | 90 | 77.78% |
| 07~12번 추가 후 | REAL01~REAL12 | 900 | 720 | 180 | 70.00% |

## 현재 해석

데이터가 늘어났는데 RandomForest 정확도가 70.00%까지 내려간 것은, 모델이 사람/촬영 조건 차이와 시간 흐름을 충분히 일반화하지 못한다는 신호로 본다.

특히 현재 split은 `D/F/L/R` 각도로 학습하고 `U` 각도로 검증하는 구조라서, 단순 샘플 내 분류보다 각도 일반화 난도가 높다.

## 다음 실험

GPU 사용 가능 환경에서 GRU/LSTM 계열 시계열 모델을 우선 비교한다.

## GPU Sequence Model

환경 확인:

- GPU: NVIDIA GeForce RTX 5080
- PyTorch 변경 전: `torch 2.1.2+cpu`, CUDA 사용 불가
- PyTorch 변경 후: `torch 2.11.0+cu128`, CUDA 사용 가능
- 학습 장치: `cuda`

| 실험 | 모델 | 주요 설정 | 저장 기준 | Best Epoch | Accuracy |
|---|---|---|---|---:|---:|
| sequence-01 | GRU | hidden=64, layers=1, dropout=0.1, epochs=50, batch=64 | validation best | 43 | 85.56% |
| sequence-02 | LSTM | hidden=64, layers=1, dropout=0.1, epochs=50, batch=64 | validation best | 48 | 76.67% |
| sequence-03 | GRU | hidden=128, layers=2, dropout=0.2, epochs=50, batch=64 | validation best | 43 | 85.00% |
| sequence-04 | GRU | hidden=128, layers=1, dropout=0.2, epochs=50, batch=64 | validation best | 36 | 80.00% |
| sequence-05 | CNN-GRU | conv=128, hidden=64, layers=1, dropout=0.1, epochs=50, batch=64 | validation best | 21 | 83.89% |
| sequence-06 | CNN-GRU | conv=64, hidden=64, layers=1, dropout=0.3, epochs=50, batch=64 | validation best | 50 | 86.11% |
| sequence-07 | CNN-GRU | conv=64, hidden=64, layers=1, dropout=0.3, epochs=80, batch=64 | validation best | 80 | 89.44% |
| sequence-08 | BiGRU + Attention | hidden=64, layers=1, dropout=0.3, epochs=80, batch=64 | validation best | 71 | 86.11% |
| sequence-09 | Transformer Encoder | hidden=128, layers=2, heads=4, dropout=0.2, epochs=80, batch=64 | validation best | 62 | 87.78% |

현재 최고 모델:

- 모델: CNN-GRU
- Accuracy: 89.44%
- Checkpoint: `outputs/checkpoints/sequence_model.pt`
- 보관본: `outputs/checkpoints/sequence_model_cnn_gru_c64_h64_d03_80e_best.pt`
- Metrics: `outputs/reports/sequence_metrics.json`

## 개선 결과

| 비교 | Accuracy |
|---|---:|
| RandomForest, REAL01~REAL12 | 70.00% |
| GPU GRU, REAL01~REAL12 | 85.56% |
| GPU BiGRU + Attention, REAL01~REAL12 | 86.11% |
| GPU Transformer Encoder, REAL01~REAL12 | 87.78% |
| GPU CNN-GRU, REAL01~REAL12 | 89.44% |

GPU CNN-GRU 적용 후 RandomForest 대비 `+19.44%p`, GPU GRU 대비 `+3.89%p` 개선되었다.

현재 01~12 데이터셋에서는 세 복잡 모델 중 CNN-GRU가 가장 높다. Transformer Encoder도 87.78%까지 올라갔지만, 데이터 수가 900개인 현재 조건에서는 CNN-GRU가 더 안정적인 최고 성능을 보였다.
