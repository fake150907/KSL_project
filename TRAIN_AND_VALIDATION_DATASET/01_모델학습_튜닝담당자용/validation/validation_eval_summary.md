# VALIDATION MVP Full Evaluation

created_at: 2026-04-29 14:25:53 +09:00

## 목적

학습에 사용하지 않은 AIHub VALIDATION 비디오에서 8개 MVP 단어 영상을 추출해, 현재 WEB 전달용 모델 2개(LSTM, CNN-GRU)의 실제 validation 성능을 확인했다.

## 검증 데이터

- source tar: `data/raw/aihub_downloads/validation_word_video_wsl/download.tar`
- merged zip: `data/raw/aihub_downloads/validation_word_video_wsl/01_real_word_video.zip`
- extracted mp4: `data/raw/videos/validation_mvp_full`
- morpheme manifest: `data/validation_word_morpheme_manifest_8labels.csv`
- output npz: `team_handover_outputs/VALIDATION_MVP_FULL/mediapipe_npz_VALIDATION_MVP_FULL.npz`
- samples: 150
- shape: `(150, 32, 225)`
- failed samples: 0

## Label Counts

- 가다: 40
- 감사: 10
- 괜찮다: 10
- 배고프다: 20
- 병원: 10
- 아프다: 10
- 우유: 20
- 자다: 30

## Results

- LSTM: 70 / 150 correct, accuracy 0.4667
- CNN-GRU: 82 / 150 correct, accuracy 0.5467

## 산출물

- LSTM metrics: `team_handover_outputs/VALIDATION_EVAL_FULL/lstm_FILE01_03-FILE10_12_on_validation_full_metrics.json`
- LSTM predictions: `team_handover_outputs/VALIDATION_EVAL_FULL/lstm_FILE01_03-FILE10_12_on_validation_full_predictions.csv`
- CNN-GRU metrics: `team_handover_outputs/VALIDATION_EVAL_FULL/cnn_gru_FILE01_03-FILE10_12_on_validation_full_metrics.json`
- CNN-GRU predictions: `team_handover_outputs/VALIDATION_EVAL_FULL/cnn_gru_FILE01_03-FILE10_12_on_validation_full_predictions.csv`

## 해석 메모

- 기존 학습 내부 validation 정확도는 train 파일 묶음 안에서 나눈 검증 점수다.
- 이 결과는 별도의 AIHub VALIDATION 비디오 150개로 평가한 외부 검증 점수다.
- 따라서 발표나 강사 점검에는 이 값을 “학습에 섞지 않은 실제 validation video 평가”로 설명하는 것이 더 정직하다.
- 현재 기준으로 CNN-GRU가 LSTM보다 validation video에서 더 높다.
