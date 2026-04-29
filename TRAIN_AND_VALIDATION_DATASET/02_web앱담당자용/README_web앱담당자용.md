# web앱 담당자용 README

이 폴더는 웹앱에서 학습된 수어 인식 모델을 연결하는 팀원이 사용하는 전달본입니다.

## 폴더 구조

```text
02_web앱담당자용/
  models/  웹 추론에 사용할 PyTorch 모델 파일
  labels/  모델 출력 index를 단어로 바꾸는 라벨 매핑
```

## 권장 모델

```text
models/sequence_model_cnn_gru_FILE01_16_FULL_valacc1.0000_ext0.7933.pt
```

- 모델 종류: CNN-GRU
- 내부 holdout 최고 정확도: 100.00%
- 외부 VALIDATION 정확도: 79.33%
- 현재 웹앱 적용 권장 모델입니다.

## 비교 모델

```text
models/sequence_model_lstm_FILE01_16_FULL_valacc0.9268_ext0.6267.pt
```

- 모델 종류: LSTM
- 내부 holdout 최고 정확도: 92.68%
- 외부 VALIDATION 정확도: 62.67%

## 필수 라벨 파일

```text
labels/model_labels.json
```

웹앱은 모델 출력 index를 이 파일의 순서대로 단어로 변환해야 합니다.

```json
["가다", "감사", "괜찮다", "배고프다", "병원", "아프다", "우유", "자다"]
```

예를 들어 모델 출력 index가 `6`이면 단어는 `우유`입니다. 이 순서를 바꾸면 웹 화면에 표시되는 예측 단어가 틀어집니다.

## 모델 입력 형식

```text
(batch, 32, 225)
```

- 32: 한 번 예측할 때 사용하는 프레임 수
- 225: MediaPipe landmark feature 수
- feature 구성: `pose33 + left_hand21 + right_hand21`, 각 landmark의 `[x, y, z]`

## MediaPipe 좌표 사용 범위

웹앱도 학습 때와 같은 방식으로 MediaPipe 좌표를 만들어야 합니다.

사용해야 하는 좌표:

```text
pose 33개 + left hand 21개 + right hand 21개
각 landmark의 x/y/z 3개 값 사용
(33 + 21 + 21) * 3 = 225
```

사용하지 않는 좌표/정보:

- face mesh 468개
- iris 좌표
- segmentation mask
- 원본 RGB 프레임
- pose visibility/presence confidence

웹앱에서 landmark 추출 순서나 누락 처리 방식이 학습 때와 달라지면 정확도가 떨어질 수 있습니다.

## 적용 주의사항

- 모델 파일명은 길어도 그대로 보관하는 것을 권장합니다. 모델 종류와 정확도가 파일명에 들어 있습니다.
- 서비스 코드에서 꼭 `sequence_model.pt`라는 이름이 필요하면, 권장 CNN-GRU 모델을 복사해서 그 이름으로 바꾸면 됩니다.
- 실사용 설명에는 내부 100.00%보다 외부 VALIDATION 79.33%를 기준으로 말하는 편이 안전합니다.
