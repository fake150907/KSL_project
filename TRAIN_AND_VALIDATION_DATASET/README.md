# TRAIN_AND_VALIDATION_DATASET 전달 구조

이 폴더는 두 담당자가 바로 사용할 수 있도록 역할별로 정리한 전달본입니다.

```text
TRAIN_AND_VALIDATION_DATASET/
  01_모델학습_튜닝담당자용/
    train/
    validation/
    configs/
    metrics/
    README_모델학습자용.md

  02_web앱담당자용/
    models/
    labels/
    README_web앱담당자용.md
```

## 누구에게 무엇을 주면 되는가

- 모델을 새로 학습하거나 튜닝할 팀원: `01_모델학습_튜닝담당자용` 폴더 전체
- 웹앱에서 모델을 연결할 팀원: `02_web앱담당자용` 폴더 전체

## 핵심 기준

- 학습 입력 shape: `(batch, 32, 225)`
- 225 feature 구성: `pose 33 + left hand 21 + right hand 21`, 각 landmark의 `x/y/z`
- 라벨 순서: `가다`, `감사`, `괜찮다`, `배고프다`, `병원`, `아프다`, `우유`, `자다`
- 권장 웹 모델: CNN-GRU
- CNN-GRU 외부 VALIDATION 정확도: 79.33%
