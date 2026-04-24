# 다음작업 TODO

## 목적

현재 프로젝트의 다음 작업을 우선순위 기준으로 정리한 문서임.

핵심 방향은 생활형 8개 수어 표현 분류 파이프라인을 더 안정화하고, 발표 가능한 수준으로 끌어올리는 것임.

## 현재 상태 요약

- AIHub morpheme 기반 전체 라벨 분석 완료
- 고유 라벨 `1723개` 확인
- 생활형 후보 검토 후 최종 라벨 `8개` 선택 완료
- 선택 라벨 기준 target sample `75개` 생성 완료
- 현재 subset manifest `60행`
- baseline 학습 및 기본 추론 경로 확인
- Streamlit UI는 일부 동작하지만 자동 반영이 불안정함

## 우선순위별 TODO

### 1. keypoint subset 확장

목표:

- 현재 8개 라벨에 대해 더 많은 keypoint sample 확보
- 남은 keypoint zip들에서 같은 라벨 sample 추가 추출

할 일:

- 작업자 분담 기준으로 남은 keypoint zip 번호 정리
- 각 작업자가 맡은 zip에서 `selected_label_targets.csv` 기준 sample 추출
- 추출 결과를 `sample_subset_manifest.csv` 병합 가능한 형태로 수집

완료 기준:

- 현재 `60행`보다 더 큰 subset manifest 확보
- 8개 라벨 모두 sample 수 증가 확인

### 2. subset manifest 병합 및 검증

목표:

- 작업자별 subset 결과를 하나의 통합 manifest로 정리

할 일:

- 작업자별 manifest 수집
- `merge_subset_manifests.py` 로 병합
- `validate_subset_manifest.py` 로 검증
- 중복 sample_id 제거 확인

완료 기준:

- 검증 통과한 최종 `sample_subset_manifest.csv` 생성

### 3. 전처리 재실행

목표:

- 확장된 subset 기준으로 입력 tensor 다시 생성

할 일:

- `preprocess_keypoints.py` 재실행
- `sign_word_subset.npz` 갱신
- 라벨별 tensor 개수와 shape 확인

완료 기준:

- 전처리 결과 파일 정상 생성
- 학습에 바로 사용할 수 있는 상태 확인

### 4. baseline 모델 재학습

목표:

- 확장된 subset 기준으로 baseline 성능 재확인

할 일:

- `train_baseline.py` 재실행
- `baseline_metrics.json` 갱신
- confusion matrix 및 accuracy 확인

완료 기준:

- 새 checkpoint 생성
- 이전보다 sample 수 증가 상태에서 baseline 결과 확보

### 5. sequence 모델 재학습

목표:

- sequence 모델 성능도 다시 점검

할 일:

- `train_sequence.py` 재실행
- `sequence_metrics.json` 확인
- baseline과 비교

완료 기준:

- 새 sequence checkpoint 생성
- baseline 대비 장단점 정리

### 6. 실시간 인식 경로 재검증

목표:

- 실시간 예측이 더 안정적으로 뜨는지 확인

할 일:

- baseline 모델 기준 실시간 후보 확인
- 카메라 입력에서 프레임 수집 상태 확인
- CLI 실시간 추론과 UI 결과 비교

완료 기준:

- 최소한 실시간 후보가 반복적으로 안정적으로 뜨는지 확인

### 7. UI 발표 범위 정리

목표:

- 발표에서 무엇을 보여주고 무엇을 제외할지 명확히 정리

할 일:

- 현재 핵심 범위를 `생활형 8개 수어 표현 분류`로 정리
- STT/TTS는 핵심 범위가 아님을 명시
- Streamlit은 보조 시연 도구로만 취급할지 결정

완료 기준:

- 팀원들이 같은 범위 인식을 공유함

## 추천 작업 순서

```text
1. keypoint subset 확장
2. subset manifest 병합 및 검증
3. 전처리 재실행
4. baseline 재학습
5. sequence 재학습
6. 실시간 인식 재검증
7. 발표 범위 정리
```

## 바로 시작할 첫 작업

가장 먼저 할 일:

```text
남은 keypoint zip들에서 8개 생활형 라벨 sample을 더 확보하는 작업
```

이 작업이 먼저 되어야 이후 전처리, 학습, 실시간 인식 안정화가 모두 의미 있게 진행됨.
