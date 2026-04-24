# 8개 MVP AI 작업자용 지시문

## 가장 먼저 확인할 전제

이번 작업은 기존 8개 MVP 라벨 기준이다.

사용 라벨 파일:

```text
data/selected_labels_small.json
```

사용 target 파일:

```text
data/selected_label_targets.csv
```

AI Hub WORD 데이터 구조상 morpheme은 `01_real_word_morpheme` 하나이고, keypoint는 `01_real_word_keypoint.zip`부터 `16_real_word_keypoint.zip`까지 나뉘어 있다.

따라서 keypoint를 찾을 때 `sample_id` 전체를 완전일치로 비교하면 안 된다.

매칭 기준은 아래처럼 잡는다.

```text
WORD번호 + angle
```

예:

```text
KEYPOINT:
NIA_SL_WORD0943_REAL02_F_000000000000_keypoints.json

MORPHEME:
NIA_SL_WORD0943_REAL01_F_morpheme.json

MATCH KEY:
WORD0943_F

LABEL:
가다
```

즉 `REAL02`, `REAL03` keypoint도 `WORD번호 + angle`이 같으면 `REAL01 morpheme`의 라벨을 붙일 수 있다.

중요:

```text
REAL번호는 매칭할 때만 무시한다.
WORD번호 + angle이 같으면 같은 단어/같은 카메라각도 동작으로 보고,
REAL01 morpheme의 start/end/duration 구간 정보도 그대로 사용한다.
```

## 작업 목표

기존 8개 라벨의 `WORD번호 + angle`을 기준으로, 담당 keypoint zip에서 해당 keypoint folder만 추출한다.

전체 keypoint를 전부 압축 해제하거나 학습하지 않는다.

## 기존 8개 라벨

```text
가다
감사
괜찮다
배고프다
병원
아프다
우유
자다
```

## 사용할 keypoint zip

담당자가 맡은 keypoint zip을 사용한다.

예:

```text
01_real_word_keypoint.zip
02_real_word_keypoint.zip
03_real_word_keypoint.zip
...
16_real_word_keypoint.zip
```

주의:

```text
01 zip은 sample_id 완전일치로도 대부분 맞는다.
02~16 zip은 sample_id 완전일치가 아니라 WORD번호 + angle 기준으로 매칭해야 한다.
```

## 산출물

작업자는 중간 산출물로 아래 두 가지를 생성해서 전달한다.

```text
selected_keypoints_8labels/
sample_subset_manifest_8labels.csv
```

최종 학습용 manifest는 통합 담당자가 병합 후 아래 이름으로 정리한다.

```text
data/sample_subset_manifest.csv
```

## 실행 예시

프로젝트 루트에서 실행한다.

```powershell
C:\Users\Samsung\anaconda3\envs\gesture-test\python.exe -m src.data.extract_keypoint_subset_from_zip `
  --config config/default.yaml `
  --match_mode "word_angle" `
  --zip_path "data/raw/aihub_downloads/word_keypoint_01/004.수어영상/1.Training/라벨링데이터/REAL/WORD/01_real_word_keypoint.zip" `
  --output_dir "data/raw/selected_keypoints_8labels" `
  --manifest_output "data/sample_subset_manifest_8labels.csv"
```

기존 8개 라벨 작업자는 `--selected_labels_path` 옵션을 쓰지 않는다.

기본 설정이 아래 파일을 사용한다.

```text
data/selected_labels_small.json
```

## 검증

```powershell
C:\Users\Samsung\anaconda3\envs\gesture-test\python.exe scripts/validate_subset_manifest.py `
  --manifest data/sample_subset_manifest_8labels.csv `
  --check_paths
```

성공 기준:

```text
[OK] Manifest validation passed.
```

## AI 작업자에게 다시 강조할 것

절대 `NIA_SL_WORDxxxx_REAL01_D`와 `NIA_SL_WORDxxxx_REAL02_D`를 완전일치 비교해서 버리면 안 된다.

반드시 `WORD번호 + angle` 기준으로 매칭한다.

그리고 매칭 후에는 `start/end/duration`을 비우지 않는다.

우리는 `REAL번호`가 아니라 `WORD번호 + angle`을 기준으로 같은 제스처라고 판단한다.
