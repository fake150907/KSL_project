# 새 PC 프로젝트 이관 지시서

## 현재 작업 상태

이 패키지는 AIHub 수어 프로젝트를 새 PC에서 이어서 개발하기 위한 이관용 묶음이다.

현재 기준 작업 상태는 다음과 같다.

```text
라벨 기준: 기존 8개 MVP 라벨
사용 keypoint: REAL01 + REAL02 + REAL03 + REAL04 + REAL05 + REAL06
최종 manifest: data/sample_subset_manifest.csv
전체 샘플: 450건
전처리 파일: data/processed/sign_word_subset.npz
학습 모델: outputs/checkpoints/baseline.joblib
검증 정확도: 77.78%
```

8개 라벨은 다음과 같다.

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

## 중요한 데이터 매칭 원칙

AIHub WORD 데이터에서 현재 로컬 morpheme은 `REAL01` 기준만 있다.

하지만 keypoint는 `REAL01`, `REAL02`, `REAL03`처럼 여러 REAL 번호로 나뉜다.

이 프로젝트에서는 다음 기준으로 매칭한다.

```text
REAL 번호는 매칭에서 무시한다.
WORD번호 + 카메라각도(F/L/R/U/D)를 기준으로 매칭한다.
```

예:

```text
MORPHEME:
NIA_SL_WORD0002_REAL01_D_morpheme.json

KEYPOINT:
NIA_SL_WORD0002_REAL02_D_000000000000_keypoints.json

MATCH KEY:
WORD0002_D
```

`WORD번호 + angle`이 같으면 같은 단어/같은 카메라각도 동작으로 보고, morpheme의 `label`, `start`, `end`, `duration` 정보를 사용한다.

## 이관 zip에 포함된 주요 파일

```text
src/
scripts/
config/
report/
backend/
web/                         단, node_modules 제외
data/*.csv
data/*.json
data/*.xlsx
data/processed/
data/raw/selected_keypoints_01/
data/raw/selected_keypoints_02/
data/raw/selected_keypoints_03/
data/raw/selected_keypoints_04/
data/raw/selected_keypoints_05/
data/raw/selected_keypoints_06/
outputs/checkpoints/
outputs/reports/
outputs/confusion_matrix.png
README.md
requirements.txt
run_ui.py
run.bat
```

아래 원본 대용량 파일은 포함하지 않는다.

```text
data/raw/aihub_downloads/
data/raw/debug_selected_keypoints_*/
web/node_modules/
__pycache__/
```

원본 AIHub zip/tar는 새 PC에서 꼭 필요하지 않다. 현재 MVP 학습/검증/웹 실행은 추출된 subset만으로 가능하다.

## 새 PC에서 압축 해제

원하는 작업 폴더에 zip을 푼다.

권장 위치:

```text
C:\github\ai-project-01
```

다른 위치에 풀어도 되지만, 명령어 실행 시 프로젝트 루트에서 실행해야 한다.

## Python 환경 준비

기존 환경명은 `gesture-test`를 사용했다.

새 PC에서도 같은 이름을 권장한다.

```powershell
conda create -n gesture-test python=3.10 -y
conda activate gesture-test
python -m pip install -r requirements.txt
```

추가 패키지가 부족하면 아래도 설치한다.

```powershell
python -m pip install scikit-learn joblib matplotlib pandas numpy PyYAML streamlit
```

## 현재 데이터 검증

프로젝트 루트에서 실행한다.

```powershell
C:\Users\Samsung\anaconda3\envs\gesture-test\python.exe scripts/validate_subset_manifest.py `
  --manifest data/sample_subset_manifest.csv `
  --check_paths
```

성공 기준:

```text
[OK] Manifest validation passed.
rows: 450
```

새 PC의 conda 설치 경로가 다르면 `python.exe` 경로를 새 PC에 맞게 바꾼다.

예:

```powershell
python scripts/validate_subset_manifest.py --manifest data/sample_subset_manifest.csv --check_paths
```

## 전처리 다시 생성

이미 `data/processed/sign_word_subset.npz`가 포함되어 있지만, 환경 확인을 위해 다시 만들어도 된다.

```powershell
python -m src.data.preprocess_keypoints --config config/default.yaml --subset_only
```

정상 결과 예:

```text
shape: (450, 32, 201)
train_count: 360
validation_count: 90
```

## 모델 재학습

현재 학습 모델은 이미 포함되어 있다.

```text
outputs/checkpoints/baseline.joblib
```

다시 학습하려면 다음 명령을 실행한다.

```powershell
python -m src.models.train_baseline --config config/default.yaml
```

현재 기준 결과:

```text
validation accuracy: 0.7778
model path: outputs/checkpoints/baseline.joblib
```

## Streamlit UI 실행

Streamlit UI는 아래 명령으로 실행한다.

```powershell
python -m streamlit run src/ui/app.py
```

또는 프로젝트에 있는 실행 스크립트를 사용할 수 있다.

```powershell
.\scripts\run_streamlit_ui.ps1
```

## React 웹 실행이 필요한 경우

`web/node_modules`는 이관 zip에서 제외되어 있다.

새 PC에서 최초 1회 설치한다.

```powershell
cd web
npm install
npm run dev
```

## 이후 개발 순서

새 PC에서 이어갈 다음 작업은 아래 순서가 좋다.

1. manifest 검증
2. 전처리 재생성
3. baseline 모델 재학습
4. Streamlit 화면에서 결과 확인
5. 필요하면 `07_real_word_keypoint.zip`부터 추가 다운로드
6. 같은 `WORD번호 + angle` 로직으로 keypoint subset 추가
7. manifest 병합 후 재학습

## 07번 이후 keypoint 추가 시 주의

새 keypoint zip을 추가할 때는 반드시 `word_angle` 매칭을 사용한다.

예:

```powershell
python -m src.data.extract_keypoint_subset_from_zip `
  --config config/default.yaml `
  --match_mode word_angle `
  --zip_path "data/raw/aihub_downloads/word_keypoint_07/004.수어영상/1.Training/라벨링데이터/REAL/WORD/07_real_word_keypoint.zip" `
  --output_dir "data/raw/selected_keypoints_07" `
  --manifest_output "data/sample_subset_manifest_07.csv"
```

그 다음 `data/sample_subset_manifest.csv`에 병합하고 전처리/재학습을 다시 진행한다.

## 포함된 작업자용 파일

239개 라벨 작업자용 파일도 포함되어 있다.

```text
data/selected_labels_small_label239.json
data/selected_label_targets_label239.csv
data/selected_lifestyle_candidates_label239.csv
report/label239_AI작업자_지시문.md
```

하지만 현재 내가 직접 학습한 모델은 기존 8개 라벨 기준이다.

239개 라벨은 팀원 작업 또는 후속 확장용으로 구분해서 사용한다.

## 최종 확인 경로

현재 이관 기준 핵심 경로는 다음과 같다.

```text
data/sample_subset_manifest.csv
data/processed/sign_word_subset.npz
outputs/checkpoints/baseline.joblib
outputs/reports/baseline_metrics.json
outputs/confusion_matrix.png
```
