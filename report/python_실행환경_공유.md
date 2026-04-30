# Python 실행환경 공유

## 목적

새 PC에서 이 프로젝트를 같은 기준으로 재현할 수 있도록, 현재 확인된 Python 실행 환경과 설치 순서를 정리한다.

## 현재 기준 실행 환경

- 프로젝트 경로: `C:\github\ai-project-01`
- 권장 가상환경 이름: `gesture-test`
- 현재 PC의 Python 경로: `C:\Users\shemanul\anaconda3\envs\gesture-test\python.exe`
- Python 버전: `3.10.20`
- 기준 확인일: `2026-04-26`

이전 PC의 경로는 `C:\Users\Samsung\anaconda3\envs\gesture-test\python.exe`였지만, 새 PC에서는 사용자명이 다르므로 위 경로를 기준으로 사용한다.

## 참고 파일

- 프로젝트 요구사항 기준: [requirements.txt](C:/github/ai-project-01/requirements.txt)
- 현재 설치 패키지 목록: [python_실행환경_패키지목록.txt](C:/github/ai-project-01/report/python_%EC%8B%A4%ED%96%89%ED%99%98%EA%B2%BD_%ED%8C%A8%ED%82%A4%EC%A7%80%EB%AA%A9%EB%A1%9D.txt)
- Markdown 패키지 목록: [python_실행환경_패키지목록.md](C:/github/ai-project-01/report/python_%EC%8B%A4%ED%96%89%ED%99%98%EA%B2%BD_%ED%8C%A8%ED%82%A4%EC%A7%80%EB%AA%A9%EB%A1%9D.md)

## 중요한 주의사항

- 기본 `python`은 사용하지 않는 것이 좋다.
- 새 PC에서 확인된 기본 `python`은 `Python 3.13.9`였다.
- 이 프로젝트는 `Python 3.10.20`의 `gesture-test` 환경 기준으로 맞춘다.
- 기존 `outputs/checkpoints/baseline.joblib`은 `scikit-learn 1.3.1`로 저장된 모델이다.
- `scikit-learn 1.7.2`에서는 기존 `baseline.joblib` 예측 시 `DecisionTreeClassifier` 속성 오류가 발생했다.
- 따라서 기존 모델을 그대로 재사용하려면 `scikit-learn==1.3.1`을 사용한다.
- `numpy==1.26.4`와 호환되도록 `opencv-python`, `opencv-contrib-python`, `mediapipe`도 아래 버전으로 맞춘다.

## 새 PC 환경 세팅 순서

프로젝트 루트에서 실행한다.

```powershell
cd C:\github\ai-project-01
```

가상환경을 만든다.

```powershell
conda create -n gesture-test python=3.10 -y
```

기본 요구 패키지를 설치한다.

```powershell
C:\Users\shemanul\anaconda3\envs\gesture-test\python.exe -m pip install -r requirements.txt
```

기존 모델 호환을 위해 핵심 버전을 맞춘다.

```powershell
C:\Users\shemanul\anaconda3\envs\gesture-test\python.exe -m pip install scikit-learn==1.3.1 numpy==1.26.4
C:\Users\shemanul\anaconda3\envs\gesture-test\python.exe -m pip install opencv-python==4.11.0.86 opencv-contrib-python==4.11.0.86 mediapipe==0.10.13
C:\Users\shemanul\anaconda3\envs\gesture-test\python.exe -m pip install grpcio-status==1.62.3
C:\Users\shemanul\anaconda3\envs\gesture-test\python.exe -m pip install torch==2.1.2
```

Matplotlib 캐시 경로를 프로젝트 내부로 고정한다.

```powershell
New-Item -ItemType Directory -Force -Path C:\github\ai-project-01\.matplotlib
conda env config vars set -n gesture-test MPLCONFIGDIR=C:\github\ai-project-01\.matplotlib
```

## 권장 실행 방식

가장 안전한 방식은 Python 경로를 직접 지정해서 실행하는 것이다.

Manifest 검증:

```powershell
C:\Users\shemanul\anaconda3\envs\gesture-test\python.exe scripts/validate_subset_manifest.py --manifest data/sample_subset_manifest.csv --check_paths
```

전처리 재생성:

```powershell
C:\Users\shemanul\anaconda3\envs\gesture-test\python.exe -m src.data.preprocess_keypoints --config config/default.yaml --subset_only
```

기준 모델 재학습:

```powershell
C:\Users\shemanul\anaconda3\envs\gesture-test\python.exe -m src.models.train_baseline --config config/default.yaml
```

Streamlit 실행:

```powershell
C:\Users\shemanul\anaconda3\envs\gesture-test\python.exe -m streamlit run src/ui/app.py
```

## 설치 후 검증 명령

패키지 충돌 확인:

```powershell
C:\Users\shemanul\anaconda3\envs\gesture-test\python.exe -m pip check
```

핵심 라이브러리 버전 확인:

```powershell
C:\Users\shemanul\anaconda3\envs\gesture-test\python.exe -c "import numpy, pandas, sklearn, streamlit, cv2, mediapipe, torch, google.protobuf; print(numpy.__version__, pandas.__version__, sklearn.__version__, streamlit.__version__, cv2.__version__, mediapipe.__version__, torch.__version__, google.protobuf.__version__)"
```

기존 모델 로드 확인:

```powershell
C:\Users\shemanul\anaconda3\envs\gesture-test\python.exe -c "import joblib, numpy as np; obj=joblib.load('outputs/checkpoints/baseline.joblib'); d=np.load('data/processed/sign_word_subset.npz', allow_pickle=True); pred=obj['model'].predict(d['X'][:3].reshape(3, -1)); print('model_loaded'); print(pred.tolist()); print(obj['labels'])"
```

## 현재 설치된 주요 라이브러리

| 라이브러리 | 버전 |
|---|---|
| `Python` | `3.10.20` |
| `numpy` | `1.26.4` |
| `pandas` | `2.3.3` |
| `scikit-learn` | `1.3.1` |
| `scipy` | `1.15.3` |
| `matplotlib` | `3.10.9` |
| `joblib` | `1.5.3` |
| `PyYAML` | `6.0.3` |
| `mediapipe` | `0.10.13` |
| `opencv-python` | `4.11.0.86` |
| `opencv-contrib-python` | `4.11.0.86` |
| `streamlit` | `1.56.0` |
| `torch` | `2.1.2+cpu` |
| `protobuf` | `4.25.9` |
| `grpcio-status` | `1.62.3` |

## 현재 확인된 상태

- `pip check`: `No broken requirements found.`
- Manifest 검증: `[OK] Manifest validation passed.`
- Manifest row 수: `225`
- 기존 `baseline.joblib` 로드 및 샘플 예측 성공

## 팀원에게 전달할 핵심 요약

- 이 프로젝트는 `gesture-test` 환경 기준으로 실행한다.
- 기본 `python` 대신 `C:\Users\shemanul\anaconda3\envs\gesture-test\python.exe`를 직접 지정한다.
- 기존 모델 재사용을 위해 `scikit-learn==1.3.1`, `numpy==1.26.4`를 유지한다.
- 추가 학습이나 Streamlit 실행 전에 manifest 검증을 먼저 돌린다.
