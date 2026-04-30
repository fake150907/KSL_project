# MediaPipe 재학습 팀원 작업 지시서

## 작업 목표

AIHub 원본 동영상에서 MediaPipe Holistic landmark를 다시 추출해서 학습용 `.npz`를 만든다.

기존 모델은 AIHub keypoint 기반이고, 웹캠 시연은 MediaPipe 기반이다. 이번 작업은 학습과 실시간 입력의 좌표계를 MediaPipe로 통일하기 위한 것이다.

## 작업 대상 라벨

이번 MediaPipe 재학습 작업은 전체 AIHub 라벨이 아니라, 현재 MVP 모델에서 사용 중인 **8개 라벨만** 대상으로 한다.

대상 라벨:

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

팀원은 본인 담당 데이터가 있더라도 위 8개 라벨에 해당하는 원본 동영상만 manifest에 포함한다. 위 목록에 없는 라벨은 이번 MediaPipe 재학습 manifest와 NPZ에 넣지 않는다.

## 팀원이 준비할 것

1. 프로젝트 소스
2. 8개 대상 라벨 중 담당 라벨의 원본 동영상
3. 각 동영상의 정답 라벨과 구간 정보
4. Python 실행 환경

API key나 개인 인증 정보는 결과 파일에 기록하지 않는다.

## 개발 환경

권장 Python 버전:

```powershell
conda create -n gesture-test python=3.10 -y
conda activate gesture-test
$env:MPLCONFIGDIR="$PWD\.matplotlib"
```

공통 패키지:

```powershell
pip install -r requirements.txt
```

MediaPipe/OpenCV 버전 충돌이 있으면 아래 조합을 우선 사용한다.

```powershell
pip install numpy==1.26.4 scikit-learn==1.3.1
pip install opencv-python==4.11.0.86 opencv-contrib-python==4.11.0.86 mediapipe==0.10.13
```

CPU만 있는 팀원:

```powershell
pip install torch
```

NVIDIA GPU가 있는 팀원:

```powershell
pip install torch
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

`True`가 나오면 학습 때 자동으로 GPU를 사용한다. `False`가 나오면 CPU로 진행해도 된다.

## 1. manifest 만들기

`data/mediapipe_video_manifest_template.csv`를 복사해서 `data/mediapipe_video_manifest.csv`를 만든다.

필수 컬럼:

```csv
sample_id,label,video_path,start,end,duration,split
```

예시:

```csv
NIA_SL_WORD0000_REAL01_D,감사,data/raw/videos/NIA_SL_WORD0000_REAL01_D.mp4,0.0,2.0,4.0,train
```

규칙:

- `sample_id`: 동영상 또는 AIHub 샘플 ID와 맞춘다.
- `label`: 반드시 8개 대상 라벨 중 하나만 사용한다.
- `video_path`: 프로젝트 루트 기준 상대 경로를 권장한다.
- `start/end/duration`: 초 단위 또는 기존 morpheme 기준값을 사용한다.
- `split`: 검증용 각도나 담당 기준이 있으면 `validation`, 나머지는 `train`.

## 2. MediaPipe landmark 추출

```powershell
python -m src.data.preprocess_mediapipe_videos `
  --config config/mediapipe.yaml `
  --manifest data/mediapipe_video_manifest.csv `
  --output data/processed/sign_word_mediapipe_subset.npz `
  --sequence_length 32 `
  --frame_step 1
```

결과 파일:

- `data/processed/sign_word_mediapipe_subset.npz`
- `data/processed/sign_word_mediapipe_subset.meta.json`
- 실패 샘플이 있으면 `data/processed/sign_word_mediapipe_subset.failed.csv`

실패 샘플이 있으면 `failed.csv`의 `video_path`, `error`를 확인한다.

## 3. 학습 실행

기본 추천 모델:

```powershell
python -m src.models.train_sequence `
  --config config/mediapipe.yaml `
  --model_type cnn_gru `
  --epochs 80 `
  --batch_size 64
```

GPU 메모리가 충분하면 비교 모델도 실행한다.

```powershell
python -m src.models.train_sequence `
  --config config/mediapipe.yaml `
  --model_type bigru_attention `
  --epochs 80 `
  --batch_size 64
```

```powershell
python -m src.models.train_sequence `
  --config config/mediapipe.yaml `
  --model_type transformer `
  --epochs 80 `
  --batch_size 64
```

CPU 팀원은 먼저 `--epochs 5`로 정상 동작만 확인하고, 최종 학습은 GPU 팀원에게 넘겨도 된다.

## 4. 제출할 산출물

필수:

- `data/mediapipe_video_manifest.csv`
- `data/processed/sign_word_mediapipe_subset.npz`
- `data/processed/sign_word_mediapipe_subset.meta.json`
- `outputs/reports/sequence_metrics.json`
- `outputs/checkpoints/sequence_model.pt`

실패가 있다면 함께 제출:

- `data/processed/sign_word_mediapipe_subset.failed.csv`

## 팀원 AI에게 줄 지시문

아래 문장을 팀원의 AI에게 그대로 전달한다.

```text
너는 이 프로젝트의 MediaPipe 재학습 데이터 생성 담당이다.

목표:
8개 대상 라벨의 원본 수어 동영상에서 MediaPipe Holistic landmark를 추출하여 기존 학습 코드가 읽을 수 있는 NPZ를 만든다.

대상 라벨:
가다, 감사, 괜찮다, 배고프다, 병원, 아프다, 우유, 자다

반드시 확인할 파일:
- report/mediapipe_재학습_팀원작업_지시서.md
- src/data/preprocess_mediapipe_videos.py
- config/mediapipe.yaml
- data/mediapipe_video_manifest_template.csv

해야 할 일:
1. Python 3.10 환경을 만들고 requirements.txt 패키지를 설치한다.
2. 8개 대상 라벨에 해당하는 원본 동영상 경로만 사용해 data/mediapipe_video_manifest.csv를 작성한다.
3. python -m src.data.preprocess_mediapipe_videos 명령으로 MediaPipe NPZ를 생성한다.
4. 생성된 meta.json에서 shape, label 목록, failed_rows를 확인한다.
5. 가능하면 src.models.train_sequence으로 cnn_gru 모델을 80 epoch 학습한다.
6. CPU 환경이면 5 epoch smoke test만 실행해도 되고, GPU 환경이면 80 epoch 학습까지 진행한다.
7. 산출물과 실패 로그를 정리해서 전달한다.

주의:
- API key, 비밀번호, 개인 인증 정보는 어떤 파일에도 쓰지 않는다.
- video_path가 깨지지 않도록 프로젝트 루트 기준 상대 경로를 우선 사용한다.
- MediaPipe landmark는 pose25 + left_hand21 + right_hand21 고정 슬롯이다.
- 대상 라벨 8개 밖의 데이터는 이번 작업에 포함하지 않는다.
```
