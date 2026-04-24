# MediaPipe 전체 인수인계 패키지 안내

이 문서는 `hand_over_mediapipe_full.zip`을 받은 팀원이 작업을 이어가기 위한 시작 안내서다.

## 패키지 목적

이 패키지는 AIHub 원본 동영상을 내려받은 뒤, MediaPipe 기준으로 landmark를 다시 추출하고, 모델을 재학습한 다음, React/Flask 웹캠 시연까지 이어가기 위한 소스와 문서를 포함한다.

이번 작업 범위는 전체 AIHub 라벨이 아니라 현재 MVP에서 사용하는 8개 라벨이다.

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

위 8개 라벨에 해당하지 않는 원본 동영상은 이번 MediaPipe 재학습 manifest에 포함하지 않는다.

보안과 용량 문제 때문에 아래 항목은 포함하지 않는다.

- AIHub API key
- AIHub 원본 데이터
- 추출된 대용량 raw 데이터
- `web/node_modules`
- 개인 PC의 가상환경

## 먼저 읽을 문서

1. `report/mediapipe_재학습_검토.md`
2. `report/mediapipe_재학습_팀원작업_지시서.md`
3. `report/python_실행환경_공유.md`
4. `report/web_history.md`

## 전체 흐름

1. Python 환경 구성
2. Node.js 패키지 설치
3. AIHub 원본 동영상 다운로드
4. 8개 대상 라벨만 골라 `data/mediapipe_video_manifest.csv` 작성
5. MediaPipe landmark NPZ 생성
6. sequence model 학습
7. Flask backend 실행
8. React web 실행
9. 웹캠 테스트

## Python 환경

```powershell
conda create -n gesture-test python=3.10 -y
conda activate gesture-test
$env:MPLCONFIGDIR="$PWD\.matplotlib"
pip install -r requirements.txt
```

GPU가 있으면 PyTorch CUDA 사용 가능 여부를 확인한다.

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 웹 환경

```powershell
cd web
npm install
npm run dev
```

## MediaPipe 전처리

`data/mediapipe_video_manifest_template.csv`를 참고해서 `data/mediapipe_video_manifest.csv`를 만든다.

```powershell
python -m src.data.preprocess_mediapipe_videos `
  --config config/mediapipe.yaml `
  --manifest data/mediapipe_video_manifest.csv `
  --output data/processed/sign_word_mediapipe_subset.npz `
  --sequence_length 32 `
  --frame_step 1
```

## 학습

```powershell
python -m src.models.train_sequence `
  --config config/mediapipe.yaml `
  --model_type cnn_gru `
  --epochs 80 `
  --batch_size 64
```

## 웹캠 시연

Backend:

```powershell
python backend/app.py
```

Frontend:

```powershell
cd web
npm run dev
```

브라우저에서 Vite가 출력한 주소로 접속한다. 보통 `http://127.0.0.1:3001` 또는 `http://127.0.0.1:5173`이다.

## 주의

- API key는 문서나 zip에 저장하지 않는다.
- 원본 동영상 경로는 프로젝트 루트 기준 상대 경로를 권장한다.
- MediaPipe 재학습 결과는 기존 OpenPose 기반 checkpoint와 직접 비교하되, 입력 좌표계가 다르다는 점을 반드시 기록한다.
