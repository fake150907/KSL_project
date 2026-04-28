# Preprocess Report REAL02_REAL02

## 기준

- REAL range: REAL02_REAL02
- layout: mediapipe_xyz
- feature: pose33 + left_hand21 + right_hand21, [x,y,z]
- feature_count: 225
- sequence_length: 32
- normalization: shoulder_center_shoulder_width
- min_valid_frames: 4

## 산출물

- manifest: `shard_manifest_REAL02_REAL02.csv`
- npz: `mediapipe_npz_REAL02_REAL02.npz`
- failed_samples: `failed_samples_REAL02_REAL02.csv`
- label_map: `label_map.json`

## 처리 결과

- input manifest rows: 80
- success samples: 80
- failed samples added in preprocess: 0
- final NPZ X shape: (80, 32, 225)
- final NPZ y shape: (80,)

## Label count

- 가다: 25
- 자다: 15
- 우유: 10
- 배고프다: 10
- 아프다: 5
- 감사: 5
- 괜찮다: 5
- 병원: 5

## REAL count

- REAL02: 80

## 실행 명령

```powershell
python 01_make_shard_manifest.py --config config_REAL02.yaml
python 02_preprocess_mediapipe.py --config config_REAL02.yaml
python 03_validate_outputs.py --config config_REAL02.yaml
```
