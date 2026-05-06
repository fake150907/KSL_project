# CSV Repair Report — FILE04_FILE06

## 입력
- base: `C:\Users\82102_asozp43\OneDrive\바탕 화면\KSL-project-ys\KSL_project_KSL_YS`
- npz: `C:\Users\82102_asozp43\OneDrive\바탕 화면\KSL-project-ys\KSL_project_KSL_YS\File04_File06\mediapipe_npz_REAL04_REAL06.npz`
- output_dir: `C:\Users\82102_asozp43\OneDrive\바탕 화면\KSL-project-ys\KSL_project_KSL_YS\File04_File06`

## 보정 기준
- source_part: `FILE04=2-1`, `FILE05=3`, `FILE06=3-1`
- real: mp4 파일명 내부 REAL 번호 기준 (`REAL02`, `REAL03`)
- run_id: `FILE04_FILE06`
- layout: `mediapipe_xyz`
- normalization: `shoulder-center + shoulder-width scale`

## NPZ 확인
- X shape: `(145, 32, 225)`
- y shape: `(145,)`
- labels: `['가다', '감사', '괜찮다', '배고프다', '병원', '아프다', '우유', '자다']`
- sample_ids: `145`
- NaN count: `0`
- Inf count: `0`

## Manifest 보정 결과
- manifest rows: `145`
- NPZ samples: `145`
- source_part counts: `{'FILE04': 70, 'FILE06': 5, 'FILE05': 70}`
- real counts: `{'REAL02': 70, 'REAL03': 75}`
- split counts: `{'train': 120, 'val': 10, 'test': 15}`
- label counts: `{'우유': 20, '가다': 40, '배고프다': 20, '아프다': 10, '감사': 10, '자다': 25, '괜찮다': 10, '병원': 10}`
- unique WORD count: `15`

## Failed/Excluded CSV
- rows: `22355`
- 의미: MediaPipe 실패 + MVP 대상이 아니어서 제외한 샘플을 함께 기록

## 생성 파일
- `shard_manifest_FILE04_FILE06.csv`
- `shard_manifest_REAL04_REAL06.csv`
- `failed_samples_FILE04_FILE06.csv`
- `failed_samples_REAL04_REAL06.csv`
- `label_map.json`
