# Preprocess/Merge Report — File04_File05_File06

## 1. 실행 정보
- Generated at: `2026-04-29 12:43:11`
- Output ID: `File04_File05_File06`
- Expected labels: `['가다', '감사', '괜찮다', '배고프다', '병원', '아프다', '우유', '자다']`
- Layout: `mediapipe_xyz`
- Normalization: `shoulder-center + shoulder-width scale`
- Sequence length / feature dim: `32` / `225`

## 2. 입력 폴더
- `File04` → `C:\Users\82102_asozp43\OneDrive\바탕 화면\REAL_PIPELINE_CMD_CLEAN_V3\handoff_parts\File04`
- `File05` → `C:\Users\82102_asozp43\OneDrive\바탕 화면\REAL_PIPELINE_CMD_CLEAN_V3\handoff_parts\File05`
- `File06` → `C:\Users\82102_asozp43\OneDrive\바탕 화면\REAL_PIPELINE_CMD_CLEAN_V3\handoff_parts\File06`

## 3. Part별 요약
| part | samples | X shape | splits | npz | manifest |
|---|---:|---|---|---|---|
| `File04` | 70 | `(70, 32, 225)` | `{'train': 65, 'val': 5}` | `mediapipe_npz_File04.npz` | `shard_manifest_File04.csv` |
| `File05` | 5 | `(5, 32, 225)` | `{'train': 5}` | `mediapipe_npz_File05.npz` | `shard_manifest_File05.csv` |
| `File06` | 70 | `(70, 32, 225)` | `{'train': 50, 'test': 15, 'val': 5}` | `mediapipe_npz_File06.npz` | `shard_manifest_File06.csv` |

## 4. Master 요약
- X shape: `(145, 32, 225)`
- y shape: `(145,)`
- sample_ids: `145`
- splits: `{'train': 120, 'val': 10, 'test': 15}`
- label counts: `{'우유': 20, '가다': 40, '배고프다': 20, '아프다': 10, '감사': 10, '자다': 25, '괜찮다': 10, '병원': 10}`
- manifest rows: `145`
- failed rows: `22355`

## 5. 이슈
- ERROR 없음

## 6. 완료 전 체크
- 모든 shard의 NPZ shape가 (N, 32, 225)인지 확인
- manifest row 수와 NPZ sample 수 일치 확인
- MVP 외 단어가 섞이지 않았는지 확인
- sample_id 중복 확인
- failed sample 병합 완료
- label_map이 학습/평가/웹에서 동일하게 사용 가능
