# Preprocess/Merge Report — REAL02-1_REAL03_REAL03-1

## 1. 실행 정보
- Generated at: `2026-04-28 22:56:14`
- Output ID: `REAL02-1_REAL03_REAL03-1`
- Expected labels: `['가다', '감사', '괜찮다', '배고프다', '병원', '아프다', '우유', '자다']`
- Layout: `mediapipe_xyz`
- Normalization: `shoulder-center + shoulder-width scale`
- Sequence length / feature dim: `32` / `225`

## 2. 입력 폴더
- `REAL02-1` → `C:\Users\82102_asozp43\OneDrive\바탕 화면\REAL_PIPELINE_CMD_CLEAN_V3\handoff_parts\REAL02-1`
- `REAL03` → `C:\Users\82102_asozp43\OneDrive\바탕 화면\REAL_PIPELINE_CMD_CLEAN_V3\handoff_parts\REAL03`
- `REAL03-1` → `C:\Users\82102_asozp43\OneDrive\바탕 화면\REAL_PIPELINE_CMD_CLEAN_V3\handoff_parts\REAL03-1`

## 3. Part별 요약
| part | samples | X shape | splits | npz | manifest |
|---|---:|---|---|---|---|
| `REAL02-1` | 40 | `(40, 32, 225)` | `{'train': 25, 'test': 10, 'val': 5}` | `mediapipe_npz_REAL02-1.npz` | `shard_manifest_REAL02-1.csv` |
| `REAL03` | 40 | `(40, 32, 225)` | `{'train': 35, 'val': 5}` | `mediapipe_npz_REAL03.npz` | `shard_manifest_REAL03.csv` |
| `REAL03-1` | 40 | `(40, 32, 225)` | `{'train': 30, 'test': 5, 'val': 5}` | `mediapipe_npz_REAL03-1.npz` | `shard_manifest_REAL03-1.csv` |

## 4. Master 요약
- X shape: `(120, 32, 225)`
- y shape: `(120,)`
- sample_ids: `120`
- splits: `{'train': 90, 'test': 15, 'val': 15}`
- label counts: `{'가다': 15, '감사': 15, '괜찮다': 15, '배고프다': 15, '병원': 15, '아프다': 15, '우유': 15, '자다': 15}`
- manifest rows: `120`
- failed rows: `22380`

## 5. 이슈
- ERROR 없음

## 6. 완료 전 체크
- 모든 shard의 NPZ shape가 (N, 32, 225)인지 확인
- manifest row 수와 NPZ sample 수 일치 확인
- MVP 외 단어가 섞이지 않았는지 확인
- sample_id 중복 확인
- failed sample 병합 완료
- label_map이 학습/평가/웹에서 동일하게 사용 가능
