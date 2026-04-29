# Preprocess Report

file_tag: VALIDATION_MVP_FULL
created_at: 2026-04-29 14:25:09
manifest: C:\github\ai-project-01\team_handover_outputs\VALIDATION_MVP_FULL\shard_manifest_VALIDATION_MVP_FULL.csv
npz: C:\github\ai-project-01\team_handover_outputs\VALIDATION_MVP_FULL\mediapipe_npz_VALIDATION_MVP_FULL.npz
failed_samples: C:\github\ai-project-01\team_handover_outputs\VALIDATION_MVP_FULL\failed_samples_VALIDATION_MVP_FULL.csv
meta_json: C:\github\ai-project-01\team_handover_outputs\VALIDATION_MVP_FULL\mediapipe_npz_VALIDATION_MVP_FULL.meta.json
manifest_status: updated from NPZ sample_ids

fixed_standard:
- layout: mediapipe_xyz
- feature_count: 225
- sequence_length: 32
- normalization: shoulder-center + shoulder-width scale
- extraction_basis: FILE zip number, not REAL signer number

command:
PYTHONPATH=C:\github\ai-project-01\team_handover_2026-04-29\REAL_SHARD_WORKER python -m src.data.preprocess_mediapipe_videos --config C:\github\ai-project-01\team_handover_2026-04-29\REAL_SHARD_WORKER\config\mediapipe_real01_xyz_body.yaml --manifest C:\github\ai-project-01\team_handover_outputs\VALIDATION_MVP_FULL\shard_manifest_VALIDATION_MVP_FULL.csv --output C:\github\ai-project-01\team_handover_outputs\VALIDATION_MVP_FULL\mediapipe_npz_VALIDATION_MVP_FULL.npz --landmark-layout mediapipe_xyz --sequence_length 32
python C:\github\ai-project-01\team_handover_2026-04-29\REAL_SHARD_WORKER\scripts\update_manifest_from_npz.py --manifest C:\github\ai-project-01\team_handover_outputs\VALIDATION_MVP_FULL\shard_manifest_VALIDATION_MVP_FULL.csv --npz C:\github\ai-project-01\team_handover_outputs\VALIDATION_MVP_FULL\mediapipe_npz_VALIDATION_MVP_FULL.npz --failed C:\github\ai-project-01\team_handover_outputs\VALIDATION_MVP_FULL\failed_samples_VALIDATION_MVP_FULL.csv
