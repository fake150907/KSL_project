# Team Handover Package 2026-04-28 v2

This is the handover package for role-based project work.

Note: the working folder is still named `team_handover_2026-04-28` because earlier history and documents already used that path. For team distribution, use:

```text
team_handover_2026-04-28_v2.zip
```

Included executable scripts:

```text
00_COMMON/check_common_environment.ps1
00_COMMON/04_핵심_기술기준_환경공지.md
create_output_dirs.ps1

REAL_SHARD_WORKER/run_check_existing_shard.ps1
REAL_SHARD_WORKER/run_extract_real_from_zip.ps1
REAL_SHARD_WORKER/run_build_shard_manifest_from_videos.ps1
REAL_SHARD_WORKER/run_preprocess_shard.ps1
REAL_SHARD_WORKER/run_inspect_shard_npz.ps1
REAL_SHARD_WORKER/run_package_shard_output.ps1
REAL_SHARD_WORKER/scripts/build_shard_manifest_from_videos.py
REAL_SHARD_WORKER/scripts/inspect_npz.py

INTEGRATION_MASTER/run_check_received_shards.ps1
INTEGRATION_MASTER/run_merge_shards.ps1
INTEGRATION_MASTER/run_inspect_master_npz.ps1
INTEGRATION_MASTER/run_train_master_smoke.ps1
INTEGRATION_MASTER/scripts/merge_shards.py
INTEGRATION_MASTER/scripts/inspect_npz.py

TEAM_LEAD_REACT/scripts/run_web_build_check.ps1
TEAM_LEAD_REACT/scripts/run_web_dev.ps1
TEAM_LEAD_REACT/scripts/run_backend_start.ps1
TEAM_LEAD_REACT/scripts/run_backend_health_check.ps1
```

The PowerShell files are convenience wrappers. Python helper scripts are placed inside each role folder so each teammate can focus on their own folder. This package does not include the full `web`, `backend`, `src`, `config`, original mp4, NPZ, or checkpoint files; those must exist in the actual project root.



