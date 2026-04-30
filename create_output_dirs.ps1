$dirs = @(
  "team_handover_outputs/REAL01_REAL03",
  "team_handover_outputs/REAL04_REAL06",
  "team_handover_outputs/REAL07_REAL09",
  "team_handover_outputs/REAL10_REAL12",
  "team_handover_outputs/REAL13_REAL16",
  "team_handover_outputs/INTEGRATION_MASTER",
  "team_handover_outputs/TEAM_LEAD_REACT",
  "team_handover_outputs/VALIDATION_HOLDOUT"
)

foreach ($dir in $dirs) {
  New-Item -ItemType Directory -Force -Path $dir | Out-Null
  Write-Host "[OK] $dir"
}


