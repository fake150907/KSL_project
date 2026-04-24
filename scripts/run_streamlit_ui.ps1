$ErrorActionPreference = "Stop"

Set-Location "C:\github\ai-project-01"

& "C:\Users\Samsung\anaconda3\envs\gesture-test\python.exe" `
  -m streamlit run src/ui/app.py `
  --server.address=0.0.0.0 `
  --server.headless=true `
  --server.port=8501
