@echo off
REM 시그널링 서버(소켓)를 Cloud Run에 배포하는 스크립트.
REM 사용법: 이 파일을 더블클릭하거나 cmd에서 실행. (gcloud 로그인 필요)
cd /d "%~dp0"
echo ===== ksl-signaling Cloud Run 배포 시작 =====
gcloud run deploy ksl-signaling --source . --region asia-northeast3 --allow-unauthenticated --timeout 3600 --min-instances 1 --max-instances 1
echo.
echo ===== 완료. 위에 Service URL 나오면 성공 =====
pause
