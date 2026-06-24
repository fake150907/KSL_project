#!/bin/bash
# ──────────────────────────────────────────────────────────────
# KSL 백엔드 자동배포 (pull 방식)
# VM이 origin/main을 감시하다가 새 커밋이 있으면 ff-merge 후
# ksl-backend(systemd) 재시작. 1분마다 cron(root)으로 실행.
#
# - 비밀키/외부 인증 없음 (공개 레포 fetch)
# - DB(SQLite)는 건드리지 않음. 재시작 시 init_db가 마이그레이션 자동 수행.
# - ff-only 라 충돌 시 배포를 건너뛰고 기존 코드 유지(안전).
# 설치 위치: /usr/local/bin/ksl-autodeploy.sh  (root 소유)
# ──────────────────────────────────────────────────────────────
set -u
REPO=/home/82102_asozp43/KSL_project
OWNER=82102_asozp43
BRANCH=main
SERVICE=ksl-backend
LOG=/var/log/ksl-autodeploy.log

ts() { date '+%F %T'; }

cd "$REPO" 2>/dev/null || { echo "$(ts) ERROR: repo 없음 $REPO" >>"$LOG"; exit 0; }

# 원격 최신 상태 확인
sudo -u "$OWNER" git fetch origin "$BRANCH" -q 2>>"$LOG" || { echo "$(ts) fetch 실패" >>"$LOG"; exit 0; }
LOCAL=$(sudo -u "$OWNER" git rev-parse HEAD 2>/dev/null)
REMOTE=$(sudo -u "$OWNER" git rev-parse "origin/$BRANCH" 2>/dev/null)

[ "$LOCAL" = "$REMOTE" ] && exit 0   # 변경 없음 → 조용히 종료

echo "$(ts) deploy ${LOCAL:0:7} -> ${REMOTE:0:7}" >>"$LOG"
if sudo -u "$OWNER" git merge --ff-only "origin/$BRANCH" >>"$LOG" 2>&1; then
  systemctl restart "$SERVICE" >>"$LOG" 2>&1
  echo "$(ts) restarted $SERVICE (now $(sudo -u "$OWNER" git rev-parse --short HEAD))" >>"$LOG"
else
  echo "$(ts) ff-merge 불가 — 배포 건너뜀(기존 코드 유지)" >>"$LOG"
fi
