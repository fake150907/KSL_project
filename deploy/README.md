# 배포 (Deploy)

## 구성요소별 배포 방식
| 구성요소 | 위치 | 배포 |
|---|---|---|
| 프론트 | Vercel | main 머지 시 **자동** |
| 시그널링 서버 | Cloud Run | main 머지 시 GitHub Actions로 **자동** (`.github/workflows/deploy-signaling.yml`) |
| 백엔드 | GCP VM `ksl-backend` (asia-northeast3-a) | main 머지 시 **자동** (VM의 pull 방식 cron, 아래) |

## 백엔드 자동배포 (pull 방식)
VM이 1분마다 `origin/main`을 확인해, 새 커밋이 있으면 ff-merge 후 `ksl-backend` 서비스를 재시작합니다.

- 스크립트: [`vm-autodeploy.sh`](vm-autodeploy.sh) → VM의 `/usr/local/bin/ksl-autodeploy.sh`
- 스케줄: `/etc/cron.d/ksl-autodeploy` (매분, root)
- 로그: VM `/var/log/ksl-autodeploy.log`
- 특징: 비밀키 없음(공개 레포 fetch), DB(SQLite) 미변경, 충돌 시 배포 스킵(기존 코드 유지)

DB 마이그레이션은 서비스 재시작 시 `init_db`가 자동 수행하므로 별도 작업 불필요.

### 수동 배포가 필요할 때
```bash
sudo -u 82102_asozp43 git -C /home/82102_asozp43/KSL_project pull
sudo systemctl restart ksl-backend
```
