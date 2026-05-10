import { type ReactNode, useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'

interface AdminHomeProps {
  onLogout: () => void
  onSessionReset: () => void
}

type AdminView = 'dashboard' | 'system' | 'statistics' | 'logs' | 'settings'
type LogLevel = 'all' | 'info' | 'success' | 'warning' | 'error'

const SETTINGS_KEY = 'ksl-admin-settings'

const navItems: Array<{ id: AdminView; label: string; description: string; icon: string }> = [
  { id: 'dashboard', label: '대시보드', description: '운영 현황', icon: 'D' },
  { id: 'system', label: '시스템 시작', description: '환자/의사 실행', icon: 'P' },
  { id: 'statistics', label: '통계 보기', description: '사용 지표', icon: 'S' },
  { id: 'logs', label: '로그 확인', description: '시스템 로그', icon: 'L' },
  { id: 'settings', label: '설정', description: '관리자 설정', icon: 'G' },
]

const defaultSettings = {
  kioskName: '수어 진료 키오스크',
  kioskLocation: '1층 로비',
  adminEmail: 'admin@example.com',
  timezone: 'Asia/Seoul',
  darkMode: true,
  emailNotifications: true,
  pushNotifications: false,
  securityAlerts: true,
  sessionTimeout: '30',
  twoFactorAuth: false,
  ipWhitelist: '',
}

type Settings = typeof defaultSettings

const logs = [
  { id: 1, timestamp: '2026-05-10 09:12:15', level: 'info', source: 'API Server', message: '관리자 로그인 요청 처리 완료' },
  { id: 2, timestamp: '2026-05-10 09:10:42', level: 'success', source: 'Frontend', message: '환자 전용 화면 빌드 검증 완료' },
  { id: 3, timestamp: '2026-05-10 09:08:18', level: 'warning', source: 'Vision', message: '현재 Python 환경에 수어 인식 의존성 일부가 없습니다' },
  { id: 4, timestamp: '2026-05-10 09:05:55', level: 'success', source: 'Summary', message: 'Anthropic API 키 환경변수 확인 완료' },
  { id: 5, timestamp: '2026-05-10 09:01:33', level: 'info', source: 'System', message: '백엔드 health check 200 응답' },
] as const

type LogEntry = (typeof logs)[number]

const statCards = [
  ['활성 사용자', '1,234', '+12.5%'],
  ['현재 FPS', '58.7', '+3.1%'],
  ['요약 사용률', '72.4%', '+9.6%'],
  ['평균 지연', '12ms', '-2.1%'],
]

const weeklyData = [
  ['월', 2400, 218],
  ['화', 3100, 241],
  ['수', 2800, 226],
  ['목', 3500, 268],
  ['금', 4200, 286],
  ['토', 3800, 252],
  ['일', 2900, 233],
] as const

export default function AdminHome({ onLogout, onSessionReset }: AdminHomeProps) {
  const navigate = useNavigate()
  const [view, setView] = useState<AdminView>('dashboard')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [settings, setSettings] = useState<Settings>(defaultSettings)
  const [saveMessage, setSaveMessage] = useState('')
  const [logFilter, setLogFilter] = useState<LogLevel>('all')
  const [logSearch, setLogSearch] = useState('')

  useEffect(() => {
    const stored = localStorage.getItem(SETTINGS_KEY)
    if (!stored) return
    try {
      setSettings({ ...defaultSettings, ...JSON.parse(stored) })
    } catch {}
  }, [])

  const filteredLogs = useMemo(() => {
    return logs.filter((log) => {
      const matchesFilter = logFilter === 'all' || log.level === logFilter
      const query = logSearch.trim().toLowerCase()
      const matchesSearch = !query || log.message.toLowerCase().includes(query) || log.source.toLowerCase().includes(query)
      return matchesFilter && matchesSearch
    })
  }, [logFilter, logSearch])

  const selectView = (next: AdminView) => {
    setView(next)
    setSidebarOpen(false)
  }

  const startPatient = () => {
    onSessionReset()
    navigate('/patient')
  }

  const startDoctor = () => {
    navigate('/doctor/launch')
  }

  const updateSetting = (key: keyof Settings, value: string | boolean) => {
    setSettings((prev) => ({ ...prev, [key]: value }))
    setSaveMessage('')
  }

  const saveSettings = () => {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings))
    setSaveMessage('변경사항이 저장되었습니다.')
  }

  return (
    <main className="min-h-screen bg-[#111827] text-[#e5e7eb]">
      {sidebarOpen && <button className="fixed inset-0 z-40 bg-black/50 lg:hidden" onClick={() => setSidebarOpen(false)} aria-label="메뉴 닫기" />}

      <aside
        className={`fixed left-0 top-0 z-50 h-screen w-64 transform border-r border-[#374151] bg-[#1f2937] transition-transform duration-300 lg:translate-x-0 ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        <div className="flex h-full flex-col">
          <div className="flex h-16 items-center justify-between border-b border-[#374151] px-4">
            <div className="flex items-center gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#4f46e5] text-sm font-black text-white">K</div>
              <span className="font-bold text-white">관리자 시스템</span>
            </div>
            <button onClick={() => setSidebarOpen(false)} className="rounded-lg px-2 py-1 text-[#9ca3af] hover:text-white lg:hidden">X</button>
          </div>

          <nav className="flex-1 space-y-1 overflow-y-auto px-3 py-4">
            {navItems.map((item) => {
              const isActive = view === item.id
              return (
                <button
                  key={item.id}
                  onClick={() => selectView(item.id)}
                  className={`flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left transition ${
                    isActive ? 'bg-[#4f46e5] text-white' : 'text-[#e5e7eb] hover:bg-[#374151]'
                  }`}
                >
                  <span className={`flex h-8 w-8 items-center justify-center rounded-md text-xs font-black ${isActive ? 'bg-white/15' : 'bg-[#374151]'}`}>{item.icon}</span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-sm font-medium">{item.label}</span>
                    <span className={`block text-xs ${isActive ? 'text-indigo-100' : 'text-[#9ca3af]'}`}>{item.description}</span>
                  </span>
                  {isActive && <span className="text-sm">›</span>}
                </button>
              )
            })}
          </nav>

          <div className="border-t border-[#374151] p-3">
            <button onClick={onLogout} className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 font-medium text-[#ef4444] transition hover:bg-[#ef4444]/10">
              <span className="flex h-8 w-8 items-center justify-center rounded-md bg-[#ef4444]/10">O</span>
              로그아웃
            </button>
          </div>
        </div>
      </aside>

      <div className="lg:pl-64">
        <header className="sticky top-0 z-30 h-16 border-b border-[#374151] bg-[#1f2937]/80 backdrop-blur-sm">
          <div className="flex h-full items-center justify-between px-4">
            <button onClick={() => setSidebarOpen(true)} className="rounded-lg p-2 text-[#e5e7eb] hover:bg-[#374151] lg:hidden" aria-label="메뉴 열기">
              <span className="block h-0.5 w-5 bg-current" />
              <span className="mt-1 block h-0.5 w-5 bg-current" />
              <span className="mt-1 block h-0.5 w-5 bg-current" />
            </button>
            <div className="ml-auto flex items-center gap-3">
              <div className="hidden text-right sm:block">
                <p className="text-sm font-medium text-white">관리자</p>
                <p className="text-xs text-[#9ca3af]">admin@system.com</p>
              </div>
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#4f46e5] text-sm font-medium text-white">A</div>
            </div>
          </div>
        </header>

        <section className="p-4 lg:p-6">
          {view === 'dashboard' && <DashboardView />}
          {view === 'system' && <SystemView startPatient={startPatient} startDoctor={startDoctor} />}
          {view === 'statistics' && <StatisticsView />}
          {view === 'logs' && (
            <LogsView logs={filteredLogs} logFilter={logFilter} setLogFilter={setLogFilter} logSearch={logSearch} setLogSearch={setLogSearch} />
          )}
          {view === 'settings' && (
            <SettingsView settings={settings} updateSetting={updateSetting} saveSettings={saveSettings} saveMessage={saveMessage} />
          )}
        </section>
      </div>
    </main>
  )
}

function DashboardView() {
  return (
    <div className="space-y-6">
      <Header title="대시보드" description="키오스크 운영 상태와 최근 활동을 확인합니다." />
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {statCards.map(([title, value, change]) => (
          <Card key={title}>
            <div className="flex items-start justify-between">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-[#4f46e5]/10 text-[#818cf8]">•</div>
              <p className={`text-sm font-medium ${change.startsWith('+') ? 'text-[#10b981]' : 'text-[#ef4444]'}`}>{change}</p>
            </div>
            <p className="mt-4 text-2xl font-bold text-white">{value}</p>
            <p className="mt-1 text-sm text-[#9ca3af]">{title}</p>
          </Card>
        ))}
      </div>
      <div className="grid gap-6 xl:grid-cols-[1.5fr_1fr]">
        <Panel title="최근 활동">
          {[
            ['시스템 업데이트 완료', '2분 전', 'success'],
            ['새 환자 세션 시작', '15분 전', 'info'],
            ['진단내용 요약 생성', '1시간 전', 'success'],
            ['응답 지연 감지', '5시간 전', 'warning'],
          ].map(([text, time, type]) => (
            <div key={text} className="flex items-center gap-3 rounded-lg bg-[#374151]/50 px-4 py-3">
              <span className={`h-2 w-2 rounded-full ${type === 'warning' ? 'bg-[#f59e0b]' : type === 'success' ? 'bg-[#10b981]' : 'bg-[#4f46e5]'}`} />
              <span className="flex-1 text-sm text-[#e5e7eb]">{text}</span>
              <span className="text-xs text-[#9ca3af]">{time}</span>
            </div>
          ))}
        </Panel>
        <Panel title="실시간 성능">
          <Metric label="현재 FPS" value="58.7" width="98%" color="bg-[#4f46e5]" />
          <Metric label="렌더링 지연" value="12ms" width="24%" color="bg-[#10b981]" />
          <Metric label="프레임 드롭" value="1.8%" width="18%" color="bg-[#f59e0b]" />
        </Panel>
      </div>
    </div>
  )
}

function SystemView({ startPatient, startDoctor }: { startPatient: () => void; startDoctor: () => void }) {
  return (
    <div className="space-y-6">
      <Header title="시스템 시작" description="환자 전용 화면과 의사 전용 대시보드를 분리 실행합니다." />
      <div className="grid gap-4 lg:grid-cols-2">
        <ActionCard title="환자 전용 시스템 시작" description="키오스크 화면을 열고 카메라 기반 수어 인식 세션을 시작합니다." tone="green" onClick={startPatient} />
        <ActionCard title="의사 전용 시스템 시작" description="의사 대시보드에서 대화, 진료 메모, 요약 전송을 관리합니다." tone="primary" onClick={startDoctor} />
      </div>
      <Panel title="시스템 모듈 상태">
        {[
          ['메인 서버', '키오스크 애플리케이션 서버', '실행 중', '15일 4시간'],
          ['진단 엔진', '수어 진단 및 요약 처리 모듈', '실행 중', '15일 4시간'],
          ['Claude 요약 API', '진단내용 요약 연결', '대기 중', '-'],
        ].map(([name, description, status, uptime]) => (
          <div key={name} className="flex flex-col gap-3 rounded-lg border border-[#374151] bg-[#374151]/35 p-4 sm:flex-row sm:items-center">
            <div className="flex-1">
              <p className="font-semibold text-white">{name}</p>
              <p className="mt-1 text-sm text-[#9ca3af]">{description}</p>
            </div>
            <span className="rounded-full bg-[#10b981]/10 px-3 py-1 text-xs font-medium text-[#6ee7b7]">{status}</span>
            <span className="text-xs text-[#9ca3af]">가동 시간: {uptime}</span>
          </div>
        ))}
      </Panel>
    </div>
  )
}

function StatisticsView() {
  const maxVisitors = Math.max(...weeklyData.map((item) => item[1]))
  const maxDuration = Math.max(...weeklyData.map((item) => item[2]))

  return (
    <div className="space-y-6">
      <Header title="통계 보기" description="키오스크 사용 통계와 요약 사용률을 확인합니다." />
      <div className="grid gap-4 md:grid-cols-3">
        {[
          ['총 방문자', '124,892', '+15.3%'],
          ['평균 체류시간', '4분 32초', '-2.1%'],
          ['요약 사용률', '72.4%', '+9.6%'],
        ].map(([title, value, change]) => (
          <Card key={title}>
            <p className="text-sm text-[#9ca3af]">{title}</p>
            <p className="mt-4 text-2xl font-bold text-white">{value}</p>
            <p className={`mt-2 text-sm font-medium ${change.startsWith('+') ? 'text-[#10b981]' : 'text-[#ef4444]'}`}>{change}</p>
          </Card>
        ))}
      </div>
      <Panel title="주간 트래픽">
        {weeklyData.map(([day, visitors, duration]) => (
          <div key={day} className="grid grid-cols-[32px_1fr] items-center gap-4">
            <span className="text-sm font-medium text-[#e5e7eb]">{day}</span>
            <div className="space-y-2">
              <Bar width={`${(visitors / maxVisitors) * 100}%`} color="bg-[#4f46e5]" label={`${visitors.toLocaleString()}명`} />
              <Bar width={`${(duration / maxDuration) * 100}%`} color="bg-[#10b981]" label={`${Math.floor(duration / 60)}분 ${duration % 60}초`} />
            </div>
          </div>
        ))}
      </Panel>
      <Panel title="진단내용 요약 이용률">
        {[
          ['요약 사용 세션', '8,924', '72%'],
          ['요약 저장', '5,381', '43%'],
          ['요약 재확인', '3,106', '25%'],
          ['요약 공유', '1,284', '10%'],
        ].map(([label, value, width]) => (
          <Metric key={label} label={label} value={value} width={width} color="bg-[#4f46e5]" />
        ))}
      </Panel>
    </div>
  )
}

function LogsView({
  logs,
  logFilter,
  setLogFilter,
  logSearch,
  setLogSearch,
}: {
  logs: readonly LogEntry[]
  logFilter: LogLevel
  setLogFilter: (value: LogLevel) => void
  logSearch: string
  setLogSearch: (value: string) => void
}) {
  return (
    <div className="space-y-6">
      <Header title="로그 확인" description="시스템 로그를 검색하고 상태별로 필터링합니다." />
      <Card>
        <div className="grid gap-3 md:grid-cols-[1fr_180px]">
          <input value={logSearch} onChange={(event) => setLogSearch(event.target.value)} className="h-11 rounded-lg border border-[#374151] bg-[#374151] px-4 text-sm outline-none focus:ring-2 focus:ring-[#4f46e5]" placeholder="로그 검색" />
          <select value={logFilter} onChange={(event) => setLogFilter(event.target.value as LogLevel)} className="h-11 rounded-lg border border-[#374151] bg-[#374151] px-3 text-sm outline-none focus:ring-2 focus:ring-[#4f46e5]">
            <option value="all">전체 레벨</option>
            <option value="info">정보</option>
            <option value="success">성공</option>
            <option value="warning">경고</option>
            <option value="error">오류</option>
          </select>
        </div>
      </Card>
      <div className="overflow-hidden rounded-lg border border-[#374151] bg-[#1f2937]">
        <div className="overflow-x-auto">
          <table className="w-full min-w-[760px]">
            <thead className="bg-[#374151]/50 text-left text-xs uppercase text-[#9ca3af]">
              <tr>
                <th className="px-4 py-3">시간</th>
                <th className="px-4 py-3">레벨</th>
                <th className="px-4 py-3">소스</th>
                <th className="px-4 py-3">메시지</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[#374151]">
              {logs.map((log) => (
                <tr key={log.id} className="hover:bg-[#374151]/25">
                  <td className="px-4 py-3 font-mono text-xs text-[#9ca3af]">{log.timestamp}</td>
                  <td className="px-4 py-3"><LogBadge level={log.level} /></td>
                  <td className="px-4 py-3 text-sm font-medium text-[#e5e7eb]">{log.source}</td>
                  <td className="px-4 py-3 text-sm text-[#e5e7eb]">{log.message}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

function SettingsView({
  settings,
  updateSetting,
  saveSettings,
  saveMessage,
}: {
  settings: Settings
  updateSetting: (key: keyof Settings, value: string | boolean) => void
  saveSettings: () => void
  saveMessage: string
}) {
  return (
    <div className="space-y-6">
      <Header title="설정" description="키오스크, 알림, 보안 관련 관리자 설정을 관리합니다." />
      <Panel title="일반 설정">
        <div className="grid gap-4 md:grid-cols-2">
          <TextSetting label="키오스크 이름" value={settings.kioskName} onChange={(value) => updateSetting('kioskName', value)} />
          <TextSetting label="키오스크 위치" value={settings.kioskLocation} onChange={(value) => updateSetting('kioskLocation', value)} />
          <TextSetting label="관리자 이메일" value={settings.adminEmail} onChange={(value) => updateSetting('adminEmail', value)} />
          <TextSetting label="시간대" value={settings.timezone} onChange={(value) => updateSetting('timezone', value)} />
        </div>
      </Panel>
      <Panel title="알림 설정">
        <Toggle label="이메일 알림" description="중요 업데이트를 이메일로 받기" checked={settings.emailNotifications} onChange={(value) => updateSetting('emailNotifications', value)} />
        <Toggle label="푸시 알림" description="브라우저 푸시 알림 받기" checked={settings.pushNotifications} onChange={(value) => updateSetting('pushNotifications', value)} />
        <Toggle label="보안 알림" description="보안 관련 이벤트 알림 받기" checked={settings.securityAlerts} onChange={(value) => updateSetting('securityAlerts', value)} />
      </Panel>
      <Panel title="보안 설정">
        <TextSetting label="세션 타임아웃(분)" value={settings.sessionTimeout} onChange={(value) => updateSetting('sessionTimeout', value)} />
        <Toggle label="2단계 인증" description="추가 보안 레이어 활성화" checked={settings.twoFactorAuth} onChange={(value) => updateSetting('twoFactorAuth', value)} />
        <label className="block">
          <span className="text-sm font-medium text-[#e5e7eb]">IP 화이트리스트</span>
          <textarea value={settings.ipWhitelist} onChange={(event) => updateSetting('ipWhitelist', event.target.value)} className="mt-2 h-24 w-full resize-none rounded-lg border border-[#374151] bg-[#374151] p-3 text-sm outline-none focus:ring-2 focus:ring-[#4f46e5]" placeholder="허용할 IP 주소를 한 줄에 하나씩 입력" />
        </label>
      </Panel>
      <div className="flex items-center gap-4">
        <button onClick={saveSettings} className="rounded-lg bg-[#4f46e5] px-5 py-3 text-sm font-medium text-white transition hover:bg-[#4338ca]">변경사항 저장</button>
        {saveMessage && <p className="text-sm font-medium text-[#10b981]">{saveMessage}</p>}
      </div>
    </div>
  )
}

function Header({ title, description }: { title: string; description: string }) {
  return (
    <div>
      <h2 className="text-2xl font-bold text-white">{title}</h2>
      <p className="mt-1 text-sm text-[#9ca3af]">{description}</p>
    </div>
  )
}

function Card({ children }: { children: ReactNode }) {
  return <section className="rounded-lg border border-[#374151] bg-[#1f2937] p-5 shadow-sm transition-colors hover:border-[#4f46e5]/50">{children}</section>
}

function Panel({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="rounded-lg border border-[#374151] bg-[#1f2937]">
      <div className="border-b border-[#374151] px-5 py-4">
        <h3 className="font-semibold text-white">{title}</h3>
      </div>
      <div className="space-y-4 p-5">{children}</div>
    </section>
  )
}

function Metric({ label, value, width, color }: { label: string; value: string; width: string; color: string }) {
  return (
    <div>
      <div className="mb-2 flex items-center justify-between">
        <span className="text-sm text-[#9ca3af]">{label}</span>
        <span className="text-sm font-medium text-[#e5e7eb]">{value}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-[#374151]">
        <div className={`h-full rounded-full ${color}`} style={{ width }} />
      </div>
    </div>
  )
}

function Bar({ width, color, label }: { width: string; color: string; label: string }) {
  return (
    <div className="flex items-center gap-3">
      <div className="h-2 flex-1 overflow-hidden rounded-full bg-[#374151]">
        <div className={`h-full rounded-full ${color}`} style={{ width }} />
      </div>
      <span className="w-20 text-right text-xs text-[#9ca3af]">{label}</span>
    </div>
  )
}

function ActionCard({ title, description, tone, onClick }: { title: string; description: string; tone: 'green' | 'primary'; onClick: () => void }) {
  const classes = tone === 'green'
    ? 'border-[#10b981]/30 bg-[#10b981]/10 text-[#d1fae5] hover:border-[#10b981]/60'
    : 'border-[#4f46e5]/30 bg-[#4f46e5]/10 text-[#e0e7ff] hover:border-[#4f46e5]/60'
  return (
    <button onClick={onClick} className={`rounded-lg border p-6 text-left transition ${classes}`}>
      <h3 className="text-lg font-bold">{title}</h3>
      <p className="mt-2 text-sm leading-6 text-[#9ca3af]">{description}</p>
    </button>
  )
}

function LogBadge({ level }: { level: 'info' | 'success' | 'warning' | 'error' }) {
  const classes = {
    info: 'bg-[#4f46e5]/10 text-[#a5b4fc]',
    success: 'bg-[#10b981]/10 text-[#6ee7b7]',
    warning: 'bg-[#f59e0b]/10 text-[#fcd34d]',
    error: 'bg-[#ef4444]/10 text-[#fca5a5]',
  }
  const labels = { info: '정보', success: '성공', warning: '경고', error: '오류' }
  return <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${classes[level]}`}>{labels[level]}</span>
}

function TextSetting({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label className="block">
      <span className="text-sm font-medium text-[#e5e7eb]">{label}</span>
      <input value={value} onChange={(event) => onChange(event.target.value)} className="mt-2 h-11 w-full rounded-lg border border-[#374151] bg-[#374151] px-3 text-sm outline-none focus:ring-2 focus:ring-[#4f46e5]" />
    </label>
  )
}

function Toggle({ label, description, checked, onChange }: { label: string; description: string; checked: boolean; onChange: (value: boolean) => void }) {
  return (
    <div className="flex items-center justify-between gap-4 rounded-lg bg-[#374151]/50 p-3">
      <div>
        <p className="text-sm font-medium text-[#e5e7eb]">{label}</p>
        <p className="mt-1 text-xs text-[#9ca3af]">{description}</p>
      </div>
      <button onClick={() => onChange(!checked)} className={`relative h-6 w-12 rounded-full transition ${checked ? 'bg-[#4f46e5]' : 'bg-[#6b7280]'}`} aria-label={label}>
        <span className={`absolute top-1 h-4 w-4 rounded-full bg-white transition ${checked ? 'left-7' : 'left-1'}`} />
      </button>
    </div>
  )
}
