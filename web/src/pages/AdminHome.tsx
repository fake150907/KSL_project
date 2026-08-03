import { type ReactNode, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { BarChart3, ChevronDown, LayoutDashboard, PlayCircle, Power, Save, ScrollText, Settings, Trash2, Wrench } from 'lucide-react'
import { useLogStream, type LogEntry, type LogLevel } from '../hooks/useLogStream'

interface AdminHomeProps {
  onLogout: () => void
  onSessionReset: () => void
}

type AdminView = 'dashboard' | 'system' | 'statistics' | 'logs' | 'settings'

type AdminSettings = {
  branchName: string
  installLocation: string
  adminEmail: string
  timezone: string
}

const ADMIN_SETTINGS_STORAGE_KEY = 'ksl_admin_settings'

const defaultAdminSettings: AdminSettings = {
  branchName: '서초동 이스트소프트',
  installLocation: '1층 민원센터',
  adminEmail: 'admin@example.com',
  timezone: 'Asia/Seoul',
}

const navItems = [
  { id: 'dashboard', label: '대시보드', description: '운영 현황', Icon: LayoutDashboard },
  { id: 'system', label: '시스템 시작', description: '민원인/상담원 실행', Icon: PlayCircle },
  { id: 'statistics', label: '통계 보기', description: '사용 지표', Icon: BarChart3 },
  { id: 'logs', label: '로그 확인', description: '시스템 로그', Icon: ScrollText },
  { id: 'settings', label: '설정', description: '관리자 설정', Icon: Settings },
] as const


const crownPixels = [
  '....................',
  '..BB.....BB.....BB..',
  '.BYYB...BYYB...BYYB.',
  '.BYYB...BYYB...BYYB.',
  '.BYYYB.BYYYYB.BYYYB.',
  '.BYYYYBYYYYYYBYYYYB.',
  '.BHYYYYYYYYYYYYYYYB.',
  '.BHHYYYYYYYYYYYYYYB.',
  '.BHYYYYYYYYYYYYYYYB.',
  '.BBBBBBBBBBBBBBBBBB.',
  '.BHHYYYYYYYYYYYYYYB.',
  '. BBBBBBBBBBBBBBBB .',
  '....................',
] as const

const crownPalette: Record<string, string> = {
  B: '#3b1605',
  Y: '#facc15',
  H: '#fde047',
  V: '#a855f7',
}

function loadAdminSettings(): AdminSettings {
  if (typeof window === 'undefined') {
    return defaultAdminSettings
  }

  try {
    const stored = window.localStorage.getItem(ADMIN_SETTINGS_STORAGE_KEY)
    if (!stored) {
      return defaultAdminSettings
    }

    return { ...defaultAdminSettings, ...JSON.parse(stored) }
  } catch {
    return defaultAdminSettings
  }
}

function PixelCrownIcon() {
  return (
    <div
      className="grid h-8 w-10"
      style={{ gridTemplateColumns: 'repeat(20, minmax(0, 1fr))', gridTemplateRows: 'repeat(13, minmax(0, 1fr))' }}
      aria-hidden="true"
    >
      {crownPixels.join('').split('').map((cell, index) => (
        <span
          key={index}
          style={{ backgroundColor: crownPalette[cell] ?? 'transparent' }}
        />
      ))}
    </div>
  )
}

export default function AdminHome({ onLogout, onSessionReset }: AdminHomeProps) {
  const navigate = useNavigate()
  const [view, setView] = useState<AdminView>('dashboard')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [adminSettings, setAdminSettings] = useState<AdminSettings>(() => loadAdminSettings())
  const [settingsSaved, setSettingsSaved] = useState(false)

  const selectView = (next: AdminView) => {
    setView(next)
    setSidebarOpen(false)
  }

  const startCitizen = () => {
    onSessionReset()
    navigate('/citizen')
  }

  const startAgent = () => {
    navigate('/agent/launch')
  }

  const updateAdminSetting = (key: keyof AdminSettings, value: string) => {
    setAdminSettings((current) => ({ ...current, [key]: value }))
    setSettingsSaved(false)
  }

  const saveAdminSettings = () => {
    window.localStorage.setItem(ADMIN_SETTINGS_STORAGE_KEY, JSON.stringify(adminSettings))
    setSettingsSaved(true)
  }

  return (
    <main className="h-screen overflow-y-auto bg-[#111827] text-[#e5e7eb]">
      {sidebarOpen && <button className="fixed inset-0 z-40 bg-black/50 lg:hidden" onClick={() => setSidebarOpen(false)} aria-label="메뉴 닫기" />}

      <aside className={`fixed left-0 top-0 z-50 h-screen w-64 transform border-r border-[#374151] bg-[#1f2937] transition-transform duration-300 lg:translate-x-0 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="flex h-full flex-col">
          <div className="flex h-16 items-center justify-between border-b border-[#374151] px-4">
            <div className="flex items-center gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#2563eb] shadow-sm shadow-[#2563eb]/30">
                <Wrench size={18} strokeWidth={2.4} className="text-white" />
              </div>
              <span className="font-bold text-white">관리자 시스템</span>
            </div>
            <button onClick={() => setSidebarOpen(false)} className="rounded-lg px-2 py-1 text-[#9ca3af] hover:text-white lg:hidden">X</button>
          </div>

          <nav className="flex-1 space-y-1 overflow-y-auto px-3 py-4">
            {navItems.map((item) => {
              const isActive = view === item.id
              const Icon = item.Icon
              return (
                <button
                  key={item.id}
                  onClick={() => selectView(item.id)}
                  className={`flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left transition ${isActive ? 'bg-[#2563eb] text-white' : 'text-[#e5e7eb] hover:bg-[#374151]'}`}
                >
                  <span className={`flex h-8 w-8 items-center justify-center rounded-md ${isActive ? 'bg-white/15' : 'bg-[#374151]'}`}>
                    <Icon size={18} strokeWidth={2.2} />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-sm font-medium">{item.label}</span>
                    <span className={`block text-xs ${isActive ? 'text-blue-100' : 'text-[#9ca3af]'}`}>{item.description}</span>
                  </span>
                </button>
              )
            })}
          </nav>

          <div className="border-t border-[#374151] p-3">
            <button onClick={onLogout} className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 font-medium text-[#ef4444] transition hover:bg-[#ef4444]/10">
              <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-[#dc2626] shadow-sm shadow-[#dc2626]/30">
                <Power size={21} strokeWidth={2.8} className="text-white" />
              </span>
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
                <p className="text-xs text-[#9ca3af]">{adminSettings.adminEmail}</p>
              </div>
              <div className="flex h-12 w-12 items-center justify-center rounded-full bg-[#2563eb] shadow-sm shadow-[#2563eb]/30">
                <PixelCrownIcon />
              </div>
            </div>
          </div>
        </header>

        <section className="p-4 pb-10 lg:p-6">
          {view === 'dashboard' && <DashboardView />}
          {view === 'system' && <SystemView startCitizen={startCitizen} startAgent={startAgent} />}
          {view === 'statistics' && <StatisticsView />}
          {view === 'logs' && <LogsView />}
          {view === 'settings' && (
            <SettingsView
              settings={adminSettings}
              saved={settingsSaved}
              onChange={updateAdminSetting}
              onSave={saveAdminSettings}
            />
          )}
        </section>
      </div>
    </main>
  )
}

function DashboardView() {
  return (
    <div className="space-y-6">
      <Header title="대시보드" description="민원 키오스크 운영 상태와 최근 활동을 확인합니다." />
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {[
          ['오늘 상담', '24건', '+12.5%'],
          ['연결 대기', '1건', '-'],
          ['평균 응답', '18초', '-4.1%'],
          ['처리 완료', '91%', '+6.2%'],
        ].map(([title, value, change]) => (
          <Card key={title}>
            <p className="text-sm text-[#9ca3af]">{title}</p>
            <p className="mt-4 text-2xl font-bold text-white">{value}</p>
            <p className="mt-2 text-sm font-medium text-[#10b981]">{change}</p>
          </Card>
        ))}
      </div>
      <Panel title="최근 활동">
        {[
          ['복지카드 재발급 상담 시작', '방금 전', 'success'],
          ['민원인 키오스크 접수', '15분 전', 'info'],
          ['상담 기록 저장 완료', '1시간 전', 'success'],
          ['수어 모델 연결 지연 감지', '5시간 전', 'warning'],
        ].map(([text, time, type]) => (
          <div key={text} className="flex items-center gap-3 rounded-lg bg-[#374151]/50 px-4 py-3">
            <span className={`h-2 w-2 rounded-full ${type === 'warning' ? 'bg-[#f59e0b]' : type === 'success' ? 'bg-[#10b981]' : 'bg-[#2563eb]'}`} />
            <span className="flex-1 text-sm text-[#e5e7eb]">{text}</span>
            <span className="text-xs text-[#9ca3af]">{time}</span>
          </div>
        ))}
      </Panel>
    </div>
  )
}

function SystemView({ startCitizen, startAgent }: { startCitizen: () => void; startAgent: () => void }) {
  return (
    <div className="space-y-6">
      <Header title="시스템 시작" description="민원인 전용 화면과 상담원 전용 대시보드를 분리 실행합니다." />
      <div className="grid gap-4 lg:grid-cols-2">
        <ActionCard title="민원인 전용 시스템 시작" description="키오스크 화면을 열고 카메라 기반 수어 인식 세션을 시작합니다." tone="green" onClick={startCitizen} />
        <ActionCard title="상담원 전용 시스템 시작" description="상담원 대시보드에서 대화, 민원 메모, 상담 기록을 관리합니다." tone="primary" onClick={startAgent} />
      </div>
      <Panel title="시스템 모듈 상태">
        {[
          ['메인 서버', '키오스크 애플리케이션 서버', '실행 중', '15시간'],
          ['수어 인식 엔진', '민원 수어 인식 및 문장 변환 모듈', '실행 중', '15시간'],
          ['상담 요약 API', '상담 내용 기록 및 요약 연결', '대기 중', '-'],
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
  return (
    <div className="space-y-6">
      <Header title="통계 보기" description="민원 상담 처리량과 시스템 사용률을 확인합니다." />
      <div className="grid gap-4 md:grid-cols-3">
        {[
          ['총 상담 건수', '1,284', '+15.3%'],
          ['평균 상담 시간', '4분 32초', '-2.1%'],
          ['처리 완료율', '91.4%', '+9.6%'],
        ].map(([title, value, change]) => (
          <Card key={title}>
            <p className="text-sm text-[#9ca3af]">{title}</p>
            <p className="mt-4 text-2xl font-bold text-white">{value}</p>
            <p className="mt-2 text-sm font-medium text-[#10b981]">{change}</p>
          </Card>
        ))}
      </div>
    </div>
  )
}

// ─── 로그 레벨별 스타일 ───────────────────────────────────────────────────────
const LEVEL_STYLE: Record<LogLevel, { badge: string; row: string }> = {
  error:   { badge: 'bg-red-500/20 text-red-400',    row: 'bg-red-500/5' },
  warning: { badge: 'bg-yellow-500/20 text-yellow-400', row: 'bg-yellow-500/5' },
  info:    { badge: 'bg-blue-500/20 text-blue-400',   row: '' },
  success: { badge: 'bg-green-500/20 text-green-400', row: '' },
}

const LEVEL_LABEL: Record<LogLevel, string> = {
  error: 'ERROR', warning: 'WARN', info: 'INFO', success: 'OK',
}

type LogFilter = 'all' | 'backend' | 'frontend'

function isFrontendSource(source: string) {
  return source.startsWith('Frontend')
}

function LogsView() {
  const { logs, connected, clearDisplay } = useLogStream()
  const [filter, setFilter] = useState<LogFilter>('all')
  const [autoScroll, setAutoScroll] = useState(true)
  const scrollRef = useRef<HTMLDivElement>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  // 새 로그 유입 시 자동 스크롤
  useEffect(() => {
    if (autoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  // logs.length 변화 때만 실행
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [logs.length, autoScroll])

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const el = e.currentTarget
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80
    setAutoScroll(atBottom)
  }

  const filtered: LogEntry[] = logs.filter((l) => {
    if (filter === 'backend')  return !isFrontendSource(l.source)
    if (filter === 'frontend') return isFrontendSource(l.source)
    return true
  })

  const counts = {
    all: logs.length,
    backend:  logs.filter((l) => !isFrontendSource(l.source)).length,
    frontend: logs.filter((l) => isFrontendSource(l.source)).length,
  }

  return (
    <div className="flex flex-col gap-4">
      {/* 헤더 */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-2xl font-bold text-white">로그 확인</h2>
          <p className="mt-1 text-sm text-[#9ca3af]">
            백엔드·프론트엔드 서버의 오류 로그를 실시간으로 수집합니다. (2xx 제외)
          </p>
        </div>
      </div>

      {/* 필터 탭 + 도구 */}
      <div className="flex flex-wrap items-center gap-2">
        {(['all', 'backend', 'frontend'] as LogFilter[]).map((f) => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition ${
              filter === f
                ? 'bg-[#2563eb] text-white'
                : 'bg-[#374151] text-[#9ca3af] hover:text-white'
            }`}
          >
            {f === 'all' ? '전체' : f === 'backend' ? '백엔드' : '프론트엔드'}
            <span className="ml-1.5 rounded-full bg-white/10 px-1.5 py-0.5 text-xs">
              {counts[f]}
            </span>
          </button>
        ))}

        <div className="ml-auto flex items-center gap-2">
          {/* 자동 스크롤 재활성화 */}
          {!autoScroll && (
            <button
              onClick={() => {
                setAutoScroll(true)
                bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
              }}
              className="inline-flex items-center gap-1 rounded-lg bg-[#2563eb]/20 px-3 py-1.5 text-xs font-medium text-[#93c5fd] hover:bg-[#2563eb]/40"
            >
              <ChevronDown size={13} /> 맨 아래로
            </button>
          )}
          {/* 화면 초기화 (서버 저장은 유지) */}
          <button
            onClick={clearDisplay}
            title="화면 로그 초기화 (서버 누적 데이터는 유지)"
            className="inline-flex items-center gap-1 rounded-lg bg-[#374151] px-3 py-1.5 text-xs font-medium text-[#9ca3af] hover:text-white"
          >
            <Trash2 size={13} /> 화면 초기화
          </button>
        </div>
      </div>

      {/* 로그 테이블 */}
      <div
        ref={scrollRef}
        onScroll={handleScroll}
        className="h-[calc(100vh-20rem)] min-h-[320px] overflow-y-auto rounded-lg border border-[#374151] bg-[#0d1117]"
      >
        {filtered.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-[#4b5563]">
            <ScrollText size={32} strokeWidth={1.2} />
            <p className="text-sm">
              {connected ? '아직 오류 로그가 없습니다.' : '서버에 연결하는 중…'}
            </p>
          </div>
        ) : (
          <table className="w-full min-w-[680px] font-mono text-xs">
            <thead className="sticky top-0 z-10 bg-[#161b22] text-[#6e7681]">
              <tr>
                <th className="px-4 py-2.5 text-left font-normal">시간</th>
                <th className="px-3 py-2.5 text-left font-normal w-16">레벨</th>
                <th className="px-3 py-2.5 text-left font-normal w-32">소스</th>
                <th className="px-4 py-2.5 text-left font-normal">메시지</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((log) => {
                const level = (log.level in LEVEL_STYLE ? log.level : 'info') as LogLevel
                const style = LEVEL_STYLE[level]
                return (
                  <tr
                    key={log.id}
                    className={`border-t border-[#21262d] transition-colors hover:bg-[#161b22] ${style.row}`}
                  >
                    <td className="px-4 py-2 text-[#6e7681] whitespace-nowrap">{log.timestamp}</td>
                    <td className="px-3 py-2">
                      <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold tracking-wide ${style.badge}`}>
                        {LEVEL_LABEL[level]}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-[#8b949e] whitespace-nowrap">{log.source}</td>
                    <td className="px-4 py-2 text-[#c9d1d9] break-all">{log.message}</td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
        {/* 자동 스크롤 앵커 */}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}

function SettingsView({
  settings,
  saved,
  onChange,
  onSave,
}: {
  settings: AdminSettings
  saved: boolean
  onChange: (key: keyof AdminSettings, value: string) => void
  onSave: () => void
}) {
  return (
    <div className="space-y-6">
      <Header title="설정" description="민원 어플리케이션과 관리자 설정을 관리합니다." />
      <Panel title="일반 설정">
        <div className="grid gap-4 md:grid-cols-2">
          <TextSetting label="지점명" value={settings.branchName} onChange={(value) => onChange('branchName', value)} />
          <TextSetting label="설치 위치" value={settings.installLocation} onChange={(value) => onChange('installLocation', value)} />
          <TextSetting label="관리자 이메일" value={settings.adminEmail} onChange={(value) => onChange('adminEmail', value)} />
          <TextSetting label="시간대" value={settings.timezone} onChange={(value) => onChange('timezone', value)} />
        </div>
        <div className="flex flex-col gap-3 pt-1 sm:flex-row sm:items-center sm:justify-end">
          {saved && <span className="text-sm font-medium text-[#6ee7b7]">저장되었습니다.</span>}
          <button
            type="button"
            onClick={onSave}
            className="inline-flex h-10 items-center justify-center gap-2 rounded-lg bg-[#2563eb] px-4 text-sm font-semibold text-white transition hover:bg-[#1d4ed8]"
          >
            <Save size={16} strokeWidth={2.4} />
            저장
          </button>
        </div>
      </Panel>
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
  return <section className="rounded-lg border border-[#374151] bg-[#1f2937] p-5 shadow-sm transition-colors hover:border-[#2563eb]/50">{children}</section>
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

function ActionCard({ title, description, tone, onClick }: { title: string; description: string; tone: 'green' | 'primary'; onClick: () => void }) {
  const classes = tone === 'green'
    ? 'border-[#10b981]/30 bg-[#10b981]/10 text-[#d1fae5] hover:border-[#10b981]/60'
    : 'border-[#2563eb]/30 bg-[#2563eb]/10 text-[#dbeafe] hover:border-[#2563eb]/60'
  return (
    <button onClick={onClick} className={`rounded-lg border p-6 text-left transition ${classes}`}>
      <h3 className="text-lg font-bold">{title}</h3>
      <p className="mt-2 text-sm leading-6 text-[#9ca3af]">{description}</p>
    </button>
  )
}

function TextSetting({ label, value, onChange }: { label: string; value: string; onChange: (value: string) => void }) {
  return (
    <label className="block">
      <span className="text-sm font-medium text-[#e5e7eb]">{label}</span>
      <input
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="mt-2 h-11 w-full rounded-lg border border-[#374151] bg-[#374151] px-3 text-sm outline-none transition focus:border-[#2563eb] focus:ring-2 focus:ring-[#2563eb]/30"
      />
    </label>
  )
}
