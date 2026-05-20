import { type ReactNode, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { BarChart3, LayoutDashboard, PlayCircle, ScrollText, Settings } from 'lucide-react'

interface AdminHomeProps {
  onLogout: () => void
  onSessionReset: () => void
}

type AdminView = 'dashboard' | 'system' | 'statistics' | 'logs' | 'settings'

const navItems = [
  { id: 'dashboard', label: '대시보드', description: '운영 현황', Icon: LayoutDashboard },
  { id: 'system', label: '시스템 시작', description: '민원인/상담원 실행', Icon: PlayCircle },
  { id: 'statistics', label: '통계 보기', description: '사용 지표', Icon: BarChart3 },
  { id: 'logs', label: '로그 확인', description: '시스템 로그', Icon: ScrollText },
  { id: 'settings', label: '설정', description: '관리자 설정', Icon: Settings },
] as const

const logs = [
  { id: 1, timestamp: '2026-05-10 09:12:15', level: 'info', source: 'API Server', message: '관리자 로그인 요청 처리 완료' },
  { id: 2, timestamp: '2026-05-10 09:10:42', level: 'success', source: 'Frontend', message: '민원인 전용 화면 빌드 검증 완료' },
  { id: 3, timestamp: '2026-05-10 09:08:18', level: 'warning', source: 'Vision', message: '수어 인식 모델 연결 대기 중' },
  { id: 4, timestamp: '2026-05-10 09:05:55', level: 'success', source: 'Summary', message: '상담 요약 API 환경 확인 완료' },
] as const

export default function AdminHome({ onLogout, onSessionReset }: AdminHomeProps) {
  const navigate = useNavigate()
  const [view, setView] = useState<AdminView>('dashboard')
  const [sidebarOpen, setSidebarOpen] = useState(false)

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

  return (
    <main className="h-screen overflow-y-auto bg-[#111827] text-[#e5e7eb]">
      {sidebarOpen && <button className="fixed inset-0 z-40 bg-black/50 lg:hidden" onClick={() => setSidebarOpen(false)} aria-label="메뉴 닫기" />}

      <aside className={`fixed left-0 top-0 z-50 h-screen w-64 transform border-r border-[#374151] bg-[#1f2937] transition-transform duration-300 lg:translate-x-0 ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
        <div className="flex h-full flex-col">
          <div className="flex h-16 items-center justify-between border-b border-[#374151] px-4">
            <div className="flex items-center gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#2563eb] text-sm font-black text-white">민</div>
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
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#2563eb] text-sm font-medium text-white">A</div>
            </div>
          </div>
        </header>

        <section className="p-4 pb-10 lg:p-6">
          {view === 'dashboard' && <DashboardView />}
          {view === 'system' && <SystemView startCitizen={startCitizen} startAgent={startAgent} />}
          {view === 'statistics' && <StatisticsView />}
          {view === 'logs' && <LogsView />}
          {view === 'settings' && <SettingsView />}
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

function LogsView() {
  return (
    <div className="space-y-6">
      <Header title="로그 확인" description="시스템 로그를 확인합니다." />
      <div className="overflow-hidden rounded-lg border border-[#374151] bg-[#1f2937]">
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
                <td className="px-4 py-3"><span className="rounded-full bg-[#2563eb]/10 px-2.5 py-1 text-xs font-medium text-[#93c5fd]">{log.level}</span></td>
                <td className="px-4 py-3 text-sm font-medium text-[#e5e7eb]">{log.source}</td>
                <td className="px-4 py-3 text-sm text-[#e5e7eb]">{log.message}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function SettingsView() {
  return (
    <div className="space-y-6">
      <Header title="설정" description="민원 키오스크와 관리자 설정을 관리합니다." />
      <Panel title="일반 설정">
        <div className="grid gap-4 md:grid-cols-2">
          <TextSetting label="키오스크 이름" value="수어 민원 상담 키오스크" />
          <TextSetting label="설치 위치" value="1층 민원센터" />
          <TextSetting label="관리자 이메일" value="admin@example.com" />
          <TextSetting label="시간대" value="Asia/Seoul" />
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

function TextSetting({ label, value }: { label: string; value: string }) {
  return (
    <label className="block">
      <span className="text-sm font-medium text-[#e5e7eb]">{label}</span>
      <input value={value} readOnly className="mt-2 h-11 w-full rounded-lg border border-[#374151] bg-[#374151] px-3 text-sm outline-none" />
    </label>
  )
}
