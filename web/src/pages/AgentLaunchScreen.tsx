import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { AgentNote } from '../types'
import { registerRole, socket } from '../socket'

interface ConsultationRecord {
  id: string
  date: string
  endDate?: string
  citizenName: string
  citizenDob: string
  citizenGender: string
  citizenPhone: string
  notes: AgentNote[]
  consultationSummary?: string
  isSent?: boolean
  deliveryStatus?: 'pending' | 'kakao_sent' | 'clipboard_copied' | 'failed'
  sentAt?: string
}

interface AgentLaunchScreenProps {
  onSessionReset?: () => void
}

interface SummaryPayload {
  citizenName?: string
  citizenPhone?: string
  consultationSummary: string
  isSent?: boolean
  deliveryStatus?: 'pending' | 'kakao_sent' | 'clipboard_copied' | 'failed'
  sentAt?: string
}

const TAG_STYLES: Record<AgentNote['tag'], string> = {
  문의: 'bg-blue-50 text-blue-700 border-blue-200',
  확인: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  처리: 'bg-violet-50 text-violet-700 border-violet-200',
}

function ServiceLogo({ className = '' }: { className?: string }) {
  return (
    <div className={`relative flex items-center justify-center rounded-[28%] bg-blue-600 shadow-xl shadow-blue-100 ${className}`}>
      <svg viewBox="0 0 120 120" className="h-[72%] w-[72%]" aria-hidden="true">
        <path d="M65 20c6 0 10 5 10 12v24h3V28c0-6 4-10 9-10s9 4 9 10v43c0 20-16 36-36 36H50c-18 0-32-14-32-32V45c0-6 4-10 9-10s9 4 9 10v16h3V31c0-6 4-10 9-10s9 4 9 10v28h3V32c0-7 4-12 10-12z" fill="#FDE2C8" />
        <path d="M36 62V45M51 59V31M63 58V32M77 56V28" stroke="#F4B58D" strokeWidth="5" strokeLinecap="round" />
        <path d="M44 82c10 11 31 12 42-2" stroke="#E9A27A" strokeWidth="6" strokeLinecap="round" fill="none" />
      </svg>
    </div>
  )
}

export default function AgentLaunchScreen({ onSessionReset }: AgentLaunchScreenProps) {
  const navigate = useNavigate()
  const [notified, setNotified] = useState(false)
  const [currentTime, setCurrentTime] = useState(() => new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' }))
  const [showRecords, setShowRecords] = useState(false)
  const [records, setRecords] = useState<ConsultationRecord[]>([])
  const [selectedRecord, setSelectedRecord] = useState<ConsultationRecord | null>(null)
  const [citizenData, setCitizenData] = useState<{ name: string; dob: string; gender: string; phone: string } | null>(null)
  const citizenDataRef = useRef(citizenData)

  useEffect(() => {
    citizenDataRef.current = citizenData
  }, [citizenData])

  useEffect(() => {
    registerRole('agent')
  }, [])

  useEffect(() => {
    const id = window.setInterval(() => {
      setCurrentTime(new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' }))
    }, 10_000)
    return () => window.clearInterval(id)
  }, [])

  const applySummary = (payload: SummaryPayload) => {
    const normalizedPhone = (value?: string) => (value || '').replace(/\D/g, '')
    const targetPhone = normalizedPhone(payload.citizenPhone)
    const patch = {
      consultationSummary: payload.consultationSummary,
      isSent: !!payload.isSent,
      deliveryStatus: payload.deliveryStatus ?? (payload.isSent ? 'kakao_sent' : 'pending'),
      sentAt: payload.sentAt || new Date().toISOString(),
    }

    const saved = JSON.parse(localStorage.getItem('consultation_records') || '[]') as ConsultationRecord[]
    const index = targetPhone ? saved.findIndex((record) => normalizedPhone(record.citizenPhone) === targetPhone) : 0

    if (index >= 0) {
      saved[index] = { ...saved[index], ...patch }
    } else {
      saved.unshift({
        id: Date.now().toString(),
        date: new Date().toISOString(),
        citizenName: payload.citizenName || '민원인',
        citizenDob: '미입력',
        citizenGender: '미입력',
        citizenPhone: payload.citizenPhone || '',
        notes: [],
        ...patch,
      })
    }

    localStorage.setItem('consultation_records', JSON.stringify(saved))
    setRecords(saved)
  }

  useEffect(() => {
    const handleCitizenArrived = (payload: { citizenData: { name: string; dob: string; gender: string; phone: string } }) => {
      setNotified(true)
      if (payload?.citizenData) setCitizenData(payload.citizenData)
    }

    const handleSessionReset = () => {
      setNotified(false)
      setCitizenData(null)
    }

    socket.on('citizen_arrived', handleCitizenArrived)
    socket.on('session_reset', handleSessionReset)
    socket.on('consultation_summary_saved', applySummary)
    return () => {
      socket.off('citizen_arrived', handleCitizenArrived)
      socket.off('session_reset', handleSessionReset)
      socket.off('consultation_summary_saved', applySummary)
    }
  }, [])

  const handleEnter = () => {
    onSessionReset?.()
    socket.emit('agent_ready')
    navigate('/agent', { state: { citizenData: citizenDataRef.current } })
  }

  const handleOpenRecords = () => {
    const saved = localStorage.getItem('consultation_records')
    setRecords(saved ? JSON.parse(saved) : [])
    setShowRecords(true)
  }

  const handleCloseRecords = () => {
    setShowRecords(false)
    setSelectedRecord(null)
  }

  const formatDateTimeRange = (startStr: string, endStr?: string) => {
    const start = new Date(startStr)
    const end = endStr ? new Date(endStr) : new Date(start.getTime() + 20 * 60000)
    const date = start.toLocaleDateString('ko-KR', { year: 'numeric', month: '2-digit', day: '2-digit' }).replace(/\. /g, '.').replace(/\.$/, '')
    const startTime = start.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', hour12: false })
    const endTime = end.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', hour12: false })
    return `${date} ${startTime} ~ ${endTime}`
  }

  const getDeliveryLabel = (record: ConsultationRecord) => {
    if (record.deliveryStatus === 'kakao_sent' || record.isSent) return '상담 내용 전송 완료'
    if (record.deliveryStatus === 'clipboard_copied') return '클립보드 복사 완료'
    if (record.deliveryStatus === 'failed') return '전송 실패'
    if (record.deliveryStatus === 'pending' && record.consultationSummary) return '민원인 수령 거부 (내역 저장됨)'
    return '미전송'
  }

  return (
    <div className="relative flex h-screen w-screen flex-col overflow-hidden bg-slate-50 text-slate-900">
      <div className="pointer-events-none absolute inset-0 opacity-40" style={{ backgroundImage: 'radial-gradient(#CBD5E1 1px, transparent 1px)', backgroundSize: '32px 32px' }} />

      <header className="relative z-10 flex items-center justify-between border-b border-slate-200 bg-white/85 px-4 py-4 backdrop-blur md:px-8">
        <div className="flex items-center gap-3">
          <ServiceLogo className="h-14 w-14 md:h-16 md:w-16" />
          <div>
            <h1 className="text-xl font-black tracking-tight md:text-2xl">수어 통역 상담원 화면</h1>
            <p className="text-sm font-bold text-slate-500">민원센터 실시간 수어 통역 상담</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <button onClick={handleOpenRecords} className="rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-black text-slate-700 shadow-sm hover:bg-slate-50">
            상담 기록 열람
          </button>
          <span className="hidden text-lg font-black tabular-nums text-slate-800 sm:block">{currentTime}</span>
        </div>
      </header>

      <main className="relative z-10 flex flex-1 flex-col items-center justify-center gap-10 overflow-y-auto px-4 py-8">
        <div className="flex flex-col items-center text-center">
          <ServiceLogo className="h-36 w-36 md:h-44 md:w-44" />
          <h2 className="mt-8 text-4xl font-black tracking-tight text-slate-950 md:text-6xl">수어 통역 상담 시스템</h2>
          <p className="mt-4 text-base font-semibold text-slate-500 md:text-xl">실시간 민원 상담 및 수어 통역 관리</p>
        </div>

        <section className={`w-full max-w-3xl rounded-[32px] border-2 bg-white p-8 text-center shadow-xl transition-all md:p-12 ${notified ? 'border-emerald-400 shadow-emerald-100' : 'border-slate-100 shadow-slate-200'}`}>
          {notified ? (
            <div className="flex flex-col items-center gap-6">
              <div className="relative flex h-20 w-20 items-center justify-center">
                <span className="absolute inset-0 animate-ping rounded-full bg-emerald-400/30" />
                <span className="relative flex h-16 w-16 items-center justify-center rounded-full bg-emerald-500 text-white">
                  <svg viewBox="0 0 24 24" className="h-8 w-8" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
                    <path d="M13.73 21a2 2 0 0 1-3.46 0" />
                  </svg>
                </span>
              </div>
              <div>
                <h3 className="text-2xl font-black text-slate-950 md:text-3xl">민원인이 도착했습니다</h3>
                <p className="mt-2 text-lg font-bold text-slate-500">{citizenData?.name || '민원인'} 대기 중</p>
              </div>
              <button onClick={handleEnter} className="w-full rounded-2xl bg-blue-600 py-5 text-lg font-black text-white shadow-lg shadow-blue-100 hover:bg-blue-700 active:scale-[0.99]">
                상담 시작하기
              </button>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-6 py-4">
              <div className="h-16 w-16 animate-spin rounded-full border-4 border-slate-100 border-t-blue-500" />
              <div>
                <h3 className="text-3xl font-black text-slate-400">민원인 대기 중</h3>
                <p className="mt-3 text-base font-bold text-slate-400">키오스크 접수 알림을 기다리고 있습니다</p>
              </div>
            </div>
          )}
        </section>
      </main>

      {showRecords && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/40 p-4 backdrop-blur-sm">
          <div className="flex h-[88vh] w-full max-w-5xl flex-col overflow-hidden rounded-3xl border border-slate-100 bg-white shadow-2xl">
            <header className="flex items-center justify-between border-b border-slate-100 px-6 py-5">
              <div className="flex items-center gap-3">
                {selectedRecord && (
                  <button onClick={() => setSelectedRecord(null)} className="rounded-full bg-slate-100 px-3 py-2 text-sm font-black text-slate-600 hover:bg-slate-200">
                    뒤로
                  </button>
                )}
                <h3 className="text-2xl font-black">{selectedRecord ? '상담 상세 기록' : '과거 상담 기록'}</h3>
              </div>
              <button onClick={handleCloseRecords} className="rounded-full bg-slate-100 px-4 py-2 text-sm font-black text-slate-600 hover:bg-slate-200">닫기</button>
            </header>

            <div className="flex-1 overflow-y-auto bg-slate-50 p-5">
              {records.length === 0 ? (
                <div className="flex h-full items-center justify-center text-lg font-black text-slate-300">저장된 상담 기록이 없습니다.</div>
              ) : selectedRecord ? (
                <article className="rounded-2xl border border-slate-100 bg-white p-6 shadow-sm">
                  <div className="grid gap-3 text-sm font-bold text-slate-700 md:grid-cols-2">
                    <p><span className="text-slate-400">민원인:</span> {selectedRecord.citizenName}</p>
                    <p><span className="text-slate-400">상담 시간:</span> {formatDateTimeRange(selectedRecord.date, selectedRecord.endDate)}</p>
                    <p><span className="text-slate-400">성별:</span> {selectedRecord.citizenGender}</p>
                    <p><span className="text-slate-400">연락처:</span> {selectedRecord.citizenPhone || '미입력'}</p>
                  </div>
                  <section className="mt-6">
                    <h4 className="text-lg font-black">상담 내용 요약</h4>
                    <p className="mt-3 whitespace-pre-wrap rounded-xl bg-slate-50 p-4 text-sm font-semibold leading-relaxed text-slate-700">
                      {selectedRecord.consultationSummary || '상담 내용이 아직 저장되지 않았습니다.'}
                    </p>
                  </section>
                  <section className="mt-6">
                    <h4 className="text-lg font-black">민원 메모(상담원 메모)</h4>
                    <div className="mt-3 space-y-2">
                      {selectedRecord.notes.length === 0 ? (
                        <p className="text-sm font-bold text-slate-400">작성된 메모가 없습니다.</p>
                      ) : (
                        selectedRecord.notes.map((note) => (
                          <div key={note.id} className="flex items-start gap-3 rounded-xl bg-slate-50 p-3">
                            <span className={`rounded-full border px-2 py-0.5 text-xs font-black ${TAG_STYLES[note.tag]}`}>{note.tag}</span>
                            <p className="text-sm font-semibold text-slate-700">{note.text}</p>
                          </div>
                        ))
                      )}
                    </div>
                  </section>
                  <p className="mt-6 text-sm font-black text-blue-600">상담 내용 전송여부: {getDeliveryLabel(selectedRecord)}</p>
                </article>
              ) : (
                <div className="space-y-3">
                  {records.map((record) => (
                    <button
                      key={record.id}
                      onClick={() => setSelectedRecord(record)}
                      className="flex w-full flex-col gap-3 rounded-2xl border border-slate-100 bg-white p-5 text-left shadow-sm hover:border-blue-200 md:flex-row md:items-center md:justify-between"
                    >
                      <div>
                        <p className="text-lg font-black text-slate-950">{record.citizenName}</p>
                        <p className="mt-1 text-sm font-bold text-slate-500">{formatDateTimeRange(record.date, record.endDate)}</p>
                        <p className="mt-2 line-clamp-1 text-sm font-semibold text-slate-500">{record.notes.map((note) => note.text).join(' / ') || '작성된 메모 없음'}</p>
                      </div>
                      <span className="rounded-lg bg-blue-50 px-4 py-2 text-sm font-black text-blue-700">상세 기록 보기</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
