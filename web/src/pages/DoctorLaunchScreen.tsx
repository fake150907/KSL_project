import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { DoctorNote } from '../types'

const NOTIFY_CHANNEL = 'patient-session-notify'

interface MedicalRecord {
  id: string; date: string; patientName: string; patientDob: string; patientGender: string; patientPhone: string; notes: DoctorNote[]
}

interface DoctorLaunchScreenProps { onSessionReset?: () => void }

const TAG_STYLES: Record<string, string> = {
  증상: 'bg-red-50 text-red-500 border-red-100',
  관찰: 'bg-amber-50 text-amber-600 border-amber-100',
  처방: 'bg-blue-50 text-blue-500 border-blue-100',
}

export default function DoctorLaunchScreen({ onSessionReset }: DoctorLaunchScreenProps) {
  const navigate = useNavigate(); const channelRef = useRef<BroadcastChannel | null>(null)
  const [notified, setNotified] = useState(false); const [pulse, setPulse] = useState(false)
  const [currentTime, setCurrentTime] = useState(() => new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' }))
  const [showRecords, setShowRecords] = useState(false); const [records, setRecords] = useState<MedicalRecord[]>([])
  const [patientData, setPatientData] = useState<{name: string, dob: string, gender: string, phone: string} | null>(null)

  useEffect(() => {
    const id = setInterval(() => setCurrentTime(new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })), 10_000)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    const ch = new BroadcastChannel(NOTIFY_CHANNEL); channelRef.current = ch
    ch.onmessage = (e) => {
      if (e.data?.type === 'patient_arrived') {
        setNotified(true); setPulse(true); if (e.data.payload) setPatientData(e.data.payload)
        setTimeout(() => setPulse(false), 1200)
      }
      if (e.data?.type === 'session_reset') { setNotified(false); setPatientData(null) }
    }
    return () => ch.close()
  }, [])

  const handleEnter = () => { if (onSessionReset) onSessionReset(); channelRef.current?.postMessage({ type: 'doctor_ready' }); navigate('/doctor', { state: { patientData } }) }
  const handleOpenRecords = () => { const saved = localStorage.getItem('medical_records'); if (saved) setRecords(JSON.parse(saved)); setShowRecords(true) }
  const formatPhone = (val: string) => { const d = val.replace(/\D/g, ''); if (d.length <= 7) return `${d.slice(0,3)}-${d.slice(3)}`; return `${d.slice(0,3)}-${d.slice(3,7)}-${d.slice(7)}` }

  return (
    <div className="h-screen w-screen flex flex-col bg-slate-50 text-slate-900 overflow-hidden relative">
      {/* 배경 장식 */}
      <div className="absolute inset-0 pointer-events-none opacity-40" style={{ backgroundImage: `radial-gradient(#CBD5E1 1px, transparent 1px)`, backgroundSize: '32px 32px' }} />

      {/* 헤더 */}
      <div className="flex-shrink-0 flex items-center justify-between px-8 py-6 border-b border-slate-200 bg-white/80 backdrop-blur-md z-10">
        <div className="flex items-center gap-3">
          <div className="w-2.5 h-2.5 rounded-full bg-blue-500 animate-pulse" />
          <span className="text-sm font-black tracking-widest text-slate-500 uppercase">Doctor Dashboard</span>
        </div>
        <div className="flex items-center gap-6">
          <button onClick={handleOpenRecords} className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 shadow-sm transition-all">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
            진료 기록 열람
          </button>
          <span className="text-lg font-black tabular-nums text-slate-800">{currentTime}</span>
        </div>
      </div>

      {/* 중앙 내용 */}
      <div className="flex-1 flex flex-col items-center justify-center gap-12 z-10 px-8">
        <div className="flex flex-col items-center gap-4">
          <div className="w-20 h-20 rounded-[32px] bg-blue-600 flex items-center justify-center mb-2 shadow-xl shadow-blue-100">
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
          </div>
          <h1 className="text-4xl font-black tracking-tight text-slate-900">수어 통역 시스템</h1>
          <p className="text-slate-500 text-sm font-medium">실시간 화상 진료 및 환자 관리</p>
        </div>

        <div className={`w-full max-w-lg rounded-[40px] border-2 p-10 transition-all duration-500 ${notified ? 'bg-white border-emerald-500 shadow-2xl shadow-emerald-100' : 'bg-white border-slate-100 shadow-xl shadow-slate-200'}`}>
          {notified ? (
            <div className="flex flex-col items-center gap-6">
              <div className="relative w-20 h-20 flex items-center justify-center">
                <div className="absolute inset-0 rounded-full bg-emerald-500 opacity-20" style={{ animation: pulse ? 'pingOnce 0.8s ease-out' : 'none' }} />
                <div className="w-16 h-16 rounded-full bg-emerald-500 flex items-center justify-center shadow-lg">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"/><path d="M13.73 21a2 2 0 0 1-3.46 0"/></svg>
                </div>
              </div>
              <div className="text-center">
                <h2 className="text-3xl font-black text-slate-900 mb-2">환자가 도착했습니다</h2>
                <p className="text-lg text-slate-500 font-medium">{patientData?.name} 환자님 대기 중</p>
              </div>
              <button onClick={handleEnter} className="w-full py-5 rounded-[24px] text-lg font-black text-white bg-blue-600 hover:bg-blue-700 shadow-lg shadow-blue-200 active:scale-[0.98] transition-all">진료실 입장하기</button>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-6 py-4">
              <div className="w-16 h-16 rounded-full border-4 border-slate-50 border-t-blue-500 animate-spin" />
              <div className="text-center">
                <h2 className="text-2xl font-black text-slate-400 mb-1">환자 대기 중</h2>
                <p className="text-sm text-slate-400 font-medium">키오스크 접수 알림을 기다리고 있습니다</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* 기록 모달 */}
      {showRecords && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-8 bg-slate-900/40 backdrop-blur-sm">
          <div className="flex flex-col w-full max-w-4xl max-h-[85vh] bg-white rounded-[40px] shadow-2xl overflow-hidden border border-slate-100">
            <div className="flex items-center justify-between px-8 py-6 border-b border-slate-100">
              <h2 className="text-2xl font-black text-slate-900">과거 진료 기록</h2>
              <button onClick={() => setShowRecords(false)} className="w-12 h-12 rounded-full bg-slate-50 text-slate-400 flex items-center justify-center hover:bg-slate-100 transition-all">✕</button>
            </div>
            <div className="flex-1 overflow-y-auto p-8 flex flex-col gap-6 bg-slate-50/30">
              {records.length === 0 ? (
                <div className="py-20 text-center text-slate-300 font-bold text-lg">기록이 없습니다.</div>
              ) : (
                records.map(r => (
                  <div key={r.id} className="bg-white border border-slate-100 rounded-3xl p-6 shadow-sm">
                    <div className="flex justify-between items-start mb-6 pb-6 border-b border-slate-50">
                      <div>
                        <h3 className="text-xl font-black text-slate-900 mb-1">{r.patientName} <span className="text-sm font-normal text-slate-400 ml-2">({r.patientGender}, {r.patientDob})</span></h3>
                        <p className="text-sm font-bold text-blue-500">{formatPhone(r.patientPhone)}</p>
                      </div>
                      <span className="text-sm font-bold text-slate-400 text-right">{new Date(r.date).toLocaleDateString()}<br/>{new Date(r.date).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})}</span>
                    </div>
                    <div className="flex flex-col gap-3">
                      {r.notes.map((n, i) => (
                        <div key={i} className="flex items-start gap-3 bg-slate-50 p-4 rounded-2xl border border-slate-100/50">
                          <span className={`shrink-0 text-[10px] font-black px-2 py-0.5 rounded-full border ${TAG_STYLES[n.tag]}`}>{n.tag}</span>
                          <p className="text-sm text-slate-700 leading-relaxed">{n.text}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      )}

      <style>{`
        @keyframes pingOnce { 0% { transform: scale(1); opacity: 0.5; } 70% { transform: scale(1.8); opacity: 0; } 100% { transform: scale(1.8); opacity: 0; } }
      `}</style>
    </div>
  )
}