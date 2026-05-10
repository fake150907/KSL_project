import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { DoctorNote } from '../types'
import { socket, registerRole } from '../socket' // ✅ 수정됨: registerRole 추가 임포트

interface MedicalRecord {
    id: string;
    date: string;
    endDate?: string;
    patientName: string;
    patientDob: string;
    patientGender: string;
    patientPhone: string;
    notes: DoctorNote[];
    isSent?: boolean;
}

interface DoctorLaunchScreenProps { onSessionReset?: () => void }

const TAG_STYLES: Record<string, string> = {
    증상: 'bg-red-50 text-red-500 border-red-100',
    관찰: 'bg-amber-50 text-amber-600 border-amber-100',
    처방: 'bg-blue-50 text-blue-500 border-blue-100',
}

export default function DoctorLaunchScreen({ onSessionReset }: DoctorLaunchScreenProps) {
    const navigate = useNavigate()
    const [notified, setNotified] = useState(false)
    const [pulse, setPulse] = useState(false)
    const [currentTime, setCurrentTime] = useState(() => new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' }))

    const [showRecords, setShowRecords] = useState(false)
    const [records, setRecords] = useState<MedicalRecord[]>([])
    const [selectedRecord, setSelectedRecord] = useState<MedicalRecord | null>(null)
    const [patientData, setPatientData] = useState<{ name: string, dob: string, gender: string, phone: string } | null>(null)
    // patientData를 ref로도 유지해 handleEnter에서 클로저 문제 없이 최신값 참조
    const patientDataRef = useRef(patientData)
    useEffect(() => { patientDataRef.current = patientData }, [patientData])

    // ✅ 수정됨: 컴포넌트 마운트 시 의사 역할 등록
    useEffect(() => {
        registerRole('doctor')
    }, [])

    useEffect(() => {
        const id = setInterval(() => setCurrentTime(new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })), 10_000)
        return () => clearInterval(id)
    }, [])

    // socket.io: 환자 도착 알림 수신
    useEffect(() => {
        const handlePatientArrived = (payload: { patientData: { name: string, dob: string, gender: string, phone: string } }) => {
            setNotified(true)
            setPulse(true)
            if (payload?.patientData) setPatientData(payload.patientData)
            setTimeout(() => setPulse(false), 1200)
        }

        const handleSessionReset = () => {
            setNotified(false)
            setPatientData(null)
        }

        socket.on('patient_arrived', handlePatientArrived)
        socket.on('session_reset', handleSessionReset)
        return () => {
            socket.off('patient_arrived', handlePatientArrived)
            socket.off('session_reset', handleSessionReset)
        }
    }, [])

    const handleEnter = () => {
        if (onSessionReset) onSessionReset()
        // socket.io: 의사 입장 → 서버 → 환자 키오스크에 전달
        socket.emit('doctor_ready')
        navigate('/doctor', { state: { patientData: patientDataRef.current } })
    }

    const handleOpenRecords = () => {
        const saved = localStorage.getItem('medical_records')
        if (saved) setRecords(JSON.parse(saved))
        setShowRecords(true)
    }

    const handleCloseRecords = () => { setShowRecords(false); setSelectedRecord(null) }
    const formatPhone = (val: string) => { const d = val.replace(/\D/g, ''); if (d.length <= 7) return `${d.slice(0, 3)}-${d.slice(3)}`; return `${d.slice(0, 3)}-${d.slice(3, 7)}-${d.slice(7)}` }

    const formatDateTimeRange = (startStr: string, endStr?: string) => {
        const start = new Date(startStr)
        const end = endStr ? new Date(endStr) : new Date(start.getTime() + 20 * 60000)
        const datePart = start.toLocaleDateString('ko-KR', { year: 'numeric', month: '2-digit', day: '2-digit' }).replace(/\. /g, '.').replace(/\.$/, '')
        const startTimePart = start.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', hour12: false })
        const endTimePart = end.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit', hour12: false })
        return `${datePart} ${startTimePart} ~ ${endTimePart}`
    }

    return (
        <div className="h-screen w-screen flex flex-col bg-slate-50 text-slate-900 overflow-hidden relative">
            <div className="absolute inset-0 pointer-events-none opacity-40" style={{ backgroundImage: `radial-gradient(#CBD5E1 1px, transparent 1px)`, backgroundSize: '32px 32px' }} />

            <div className="flex-shrink-0 flex items-center justify-between px-4 md:px-8 py-4 md:py-6 border-b border-slate-200 bg-white/80 backdrop-blur-md z-10">
                <div className="flex items-center gap-2 md:gap-3">
                    <div className="w-2.5 h-2.5 rounded-full bg-blue-500 animate-pulse hidden sm:block" />
                    <span className="text-xs md:text-sm font-black tracking-widest text-slate-500 uppercase">Dashboard</span>
                </div>
                <div className="flex items-center gap-3 md:gap-6">
                    <button onClick={handleOpenRecords} className="flex items-center gap-1.5 md:gap-2 px-3 md:px-5 py-2 md:py-2.5 rounded-xl text-xs md:text-sm font-bold bg-white border border-slate-200 text-slate-700 hover:bg-slate-50 shadow-sm transition-all">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><polyline points="14 2 14 8 20 8" /></svg>
                        <span className="hidden sm:inline">진료 기록 열람</span>
                        <span className="sm:hidden">기록</span>
                    </button>
                    <span className="text-sm md:text-lg font-black tabular-nums text-slate-800">{currentTime}</span>
                </div>
            </div>

            {/* 대시보드 본문 */}
            <div className="flex-1 flex flex-col items-center justify-center gap-8 md:gap-12 z-10 px-4 md:px-8 overflow-y-auto py-8">
                <div className="flex flex-col items-center gap-3 md:gap-4 text-center">
                    <div className="w-16 h-16 md:w-20 md:h-20 rounded-[24px] md:rounded-[32px] bg-blue-600 flex items-center justify-center mb-2 shadow-xl shadow-blue-100">
                        <svg className="w-8 h-8 md:w-10 md:h-10" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>
                    </div>
                    <h1 className="text-2xl md:text-4xl font-black tracking-tight text-slate-900">수어 통역 시스템</h1>
                    <p className="text-slate-500 text-xs md:text-sm font-medium">실시간 화상 진료 및 환자 관리</p>
                </div>

                <div className={`w-full max-w-lg rounded-[24px] md:rounded-[40px] border-2 p-6 md:p-10 transition-all duration-500 ${notified ? 'bg-white border-emerald-500 shadow-2xl shadow-emerald-100' : 'bg-white border-slate-100 shadow-xl shadow-slate-200'}`}>
                    {notified ? (
                        <div className="flex flex-col items-center gap-6">
                            <div className="relative w-16 h-16 md:w-20 md:h-20 flex items-center justify-center">
                                <div className="absolute inset-0 rounded-full bg-emerald-500 opacity-20" style={{ animation: 'pingOnce 0.8s ease-out' }} />
                                <div className="w-12 h-12 md:w-16 md:h-16 rounded-full bg-emerald-500 flex items-center justify-center shadow-lg">
                                    <svg className="w-6 h-6 md:w-8 md:h-8" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" /><path d="M13.73 21a2 2 0 0 1-3.46 0" /></svg>
                                </div>
                            </div>
                            <div className="text-center">
                                <h2 className="text-2xl md:text-3xl font-black text-slate-900 mb-2">환자가 도착했습니다</h2>
                                <p className="text-base md:text-lg text-slate-500 font-medium">{patientData?.name} 환자님 대기 중</p>
                            </div>
                            <button onClick={handleEnter} className="w-full py-4 md:py-5 rounded-[20px] md:rounded-[24px] text-base md:text-lg font-black text-white bg-blue-600 hover:bg-blue-700 shadow-lg shadow-blue-200 active:scale-[0.98] transition-all">진료실 입장하기</button>
                        </div>
                    ) : (
                        <div className="flex flex-col items-center gap-4 md:gap-6 py-4">
                            <div className="w-12 h-12 md:w-16 md:h-16 rounded-full border-4 border-slate-50 border-t-blue-500 animate-spin" />
                            <div className="text-center">
                                <h2 className="text-xl md:text-2xl font-black text-slate-400 mb-1">환자 대기 중</h2>
                                <p className="text-xs md:text-sm text-slate-400 font-medium">키오스크 접수 알림을 기다리고 있습니다</p>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* 기록 모달 */}
            {showRecords && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4 md:p-8 bg-slate-900/40 backdrop-blur-sm">
                    <div className="flex flex-col w-full max-w-4xl h-[90vh] md:max-h-[85vh] bg-white rounded-[24px] md:rounded-[40px] shadow-2xl overflow-hidden border border-slate-100">
                        <div className="flex items-center justify-between px-6 md:px-8 py-4 md:py-6 border-b border-slate-100 bg-white z-10 flex-shrink-0">
                            <div className="flex items-center gap-3 md:gap-4">
                                {selectedRecord && (
                                    <button onClick={() => setSelectedRecord(null)} className="w-8 h-8 md:w-10 md:h-10 rounded-full bg-slate-50 flex items-center justify-center hover:bg-slate-100 transition-all">
                                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#64748B" strokeWidth="2.5"><polyline points="15 18 9 12 15 6" /></svg>
                                    </button>
                                )}
                                <h2 className="text-lg md:text-2xl font-black text-slate-900">{selectedRecord ? '진료 상세 기록' : '과거 진료 기록'}</h2>
                            </div>
                            <button onClick={handleCloseRecords} className="w-10 h-10 md:w-12 md:h-12 rounded-full bg-slate-50 text-slate-400 flex items-center justify-center hover:bg-slate-100 transition-all">✕</button>
                        </div>

                        <div className="flex-1 overflow-y-auto p-4 md:p-8 flex flex-col gap-4 bg-slate-50/30">
                            {records.length === 0 ? (
                                <div className="py-20 text-center text-slate-300 font-bold text-lg">기록이 없습니다.</div>
                            ) : selectedRecord ? (
                                <div className="bg-white border border-slate-100 rounded-2xl md:rounded-3xl p-6 md:p-10 shadow-sm animate-in fade-in slide-in-from-right-4 duration-300">
                                    <div className="grid grid-cols-1 gap-6 text-sm md:text-lg">
                                        <div className="flex flex-col gap-3 md:gap-4 pb-6 border-b border-slate-100">
                                            <p className="flex flex-col sm:flex-row sm:items-center"><span className="font-bold text-slate-500 w-full sm:w-32 mb-1 sm:mb-0">이름 :</span> <span className="font-black text-slate-900">{selectedRecord.patientName}</span></p>
                                            <p className="flex flex-col sm:flex-row sm:items-center"><span className="font-bold text-slate-500 w-full sm:w-32 mb-1 sm:mb-0">생년월일 :</span> <span className="font-black text-slate-900">{selectedRecord.patientDob}</span></p>
                                            <p className="flex flex-col sm:flex-row sm:items-center"><span className="font-bold text-slate-500 w-full sm:w-32 mb-1 sm:mb-0">성별 :</span> <span className="font-black text-slate-900">{selectedRecord.patientGender}</span></p>
                                            <p className="flex flex-col sm:flex-row sm:items-center"><span className="font-bold text-slate-500 w-full sm:w-32 mb-1 sm:mb-0">연락처 :</span> <span className="font-black text-slate-900">{formatPhone(selectedRecord.patientPhone)}</span></p>
                                            <p className="flex flex-col sm:flex-row sm:items-center"><span className="font-bold text-slate-500 w-full sm:w-32 mb-1 sm:mb-0">진료 시간 :</span> <span className="font-black text-slate-900">{formatDateTimeRange(selectedRecord.date, selectedRecord.endDate)}</span></p>
                                        </div>
                                        <div className="pt-2">
                                            <h4 className="text-lg md:text-xl font-black text-slate-900 mb-4">진단 내용 요약</h4>
                                            <span className="text-gray-600 font-bold text-sm md:text-base">진단내용 보류 (백엔드 대기중)</span>
                                        </div>
                                        <div className="mt-4 pt-6 border-t border-slate-100 flex flex-col gap-4">
                                            <p><span className="font-bold text-slate-500 w-full sm:w-48 block sm:inline-block mb-2 sm:mb-0">진단 메모(의사 메모) :</span></p>
                                            <div className="flex flex-col gap-3">
                                                {selectedRecord.notes.map((n, i) => (
                                                    <div key={i} className="flex items-start gap-3 md:gap-4 bg-slate-50 p-3 md:p-4 rounded-xl md:rounded-2xl border border-slate-100/50">
                                                        <span className={`shrink-0 text-[10px] font-black px-2 py-0.5 rounded-full border ${TAG_STYLES[n.tag]}`}>{n.tag}</span>
                                                        <p className="text-xs md:text-sm text-slate-700 leading-relaxed">{n.text}</p>
                                                    </div>
                                                ))}
                                            </div>
                                            <p className="flex flex-col sm:flex-row sm:items-center mt-2"><span className="font-bold text-slate-500 w-full sm:w-48 mb-1 sm:mb-0">진단 내용 전송여부 :</span> <span className={`font-black ${selectedRecord.isSent ? 'text-emerald-500' : 'text-slate-400'}`}>{selectedRecord.isSent ? 'YES' : 'NO'}</span></p>
                                        </div>
                                    </div>
                                </div>
                            ) : (
                                <div className="flex flex-col gap-3 md:gap-4 animate-in fade-in slide-in-from-left-4 duration-300">
                                    {records.map(r => {
                                        const memoPreview = r.notes.map(n => n.text).join(' / ') || '작성된 메모 없음'
                                        return (
                                            <div key={r.id} className="bg-white border border-slate-100 rounded-xl md:rounded-3xl p-4 md:p-6 shadow-sm flex flex-col sm:flex-row sm:items-center justify-between gap-4 transition-all hover:border-blue-200">
                                                <div className="flex-1 min-w-0">
                                                    <div className="flex flex-wrap items-center gap-2 md:gap-3 text-sm md:text-[17px] font-bold text-slate-800">
                                                        <span>{r.patientName}</span>
                                                        <span className="text-slate-300 text-xs md:text-sm">|</span>
                                                        <span className="text-xs md:text-sm text-slate-400 font-medium">P-{r.id}</span>
                                                        <span className="hidden sm:inline text-slate-300 text-sm">|</span>
                                                        <span className="text-xs md:text-sm text-slate-500">{new Date(r.date).toLocaleString('ko-KR', { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' }).replace(/\. /g, '.').replace(/\.$/, '')}</span>
                                                    </div>
                                                    <div className="mt-2 text-xs md:text-base flex items-center">
                                                        <span className="text-slate-400 font-bold shrink-0">보류중</span>
                                                        <span className="mx-2 text-slate-300 shrink-0">/</span>
                                                        <span className="text-slate-600 font-medium truncate">메모: {memoPreview}</span>
                                                    </div>
                                                </div>
                                                <button onClick={() => setSelectedRecord(r)} className="w-full sm:w-auto px-4 md:px-6 py-2.5 md:py-3 rounded-lg md:rounded-xl text-xs md:text-sm font-black text-blue-600 bg-blue-50 hover:bg-blue-100 transition-colors shrink-0">상세 기록 보기</button>
                                            </div>
                                        )
                                    })}
                                </div>
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