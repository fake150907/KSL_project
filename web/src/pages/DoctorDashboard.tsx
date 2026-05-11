import { useRef, useEffect, useState, useCallback } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import type { ChatMessage as ChatMessageType, DoctorNote } from '../types'
import ChatMessage from '../components/ChatMessage'
import { useSpeechRecognition } from '../hooks/useSpeechRecognition'
import { socket, registerRole } from '../socket'

interface DoctorDashboardProps {
  messages: ChatMessageType[]
  onNewMessage: (msg: ChatMessageType) => void
  onSessionEnd: () => void
  onSessionReset: () => void
  patientName?: string
  patientDob?: string
  patientGender?: string
  patientPhone?: string
}

const TAG_STYLES: Record<DoctorNote['tag'], string> = {
  증상: 'bg-red-50 text-red-600 border-red-200',
  관찰: 'bg-amber-50 text-amber-600 border-amber-200',
  처방: 'bg-blue-50 text-blue-600 border-blue-200',
}

const VOICE_BAR_COLORS = ['#F0ABFC', '#93C5FD', '#86EFAC', '#FDE68A', '#C4B5FD', '#67E8F9']

// TURN 서버 포함 ICE 설정 (모바일 연결 필수)
const ICE_SERVERS: RTCIceServer[] = [
  { urls: 'stun:stun.l.google.com:19302' },
  { urls: 'stun:stun1.l.google.com:19302' },
  {
    urls: 'turn:openrelay.metered.ca:80',
    username: 'openrelayproject',
    credential: 'openrelayproject',
  },
  {
    urls: 'turn:openrelay.metered.ca:443',
    username: 'openrelayproject',
    credential: 'openrelayproject',
  },
  {
    urls: 'turn:openrelay.metered.ca:443?transport=tcp',
    username: 'openrelayproject',
    credential: 'openrelayproject',
  },
]

export default function DoctorDashboard({
  messages,
  onNewMessage,
  onSessionEnd,
  onSessionReset,
  patientName: propPatientName,
  patientDob: propPatientDob,
  patientGender: propPatientGender,
  patientPhone: propPatientPhone,
}: DoctorDashboardProps) {
  const navigate = useNavigate()
  const location = useLocation()
  const chatEndRef = useRef<HTMLDivElement>(null)

  const navState = location.state as { patientData?: { name: string, dob: string, gender: string, phone: string } } | null
  const actualPatientName = propPatientName || navState?.patientData?.name || '익명 환자'
  const actualPatientDob = propPatientDob || navState?.patientData?.dob || '생년월일 미상'
  const actualPatientGender = propPatientGender || navState?.patientData?.gender || '성별 미상'
  const actualPatientPhone = propPatientPhone || navState?.patientData?.phone || '번호 없음'

  const remoteVideoRef = useRef<HTMLVideoElement>(null)
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null)
  const [patientVideoConnected, setPatientVideoConnected] = useState(false)

  const [notes, setNotes] = useState<DoctorNote[]>([])
  const [noteInput, setNoteInput] = useState('')
  const [noteTag, setNoteTag] = useState<DoctorNote['tag']>('증상')
  const [chatInput, setChatInput] = useState('')
  const [showEndConfirm, setShowEndConfirm] = useState(false)
  const [sessionDone, setSessionDone] = useState(false)

  const handleSpeechMessage = useCallback((msg: ChatMessageType) => {
    onNewMessage(msg)
    socket.emit('chat_message', msg)
  }, [onNewMessage])

  const { isActive: isSpeechActive, voiceLevels, start: startSpeech, stop: stopSpeech } = useSpeechRecognition(handleSpeechMessage)

  useEffect(() => {
    registerRole('doctor')
  }, [])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    const handleIncomingMessage = (msg: ChatMessageType) => {
      onNewMessage(msg)
    }
    socket.on('chat_message', handleIncomingMessage)
    return () => {
      socket.off('chat_message', handleIncomingMessage)
    }
  }, [onNewMessage])

  useEffect(() => {
    const pc = new RTCPeerConnection({ iceServers: ICE_SERVERS })
    peerConnectionRef.current = pc
    const pendingCandidates: RTCIceCandidateInit[] = []
    let remoteDescriptionReady = false

    const flushPendingCandidates = async () => {
      while (pendingCandidates.length > 0 && pc.signalingState !== 'closed') {
        const candidate = pendingCandidates.shift()
        if (candidate) {
          await pc.addIceCandidate(new RTCIceCandidate(candidate))
        }
      }
    }

    pc.ontrack = (event) => {
      if (remoteVideoRef.current && event.streams[0]) {
        remoteVideoRef.current.srcObject = event.streams[0]
        setPatientVideoConnected(true)
      }
    }

    pc.oniceconnectionstatechange = () => {
      if (pc.iceConnectionState === 'disconnected' || pc.iceConnectionState === 'failed') {
        setPatientVideoConnected(false)
      }
    }

    pc.onicecandidate = (event) => {
      if (event.candidate) {
        socket.emit('webrtc_ice_candidate', { target: 'kiosk', candidate: event.candidate })
      }
    }

    const handleOffer = async (data: { offer: RTCSessionDescriptionInit }) => {
      if (pc.signalingState === 'closed') return
      try {
        await pc.setRemoteDescription(new RTCSessionDescription(data.offer))
        remoteDescriptionReady = true
        await flushPendingCandidates()
        const answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        socket.emit('webrtc_answer', { target: 'kiosk', answer })
      } catch (err) {
        console.error('Offer 처리 실패:', err)
      }
    }

    const handleCandidate = async (data: { candidate: RTCIceCandidateInit }) => {
      if (data.candidate && pc.signalingState !== 'closed') {
        try {
          if (remoteDescriptionReady) {
            await pc.addIceCandidate(new RTCIceCandidate(data.candidate))
          } else {
            pendingCandidates.push(data.candidate)
          }
        } catch (err) {
          console.error('ICE candidate 추가 실패:', err)
        }
      }
    }

    socket.on('webrtc_offer', handleOffer)
    socket.on('webrtc_ice_candidate', handleCandidate)

    return () => {
      pc.close()
      socket.off('webrtc_offer', handleOffer)
      socket.off('webrtc_ice_candidate', handleCandidate)
    }
  }, [])

  const addNote = () => {
    const text = noteInput.trim()
    if (!text) return
    setNotes((prev) => [...prev, { id: `${Date.now()}`, text, tag: noteTag, timestamp: new Date() }])
    setNoteInput('')
  }
  const handleNoteKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); addNote() }
  }

  const sendChatMessage = () => {
    const text = chatInput.trim()
    if (!text) return
    const msg: ChatMessageType = {
      id: `${Date.now()}-${Math.random()}`,
      sender: 'doctor',
      text,
      timestamp: new Date(),
      label: '의사',
    }
    onNewMessage(msg)                        // 자기 화면 업데이트
    socket.emit('chat_message', msg)         // ✅ 환자에게 전송
    setChatInput('')
  }
  const handleChatKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); sendChatMessage() }
  }

  const timeLabel = (d: Date) => new Date(d).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })

  const handleConfirmEnd = () => {
    try {
      const existingRecords = JSON.parse(localStorage.getItem('medical_records') || '[]')
      const newRecord = {
        id: Date.now().toString(),
        date: new Date().toISOString(),
        endDate: new Date().toISOString(),
        patientName: actualPatientName,
        patientDob: actualPatientDob,
        patientGender: actualPatientGender,
        patientPhone: actualPatientPhone,
        notes,
        diagnosisSummary: '',
        isSent: false,
        deliveryStatus: 'pending',
      }
      localStorage.setItem('medical_records', JSON.stringify([newRecord, ...existingRecords]))
    } catch (e) {
      console.error('기록 저장 실패', e)
    }
    stopSpeech()
    onSessionEnd()
    socket.emit('session_end')
    setSessionDone(true)
    setShowEndConfirm(false)
    navigate('/doctor/launch')
  }

  const handleNewSession = () => { setSessionDone(false); setNotes([]); onSessionReset() }

  return (
    <div className="h-screen w-screen flex flex-col bg-white text-slate-900 overflow-hidden">
      {/* 헤더 */}
      <div className="flex items-center justify-between px-6 py-4 border-b border-slate-200 flex-shrink-0 bg-white">
        <div className="flex items-center gap-3">
          <span className="text-lg font-bold text-slate-800">
            진료 대시보드 - {actualPatientName} 환자
            <span className="text-sm text-slate-500 font-normal ml-2">({actualPatientGender}, {actualPatientDob})</span>
          </span>
        </div>
        <div className="flex items-center gap-4">
          {sessionDone ? (
            <>
              <span className="flex items-center gap-2 text-sm font-semibold text-slate-400">
                <span className="w-2 h-2 rounded-full bg-slate-300" /> 진료 종료됨
              </span>
              <button onClick={handleNewSession} className="px-4 py-2 rounded-lg text-sm font-bold bg-blue-50 border border-blue-200 text-blue-600 hover:bg-blue-100">
                새 진료 시작
              </button>
            </>
          ) : (
            <>
              <span className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                <span className="text-sm font-semibold text-blue-600">진료 중</span>
              </span>
              <button onClick={() => setShowEndConfirm(true)} className="px-4 py-2 rounded-lg text-sm font-bold bg-red-50 border border-red-200 text-red-600 hover:bg-red-100">
                진료 끝내기
              </button>
            </>
          )}
        </div>
      </div>

      <div className="flex flex-1 min-h-0">
        {/* 사이드바: 환자 영상 + 메모 */}
        <div className="w-[400px] flex-shrink-0 flex flex-col border-r border-slate-200 bg-slate-50/50 overflow-hidden">
          <div className="p-4">
            <div className="flex items-center justify-between mb-3">
              <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">환자 라이브</span>
              <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold border ${patientVideoConnected ? 'bg-emerald-50 text-emerald-600 border-emerald-200' : 'bg-slate-100 text-slate-500 border-slate-200'}`}>
                {patientVideoConnected ? '연결됨' : '대기 중'}
              </span>
            </div>
            <div className="aspect-video bg-slate-900 rounded-2xl overflow-hidden shadow-inner relative">
              <video
                ref={remoteVideoRef}
                autoPlay
                playsInline
                className="w-full h-full object-cover"
                style={{ display: patientVideoConnected ? 'block' : 'none', transform: 'scaleX(-1)' }}
              />
              {!patientVideoConnected && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-slate-600">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                    <polygon points="23 7 16 12 23 17 23 7"/>
                    <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
                  </svg>
                  <span className="text-xs font-medium text-center">환자 카메라 대기 중</span>
                </div>
              )}
            </div>
          </div>

          {/* 진료 메모 */}
          <div className="flex-1 flex flex-col overflow-hidden px-4 pb-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-bold text-slate-700">진료 메모</span>
              <span className="text-xs text-slate-400">{notes.length}개</span>
            </div>
            <div className="flex gap-2 mb-3">
              <input
                value={noteInput}
                onChange={(e) => setNoteInput(e.target.value)}
                onKeyDown={handleNoteKeyDown}
                placeholder="메모 입력..."
                className="flex-1 bg-white border border-slate-200 rounded-xl px-3 py-2 text-sm text-slate-900 outline-none focus:border-blue-400"
              />
              <select value={noteTag} onChange={(e) => setNoteTag(e.target.value as DoctorNote['tag'])} className="bg-white border border-slate-200 rounded-xl px-2 text-xs text-slate-600 outline-none">
                <option value="증상">증상</option>
                <option value="관찰">관찰</option>
                <option value="처방">처방</option>
              </select>
              <button onClick={addNote} className="w-9 h-9 flex items-center justify-center rounded-xl bg-blue-600 text-white font-bold hover:bg-blue-700">+</button>
            </div>
            <div className="flex-1 overflow-y-auto flex flex-col gap-2">
              {notes.map((note) => (
                <div key={note.id} className="p-3 rounded-2xl border border-slate-100 bg-white shadow-sm">
                  <div className="flex items-start justify-between gap-2">
                    <p className="text-sm text-slate-800 font-medium leading-snug">{note.text}</p>
                    <span className={`flex-shrink-0 text-[10px] font-bold px-2 py-0.5 rounded-full border ${TAG_STYLES[note.tag]}`}>{note.tag}</span>
                  </div>
                  <span className="text-[10px] text-slate-400 mt-2 block">{timeLabel(note.timestamp)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* 채팅 */}
        <div className="flex-1 flex flex-col min-w-0 bg-white">
          <div className="flex-1 overflow-y-auto px-6 py-6 flex flex-col gap-4">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full gap-3 text-slate-300">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                <p className="text-sm font-medium">대화가 시작되면 여기에 표시됩니다</p>
              </div>
            ) : (
              messages.map((msg) => <ChatMessage key={msg.id} message={msg} />)
            )}
            <div ref={chatEndRef} />
          </div>

          {!sessionDone && (
            <div className="p-4 border-t border-slate-100 bg-slate-50/30">
              {isSpeechActive && (
                <div className="h-10 flex items-end gap-1 px-4 mb-3">
                  {voiceLevels.map((l, i) => (
                    <div key={i} className="flex-1 rounded-full transition-all" style={{ height: `${Math.round(l * 100)}%`, minHeight: '10%', background: l > 0.2 ? VOICE_BAR_COLORS[i % 6] : '#E2E8F0' }} />
                  ))}
                </div>
              )}
              <div className="flex items-center gap-3">
                <input
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  onKeyDown={handleChatKeyDown}
                  placeholder="환자에게 전달할 메시지 입력..."
                  className="flex-1 bg-white border border-slate-200 rounded-2xl px-5 py-3 text-sm text-slate-900 outline-none focus:border-blue-500 shadow-sm"
                />
                <button onClick={sendChatMessage} disabled={!chatInput.trim()} className="px-6 py-3 rounded-2xl bg-blue-600 text-white font-bold text-sm disabled:opacity-30 hover:bg-blue-700 shadow-md">
                  전송
                </button>
              </div>
              <button
                onClick={isSpeechActive ? stopSpeech : startSpeech}
                className={`w-full mt-3 flex items-center justify-center gap-2 py-2 text-xs font-bold rounded-xl transition-all ${isSpeechActive ? 'bg-amber-50 text-amber-600 border border-amber-200' : 'bg-slate-100 text-slate-500 border border-slate-200 hover:bg-slate-200'}`}
              >
                {isSpeechActive ? '음성 인식 중지' : '마이크 켜기 (음성 인식)'}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* 진료 종료 팝업 */}
      {showEndConfirm && (
        <div className="fixed inset-0 flex items-center justify-center z-50 bg-slate-900/60 backdrop-blur-sm">
          <div className="bg-white rounded-[32px] p-8 max-w-sm w-full mx-4 shadow-2xl border border-slate-100">
            <h3 className="text-xl font-black text-slate-900 mb-2">진료를 종료하시겠습니까?</h3>
            <p className="text-sm text-slate-500 mb-8 leading-relaxed">작성한 진료 메모와 환자 정보가 기록에 저장됩니다.</p>
            <div className="flex gap-3">
              <button onClick={() => setShowEndConfirm(false)} className="flex-1 py-4 rounded-2xl text-sm font-bold text-slate-500 bg-slate-100 hover:bg-slate-200 transition-all">취소</button>
              <button onClick={handleConfirmEnd} className="flex-1 py-4 rounded-2xl text-sm font-bold text-white bg-red-500 hover:bg-red-600 shadow-lg shadow-red-200 transition-all">종료 및 저장</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
