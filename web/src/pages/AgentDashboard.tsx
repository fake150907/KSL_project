import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { KeyboardEvent, WheelEvent } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import type { ChatMessage as ChatMessageType, AgentNote } from '../types'
import ChatMessage from '../components/ChatMessage'
import { useSpeechRecognition } from '../hooks/useSpeechRecognition'
import { registerRole, socket } from '../socket'
import { maskName } from '../components/hangul'

interface AgentDashboardProps {
  messages: ChatMessageType[]
  onNewMessage: (msg: ChatMessageType) => void
  onSessionEnd: () => void
  onSessionReset: () => void
  citizenName?: string
  citizenDob?: string
  citizenGender?: string
  citizenPhone?: string
}

type WelfareThemeMode = 'light' | 'dark'

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
]

const VOICE_BAR_COLORS = ['#2563EB', '#38BDF8', '#22C55E', '#F59E0B', '#6366F1', '#14B8A6']

const TAG_STYLES: Record<AgentNote['tag'], string> = {
  문의: 'bg-blue-50 text-blue-700 border-blue-200',
  확인: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  처리: 'bg-violet-50 text-violet-700 border-violet-200',
}

const TAG_STYLES_DARK: Record<AgentNote['tag'], string> = {
  문의: 'bg-blue-900/40 text-blue-300 border-blue-700',
  확인: 'bg-emerald-900/40 text-emerald-300 border-emerald-700',
  처리: 'bg-violet-900/40 text-violet-300 border-violet-700',
}

const QUICK_REPLIES = [
  { title: '첫 인사', text: '안녕하세요. 어떻게 도와드릴까요?' },
  { title: '분실 접수', text: '복지카드를 잃어버리셨군요. 복지카드 분실 신고로 도와드릴게요. 재발급을 원하시나요?' },
  { title: '재발급 가능', text: '가능합니다. 본인 확인부터 도와드릴게요. 다른 신분증이 있으신가요?' },
  { title: '확인 대기', text: '잠시 확인하겠습니다.' },
  { title: '장소 확인', text: '확인 완료했습니다. 어디서 잃어버리셨나요?' },
  { title: '수수료 안내', text: '재발급 수수료는 5,000원입니다.' },
  { title: '면제 안내', text: '네, 면제 적용됩니다. 잠시 후 직원이 신청서 작성을 도와드리겠습니다.' },
  { title: '수령 안내', text: '신청 후 약 7~10일 뒤 등기우편으로 도착합니다.' },
]

export default function AgentDashboard({
  messages,
  onNewMessage,
  onSessionEnd,
  onSessionReset,
  citizenName: propCitizenName,
  citizenDob: propCitizenDob,
  citizenGender: propCitizenGender,
  citizenPhone: propCitizenPhone,
}: AgentDashboardProps) {
  const navigate = useNavigate()
  const location = useLocation()
  const dashboardScrollRef = useRef<HTMLElement>(null)
  const chatListRef = useRef<HTMLDivElement>(null)
  const remoteVideoRef = useRef<HTMLVideoElement>(null)
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null)

  const navState = location.state as { citizenData?: { name: string; dob: string; gender: string; phone: string } } | null
  const storedCitizenData = (() => {
    try {
      return JSON.parse(sessionStorage.getItem('current_citizen_data') || 'null') as { name?: string; dob?: string; gender?: string; phone?: string } | null
    } catch {
      return null
    }
  })()
  const activeCitizenData = navState?.citizenData || storedCitizenData
  const citizenName = activeCitizenData?.name || propCitizenName || '민원인'
  const citizenDob = activeCitizenData?.dob || propCitizenDob || '미입력'
  const citizenGender = activeCitizenData?.gender || propCitizenGender || '미입력'
  const citizenPhone = activeCitizenData?.phone || propCitizenPhone || ''
  const maskedCitizenName = maskName(citizenName)

  const [videoConnected, setVideoConnected] = useState(false)
  const [notes, setNotes] = useState<AgentNote[]>([])
  const [noteInput, setNoteInput] = useState('')
  const [noteTag, setNoteTag] = useState<AgentNote['tag']>('문의')
  const [replyInput, setReplyInput] = useState('')
  const [showEndConfirm, setShowEndConfirm] = useState(false)
  const [sessionDone, setSessionDone] = useState(false)
  const [taskType, setTaskType] = useState('복지카드 재발급')
  const [welfareThemeMode, setWelfareThemeMode] = useState<WelfareThemeMode>('light')

  const isDarkMode = welfareThemeMode === 'dark'

  const handleSpeechMessage = useCallback((msg: ChatMessageType) => {
    onNewMessage(msg)
    socket.emit('chat_message', msg)
  }, [onNewMessage])

  const { isActive: isSpeechActive, voiceLevels, start: startSpeech, stop: stopSpeech } = useSpeechRecognition(handleSpeechMessage)

  const latestCitizenText = useMemo(
    () => [...messages].reverse().find((message) => message.sender === 'citizen')?.text || '',
    [messages],
  )
  const latestAgentText = useMemo(
    () => [...messages].reverse().find((message) => message.sender === 'agent')?.text || '',
    [messages],
  )

  useEffect(() => {
    registerRole('agent')
  }, [])

  useEffect(() => {
    const chatList = chatListRef.current
    if (!chatList) return
    chatList.scrollTop = chatList.scrollHeight
  }, [messages])

  useEffect(() => {
    socket.emit('agent_voice_status', { active: isSpeechActive })

    return () => {
      socket.emit('agent_voice_status', { active: false })
    }
  }, [isSpeechActive])

  useEffect(() => {
    if (!isSpeechActive) return
    socket.emit('agent_voice_status', { active: true, levels: voiceLevels })
  }, [isSpeechActive, voiceLevels])

  useEffect(() => {
    const handleIncomingMessage = (msg: ChatMessageType) => onNewMessage(msg)
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
        if (candidate) await pc.addIceCandidate(new RTCIceCandidate(candidate))
      }
    }

    pc.ontrack = (event) => {
      if (remoteVideoRef.current && event.streams[0]) {
        remoteVideoRef.current.srcObject = event.streams[0]
        setVideoConnected(true)
      }
    }

    pc.oniceconnectionstatechange = () => {
      if (pc.iceConnectionState === 'disconnected' || pc.iceConnectionState === 'failed') {
        setVideoConnected(false)
      }
    }

    pc.onicecandidate = (event) => {
      if (event.candidate) socket.emit('webrtc_ice_candidate', { target: 'kiosk', candidate: event.candidate })
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
      } catch (error) {
        console.error('Offer 처리 실패:', error)
      }
    }

    const handleCandidate = async (data: { candidate: RTCIceCandidateInit }) => {
      if (!data.candidate || pc.signalingState === 'closed') return
      try {
        if (remoteDescriptionReady) {
          await pc.addIceCandidate(new RTCIceCandidate(data.candidate))
        } else {
          pendingCandidates.push(data.candidate)
        }
      } catch (error) {
        console.error('ICE candidate 추가 실패:', error)
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

  const addNote = (tagOverride?: AgentNote['tag'], textOverride?: string) => {
    const text = (textOverride ?? noteInput).trim()
    if (!text) return
    const newNote: AgentNote = {
      id: `${Date.now()}`,
      text,
      tag: tagOverride ?? noteTag,
      timestamp: new Date(),
    }
    setNotes((prev) => [newNote, ...prev])
    if (!textOverride) setNoteInput('')
  }

  const handleNoteKeyDown = (event: KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      addNote()
    }
  }

  const sendReply = () => {
    const text = replyInput.trim()
    if (!text) return
    const msg: ChatMessageType = {
      id: `${Date.now()}-${Math.random()}`,
      sender: 'agent',
      text,
      timestamp: new Date(),
      label: '상담원',
    }
    onNewMessage(msg)
    socket.emit('chat_message', msg)
    setReplyInput('')
  }

  const handleReplyKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault()
      sendReply()
    }
  }

  const timeLabel = (date: Date) => new Date(date).toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })

  const handleConfirmEnd = () => {
    try {
      const existingRecords = JSON.parse(localStorage.getItem('consultation_records') || '[]')
      const newRecord = {
        id: Date.now().toString(),
        date: new Date().toISOString(),
        endDate: new Date().toISOString(),
        citizenName: citizenName,
        citizenDob: citizenDob,
        citizenGender: citizenGender,
        citizenPhone: citizenPhone,
        notes,
        consultationSummary: messages.map((message) => `${message.sender === 'agent' ? '상담원' : '민원인'}: ${message.text}`).join('\n'),
        isSent: false,
        deliveryStatus: 'pending',
      }
      localStorage.setItem('consultation_records', JSON.stringify([newRecord, ...existingRecords]))
    } catch (error) {
      console.error('상담 기록 저장 실패:', error)
    }
    stopSpeech()
    onSessionEnd()
    socket.emit('session_end')
    setSessionDone(true)
    setShowEndConfirm(false)
    navigate('/agent/launch')
  }

  const handleNewSession = () => {
    setSessionDone(false)
    setNotes([])
    onSessionReset()
  }

  const handleDashboardWheel = useCallback((event: WheelEvent<HTMLElement>) => {
    const scrollRoot = dashboardScrollRef.current
    if (!scrollRoot || Math.abs(event.deltaY) < 1) return

    let node = event.target as HTMLElement | null
    while (node && node !== scrollRoot) {
      const style = window.getComputedStyle(node)
      const canScrollNode = /(auto|scroll)/.test(style.overflowY) && node.scrollHeight > node.clientHeight + 1

      if (canScrollNode) {
        const atTop = node.scrollTop <= 0
        const atBottom = node.scrollTop + node.clientHeight >= node.scrollHeight - 1
        const canScrollInWheelDirection = event.deltaY < 0 ? !atTop : !atBottom

        if (canScrollInWheelDirection) return
      }

      node = node.parentElement
    }

    if (scrollRoot.scrollHeight <= scrollRoot.clientHeight + 1) return
    event.preventDefault()
    scrollRoot.scrollTop += event.deltaY
  }, [])

  return (
    <div data-theme={isDarkMode ? 'dark' : 'light'} className={`fixed inset-0 flex flex-col overflow-hidden transition-colors duration-300 ${isDarkMode ? 'bg-[#0f172a] text-slate-50' : 'bg-slate-100 text-slate-900'}`}>
      <header className={`z-10 flex-shrink-0 border-b backdrop-blur transition-colors duration-300 ${isDarkMode ? 'border-[#263244] bg-[#121b2b]/95' : 'border-slate-200 bg-white/95'}`}>
        <div className="flex flex-col gap-3 px-4 py-4 md:flex-row md:items-center md:justify-between md:px-6">
          <div className="min-w-0">
            <p className="break-words text-xs font-black uppercase tracking-[0.18em] text-blue-600 sm:tracking-[0.24em]">Civil Sign Interpreter</p>
            <h1 className={`mt-1 break-words text-xl font-black tracking-tight md:text-2xl ${isDarkMode ? 'text-slate-50' : 'text-slate-950'}`}>수어 통역 상담원 화면</h1>
            <p className={`break-words text-sm font-semibold ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>{maskedCitizenName} 민원인 상담 중</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <div className={`flex items-center overflow-hidden rounded-lg border shadow-sm ${isDarkMode ? 'border-[#324155] bg-[#172235]' : 'border-[#d8e0ea] bg-[#f8fafc]'}`}>
              <span className={`border-r px-3 py-2 text-sm font-black ${isDarkMode ? 'border-[#324155] text-slate-200' : 'border-[#d8e0ea] text-slate-600'}`}>화면 모드</span>
              {(['light', 'dark'] as const).map((mode) => (
                <button
                  key={mode}
                  type="button"
                  onClick={() => setWelfareThemeMode(mode)}
                  className={`flex h-10 min-w-14 items-center justify-center border-r px-3 text-sm font-black transition-all last:border-r-0 active:scale-95 ${isDarkMode ? 'border-[#324155]' : 'border-[#d8e0ea]'} ${welfareThemeMode === mode ? 'bg-[#2563eb] text-white shadow-inner' : isDarkMode ? 'bg-[#111827] text-slate-100 hover:bg-[#1f2a3d]' : 'bg-white text-slate-700 hover:bg-[#f1f5f9]'}`}
                >
                  {mode === 'light' ? '라이트' : '다크'}
                </button>
              ))}
            </div>
            <span className={`rounded-full border px-3 py-1 text-xs font-black ${sessionDone ? (isDarkMode ? 'border-slate-700 bg-slate-800 text-slate-400' : 'border-slate-200 bg-slate-50 text-slate-500') : (isDarkMode ? 'border-emerald-700 bg-emerald-900/40 text-emerald-400' : 'border-emerald-200 bg-emerald-50 text-emerald-700')}`}>
              {sessionDone ? '상담 종료됨' : '양방향 통역 중'}
            </span>
            {sessionDone ? (
              <button onClick={handleNewSession} className={`rounded-lg border px-4 py-2 text-sm font-black ${isDarkMode ? 'border-blue-700 bg-blue-900/40 text-blue-400 hover:bg-blue-900/60' : 'border-blue-200 bg-blue-50 text-blue-700 hover:bg-blue-100'}`}>
                새 상담 시작
              </button>
            ) : (
              <button onClick={() => setShowEndConfirm(true)} className={`rounded-lg border px-4 py-2 text-sm font-black ${isDarkMode ? 'border-red-700 bg-red-900/40 text-red-400 hover:bg-red-900/60' : 'border-red-200 bg-red-50 text-red-600 hover:bg-red-100'}`}>
                상담 끝내기
              </button>
            )}
          </div>
        </div>
      </header>

      <main ref={dashboardScrollRef} onWheelCapture={handleDashboardWheel} className="grid min-h-0 min-w-0 flex-1 auto-rows-max grid-cols-1 gap-3 overflow-y-auto overflow-x-hidden overscroll-auto p-3 md:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)] xl:grid-cols-[minmax(0,0.8fr)_minmax(0,1.15fr)_minmax(0,0.9fr)]">
        {/* 민원인 수어 영상 + 메모 */}
        <section className={`min-w-0 overflow-hidden rounded-lg border-[3px] transition-colors duration-300 ${isDarkMode ? 'border-[#2b3a50] bg-[#121b2b]' : 'border-slate-200 bg-white'}`}>
          <div className={`flex flex-wrap items-center justify-between gap-2 border-b px-4 py-3 ${isDarkMode ? 'border-[#263244]' : 'border-slate-100'}`}>
            <h2 className={`text-base font-black ${isDarkMode ? 'text-slate-50' : ''}`}>민원인 수어 영상</h2>
            <span className={`rounded-full border px-2.5 py-1 text-xs font-black ${videoConnected ? (isDarkMode ? 'border-emerald-700 bg-emerald-900/40 text-emerald-400' : 'border-emerald-200 bg-emerald-50 text-emerald-700') : (isDarkMode ? 'border-slate-700 bg-slate-800 text-slate-400' : 'border-slate-200 bg-slate-50 text-slate-500')}`}>
              {videoConnected ? '연결됨' : '대기 중'}
            </span>
          </div>
          <div className="bg-slate-950 p-3">
            <div className="relative aspect-video overflow-hidden rounded-md bg-slate-900">
              <video
                ref={remoteVideoRef}
                autoPlay
                playsInline
                className="h-full w-full object-contain"
                style={{ display: videoConnected ? 'block' : 'none', transform: 'scaleX(-1)' }}
              />
              {!videoConnected && (
                <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 text-slate-400">
                  <div className="h-10 w-10 rounded-full border-2 border-slate-700 border-t-blue-500 animate-spin" />
                  <p className="text-sm font-bold">민원인 카메라 대기 중</p>
                </div>
              )}
            </div>
          </div>

          <div className="space-y-3 p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className={`text-sm font-black ${isDarkMode ? 'text-slate-300' : 'text-slate-700'}`}>민원 메모</h3>
              <span className={`text-xs font-bold ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>{notes.length}개</span>
            </div>
            <div className="grid grid-cols-[minmax(0,1fr)_40px] gap-2 sm:grid-cols-[minmax(0,1fr)_minmax(78px,92px)_40px]">
              <input
                value={noteInput}
                onChange={(event) => setNoteInput(event.target.value)}
                onKeyDown={handleNoteKeyDown}
                placeholder="메모 입력..."
                className={`h-10 min-w-0 rounded-lg border px-3 text-sm font-semibold outline-none ${isDarkMode ? 'border-[#334155] bg-[#111827] text-slate-100 placeholder:text-slate-500 focus:border-blue-500' : 'border-slate-200 bg-white focus:border-blue-400'}`}
              />
              <select value={noteTag} onChange={(event) => setNoteTag(event.target.value as AgentNote['tag'])} className={`h-10 min-w-0 rounded-lg border px-2 text-sm font-bold outline-none max-sm:col-span-2 ${isDarkMode ? 'border-[#334155] bg-[#111827] text-slate-100' : 'border-slate-200 bg-white'}`}>
                <option value="문의">문의</option>
                <option value="확인">확인</option>
                <option value="처리">처리</option>
              </select>
              <button onClick={() => addNote()} className="h-10 rounded-lg bg-blue-600 text-lg font-black text-white hover:bg-blue-700">+</button>
            </div>
            <div className="max-h-[240px] space-y-2 overflow-y-auto pr-1">
              {notes.map((note) => (
                <article key={note.id} className={`rounded-lg border p-3 ${isDarkMode ? 'border-[#263244] bg-[#0f172a]' : 'border-slate-100 bg-slate-50'}`}>
                  <div className="flex items-start justify-between gap-2">
                    <p className={`min-w-0 break-words text-sm font-bold leading-relaxed ${isDarkMode ? 'text-slate-200' : 'text-slate-800'}`}>{note.text}</p>
                    <span className={`shrink-0 rounded-full border px-2 py-0.5 text-[11px] font-black ${isDarkMode ? TAG_STYLES_DARK[note.tag] : TAG_STYLES[note.tag]}`}>{note.tag}</span>
                  </div>
                  <span className={`mt-2 block text-[11px] font-semibold ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>{timeLabel(note.timestamp)}</span>
                </article>
              ))}
            </div>
          </div>
          <div className={`border-t p-4 ${isDarkMode ? 'border-[#263244]' : 'border-slate-100'}`}>
            <div className="grid gap-2">
              <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4 xl:grid-cols-2">
                <button onClick={() => addNote('처리', `${taskType} 분실 접수 진행`)} className={`whitespace-normal rounded-lg border px-3 py-2 text-sm font-black ${isDarkMode ? 'border-[#334155] text-slate-200 hover:bg-[#1e293b]' : 'border-slate-200 hover:bg-slate-50'}`}>분실 접수</button>
                <button onClick={() => addNote('확인', '신분증 본인 확인 완료')} className={`whitespace-normal rounded-lg border px-3 py-2 text-sm font-black ${isDarkMode ? 'border-[#334155] text-slate-200 hover:bg-[#1e293b]' : 'border-slate-200 hover:bg-slate-50'}`}>본인 확인</button>
                <button onClick={() => addNote('처리', '재발급 수수료 면제 적용')} className={`whitespace-normal rounded-lg border px-3 py-2 text-sm font-black ${isDarkMode ? 'border-[#334155] text-slate-200 hover:bg-[#1e293b]' : 'border-slate-200 hover:bg-slate-50'}`}>면제 적용</button>
                <button onClick={() => addNote('문의', '등기우편 수령 안내 완료')} className={`whitespace-normal rounded-lg border px-3 py-2 text-sm font-black ${isDarkMode ? 'border-[#334155] text-slate-200 hover:bg-[#1e293b]' : 'border-slate-200 hover:bg-slate-50'}`}>수령 안내</button>
              </div>
            </div>
          </div>
        </section>

        {/* 실시간 통역 + 채팅 */}
        <section className={`flex min-h-[420px] min-w-0 flex-col overflow-hidden rounded-lg border-[3px] transition-colors duration-300 xl:min-h-0 ${isDarkMode ? 'border-[#2b3a50] bg-[#121b2b]' : 'border-slate-200 bg-white'}`}>
          <div className={`flex flex-wrap items-center justify-between gap-2 border-b px-4 py-3 ${isDarkMode ? 'border-[#263244]' : 'border-slate-100'}`}>
            <h2 className={`text-base font-black ${isDarkMode ? 'text-slate-50' : ''}`}>실시간 통역</h2>
            <span className={`flex items-center gap-2 text-xs font-black ${isDarkMode ? 'text-emerald-400' : 'text-emerald-700'}`}>
              <span className="h-2 w-2 rounded-full bg-emerald-500" /> 양방향 통역 중
            </span>
          </div>

          <div className={`grid min-w-0 gap-3 p-4 sm:grid-cols-2 ${isDarkMode ? 'bg-[#0f172a]' : 'bg-slate-50'}`}>
            <div className={`min-h-[104px] min-w-0 rounded-lg border p-4 md:min-h-[124px] ${isDarkMode ? 'border-emerald-700/50 bg-[#121b2b]' : 'border-emerald-200 bg-white'}`}>
              <p className={`text-sm font-black ${isDarkMode ? 'text-emerald-400' : 'text-emerald-700'}`}>민원인 발화</p>
              <p className={`mt-3 break-words text-base font-black leading-relaxed md:text-lg ${isDarkMode ? 'text-slate-100' : 'text-slate-950'}`}>{latestCitizenText || '민원인의 수어/문자 입력을 기다리고 있습니다.'}</p>
            </div>
            <div className={`min-h-[104px] min-w-0 rounded-lg border p-4 md:min-h-[124px] ${isDarkMode ? 'border-blue-700/50 bg-[#121b2b]' : 'border-blue-200 bg-white'}`}>
              <p className={`text-sm font-black ${isDarkMode ? 'text-blue-400' : 'text-blue-700'}`}>상담원 응답</p>
              <p className={`mt-3 whitespace-pre-wrap break-words text-base font-black leading-relaxed md:text-lg ${isDarkMode ? 'text-slate-100' : 'text-slate-950'}`}>{latestAgentText}</p>
            </div>
          </div>

          <div ref={chatListRef} className={`min-h-[200px] max-h-[420px] overflow-y-auto overscroll-auto px-4 py-5 transition-colors duration-300 ${isDarkMode ? 'bg-[#121b2b]' : 'bg-white'}`}>
            {messages.length === 0 ? (
              <div className={`flex h-full items-center justify-center text-center text-sm font-bold ${isDarkMode ? 'text-slate-600' : 'text-slate-300'}`}>
                상담이 시작되면 민원인 발화와 상담원 응답이 여기에 기록됩니다.
              </div>
            ) : (
              messages.map((message) => <ChatMessage key={message.id} message={message} dark={isDarkMode} />)
            )}
          </div>
        </section>

        {/* 상담원 답변 작성 */}
        <section className={`flex min-h-[520px] min-w-0 flex-col overflow-hidden rounded-lg border-[3px] transition-colors duration-300 md:col-span-2 xl:col-span-1 xl:min-h-0 ${isDarkMode ? 'border-[#2b3a50] bg-[#121b2b]' : 'border-slate-200 bg-white'}`}>
          <div className={`shrink-0 border-b px-4 py-3 ${isDarkMode ? 'border-[#263244]' : 'border-slate-100'}`}>
            <h2 className={`text-base font-black ${isDarkMode ? 'text-slate-50' : ''}`}>상담원 답변 작성</h2>
            <p className={`mt-1 text-sm font-semibold ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>답변은 민원인 화면에서 수어/문자로 안내됩니다.</p>
          </div>
          <div className="min-h-0 flex-1 overflow-y-auto overscroll-auto">
            <div className="space-y-4 p-4">
              <div className="flex h-12 items-end gap-1">
                {voiceLevels.map((level, index) => {
                  const displayLevel = isSpeechActive ? level : 0.12
                  return (
                    <div
                      key={index}
                      className="flex-1 rounded-full transition-all"
                      style={{
                        height: `${Math.max(10, Math.round(displayLevel * 100))}%`,
                        background: isSpeechActive && level > 0.2 ? VOICE_BAR_COLORS[index % VOICE_BAR_COLORS.length] : (isDarkMode ? '#334155' : '#E2E8F0'),
                      }}
                    />
                  )
                })}
              </div>
              <textarea
                value={replyInput}
                onChange={(event) => setReplyInput(event.target.value)}
                onKeyDown={handleReplyKeyDown}
                placeholder="민원인에게 전달할 답변을 입력하세요."
                className={`h-36 w-full min-w-0 resize-none rounded-lg border p-4 text-base font-bold leading-relaxed outline-none ${isDarkMode ? 'border-[#334155] bg-[#0f172a] text-slate-100 placeholder:text-slate-500 focus:border-blue-500' : 'border-slate-200 bg-white focus:border-blue-500'}`}
              />
              <div className="grid gap-2 sm:grid-cols-2">
                <button onClick={sendReply} disabled={!replyInput.trim()} className={`whitespace-normal rounded-lg px-4 py-3 text-sm font-black text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed ${replyInput.trim() ? 'bg-blue-600' : isDarkMode ? 'bg-[#334155] text-slate-500' : 'bg-blue-200'}`}>
                  전송
                </button>
                <button onClick={isSpeechActive ? stopSpeech : startSpeech} className={`whitespace-normal break-words rounded-lg border px-4 py-3 text-sm font-black ${isSpeechActive ? (isDarkMode ? 'border-amber-700 bg-amber-900/40 text-amber-400' : 'border-amber-200 bg-amber-50 text-amber-700') : (isDarkMode ? 'border-[#334155] bg-[#0f172a] text-slate-200 hover:bg-[#1e293b]' : 'border-slate-200 bg-slate-50 text-slate-700 hover:bg-slate-100')}`}>
                  {isSpeechActive ? '음성 입력 중지' : '음성으로 말하기'}
                </button>
              </div>
            </div>

            <div className={`border-t p-4 ${isDarkMode ? 'border-[#263244]' : 'border-slate-100'}`}>
              <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
                <div className="min-w-0">
                  <h3 className={`text-sm font-black ${isDarkMode ? 'text-slate-200' : 'text-slate-800'}`}>자주쓰는 문장</h3>
                  <p className={`mt-1 text-xs font-semibold ${isDarkMode ? 'text-slate-500' : 'text-slate-500'}`}>문장을 선택하면 답변창에 입력됩니다. 상담원이 수정한 뒤 전송하세요.</p>
                </div>
                <span className={`text-xs font-black ${isDarkMode ? 'text-slate-500' : 'text-slate-400'}`}>{QUICK_REPLIES.length}개</span>
              </div>
              <div className="grid gap-2 pr-1 sm:grid-cols-2 xl:grid-cols-1">
                {QUICK_REPLIES.map((reply) => (
                  <button
                    key={reply.title}
                    onClick={() => setReplyInput(reply.text)}
                    className={`min-w-0 rounded-lg border p-3 text-left ${isDarkMode ? 'border-[#263244] bg-[#0f172a] hover:border-blue-600 hover:bg-blue-900/30' : 'border-slate-200 bg-slate-50 hover:border-blue-300 hover:bg-blue-50'}`}
                  >
                    <p className={`break-words text-sm font-black ${isDarkMode ? 'text-slate-100' : 'text-slate-900'}`}>{reply.title}</p>
                    <p className={`mt-1 break-words text-sm font-semibold leading-relaxed ${isDarkMode ? 'text-slate-400' : 'text-slate-600'}`}>{reply.text}</p>
                  </button>
                ))}
              </div>
            </div>

          </div>
        </section>
      </main>

      {showEndConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/50 p-4 backdrop-blur-sm">
          <div className={`w-full max-w-sm rounded-2xl p-6 shadow-2xl ${isDarkMode ? 'bg-[#121b2b] text-slate-50' : 'bg-white'}`}>
            <h3 className={`text-xl font-black ${isDarkMode ? 'text-slate-50' : 'text-slate-950'}`}>상담을 종료하시겠습니까?</h3>
            <p className={`mt-2 text-sm font-semibold leading-relaxed ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>작성한 민원 메모와 상담 내용이 기록에 저장됩니다.</p>
            <div className="mt-6 grid grid-cols-2 gap-2">
              <button onClick={() => setShowEndConfirm(false)} className={`rounded-lg px-4 py-3 text-sm font-black ${isDarkMode ? 'bg-[#1e293b] text-slate-300 hover:bg-[#263244]' : 'bg-slate-100 text-slate-600 hover:bg-slate-200'}`}>취소</button>
              <button onClick={handleConfirmEnd} className="rounded-lg bg-red-500 px-4 py-3 text-sm font-black text-white hover:bg-red-600">종료 및 저장</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
