import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { KeyboardEvent } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import type { ChatMessage as ChatMessageType, AgentNote } from '../types'
import ChatMessage from '../components/ChatMessage'
import { useSpeechRecognition } from '../hooks/useSpeechRecognition'
import { registerRole, socket } from '../socket'

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

const SCENARIO_STEPS = [
  { code: 'SEN0354', sign: '안녕하세요', text: '안녕하세요' },
  { code: 'WORD0579 + SEN0322', sign: '복지카드 + 잃어버리다', text: '복지카드를 잃어버렸어요' },
  { code: 'WORD1174 + WORD1282', sign: '맞다 + 가능', text: '재발급 가능한가요?' },
  { code: 'SEN0169 + SEN0175', sign: '신분증 + 여기 있다', text: '신분증 여기에 있어요' },
  { code: 'SEN0278', sign: '지하철 안 없다', text: '지하철에서 잃어버렸어요' },
  { code: 'WORD0602 + SEN0109', sign: '면제 + 가능', text: '면제 받을 수 있나요?' },
  { code: 'SEN0355', sign: '감사합니다', text: '감사합니다' },
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
  const chatEndRef = useRef<HTMLDivElement>(null)
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

  const [videoConnected, setVideoConnected] = useState(false)
  const [notes, setNotes] = useState<AgentNote[]>([])
  const [noteInput, setNoteInput] = useState('')
  const [noteTag, setNoteTag] = useState<AgentNote['tag']>('문의')
  const [replyInput, setReplyInput] = useState('')
  const [showEndConfirm, setShowEndConfirm] = useState(false)
  const [sessionDone, setSessionDone] = useState(false)
  const [taskType, setTaskType] = useState('복지카드 재발급')
  const [understood, setUnderstood] = useState(false)

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
    socket.emit('agent_voice_status', { active: isSpeechActive })
  }, [isSpeechActive])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

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
    setUnderstood(false)
    onSessionReset()
  }

  return (
    <div className="flex h-screen w-full flex-col overflow-hidden bg-slate-100 text-slate-900">
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/95 backdrop-blur">
        <div className="flex flex-col gap-3 px-4 py-4 md:flex-row md:items-center md:justify-between md:px-6">
          <div className="min-w-0">
            <p className="break-words text-xs font-black uppercase tracking-[0.18em] text-blue-600 sm:tracking-[0.24em]">Civil Sign Interpreter</p>
            <h1 className="mt-1 break-words text-xl font-black tracking-tight text-slate-950 md:text-2xl">수어 통역 상담원 화면</h1>
            <p className="break-words text-sm font-semibold text-slate-500">{citizenName} 민원인 상담 중</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className={`rounded-full border px-3 py-1 text-xs font-black ${sessionDone ? 'border-slate-200 bg-slate-50 text-slate-500' : 'border-emerald-200 bg-emerald-50 text-emerald-700'}`}>
              {sessionDone ? '상담 종료됨' : '양방향 통역 중'}
            </span>
            {sessionDone ? (
              <button onClick={handleNewSession} className="rounded-lg border border-blue-200 bg-blue-50 px-4 py-2 text-sm font-black text-blue-700 hover:bg-blue-100">
                새 상담 시작
              </button>
            ) : (
              <button onClick={() => setShowEndConfirm(true)} className="rounded-lg border border-red-200 bg-red-50 px-4 py-2 text-sm font-black text-red-600 hover:bg-red-100">
                상담 끝내기
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="grid min-h-0 min-w-0 flex-1 grid-cols-1 gap-3 overflow-hidden p-3 md:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)] xl:grid-cols-[minmax(0,0.8fr)_minmax(0,1.15fr)_minmax(0,0.9fr)]">
        <section className="min-w-0 overflow-hidden rounded-lg border border-slate-200 bg-white">
          <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
            <h2 className="text-base font-black">민원인 수어 영상</h2>
            <span className={`rounded-full border px-2.5 py-1 text-xs font-black ${videoConnected ? 'border-emerald-200 bg-emerald-50 text-emerald-700' : 'border-slate-200 bg-slate-50 text-slate-500'}`}>
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
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-black text-slate-700">민원 메모</h3>
              <span className="text-xs font-bold text-slate-400">{notes.length}개</span>
            </div>
            <div className="grid grid-cols-[minmax(0,1fr)_minmax(78px,92px)_40px] gap-2">
              <input
                value={noteInput}
                onChange={(event) => setNoteInput(event.target.value)}
                onKeyDown={handleNoteKeyDown}
                placeholder="메모 입력..."
                className="h-10 min-w-0 rounded-lg border border-slate-200 px-3 text-sm font-semibold outline-none focus:border-blue-400"
              />
              <select value={noteTag} onChange={(event) => setNoteTag(event.target.value as AgentNote['tag'])} className="h-10 min-w-0 rounded-lg border border-slate-200 bg-white px-2 text-sm font-bold outline-none">
                <option value="문의">문의</option>
                <option value="확인">확인</option>
                <option value="처리">처리</option>
              </select>
              <button onClick={() => addNote()} className="h-10 rounded-lg bg-blue-600 text-lg font-black text-white hover:bg-blue-700">+</button>
            </div>
            <div className="max-h-[240px] space-y-2 overflow-y-auto pr-1">
              {notes.map((note) => (
                <article key={note.id} className="rounded-lg border border-slate-100 bg-slate-50 p-3">
                  <div className="flex items-start justify-between gap-2">
                    <p className="min-w-0 break-words text-sm font-bold leading-relaxed text-slate-800">{note.text}</p>
                    <span className={`shrink-0 rounded-full border px-2 py-0.5 text-[11px] font-black ${TAG_STYLES[note.tag]}`}>{note.tag}</span>
                  </div>
                  <span className="mt-2 block text-[11px] font-semibold text-slate-400">{timeLabel(note.timestamp)}</span>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="flex min-h-0 min-w-0 flex-col overflow-hidden rounded-lg border border-slate-200 bg-white">
          <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
            <h2 className="text-base font-black">실시간 통역</h2>
            <span className="flex items-center gap-2 text-xs font-black text-emerald-700">
              <span className="h-2 w-2 rounded-full bg-emerald-500" /> 양방향 통역 중
            </span>
          </div>

          <div className="grid min-w-0 gap-3 bg-slate-50 p-4 sm:grid-cols-2">
            <div className="min-h-[104px] min-w-0 rounded-lg border border-emerald-200 bg-white p-4 md:min-h-[124px]">
              <p className="text-sm font-black text-emerald-700">민원인 발화</p>
              <p className="mt-3 break-words text-base font-black leading-relaxed text-slate-950 md:text-lg">{latestCitizenText || '민원인의 수어/문자 입력을 기다리고 있습니다.'}</p>
            </div>
            <div className="min-h-[104px] min-w-0 rounded-lg border border-blue-200 bg-white p-4 md:min-h-[124px]">
              <p className="text-sm font-black text-blue-700">상담원 응답</p>
              <p className="mt-3 whitespace-pre-wrap break-words text-base font-black leading-relaxed text-slate-950 md:text-lg">{latestAgentText}</p>
            </div>
          </div>

          <div className="min-h-0 flex-1 overflow-y-scroll overscroll-contain px-4 py-5">
            {messages.length === 0 ? (
              <div className="flex h-full items-center justify-center text-center text-sm font-bold text-slate-300">
                상담이 시작되면 민원인 발화와 상담원 응답이 여기에 기록됩니다.
              </div>
            ) : (
              messages.map((message) => <ChatMessage key={message.id} message={message} />)
            )}
            <div ref={chatEndRef} />
          </div>
        </section>

        <section className="flex min-w-0 flex-col overflow-hidden rounded-lg border border-slate-200 bg-white md:col-span-2 xl:col-span-1">
          <div className="border-b border-slate-100 px-4 py-3">
            <h2 className="text-base font-black">상담원 답변 작성</h2>
            <p className="mt-1 text-sm font-semibold text-slate-500">답변은 민원인 화면에서 수어/문자로 안내됩니다.</p>
          </div>
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
                      background: isSpeechActive && level > 0.2 ? VOICE_BAR_COLORS[index % VOICE_BAR_COLORS.length] : '#E2E8F0',
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
              className="h-36 w-full min-w-0 resize-none rounded-lg border border-slate-200 p-4 text-base font-bold leading-relaxed outline-none focus:border-blue-500"
            />
            <div className="grid gap-2 sm:grid-cols-2">
              <button onClick={sendReply} disabled={!replyInput.trim()} className="whitespace-normal rounded-lg bg-blue-600 px-4 py-3 text-sm font-black text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-blue-200">
                전송
              </button>
              <button onClick={isSpeechActive ? stopSpeech : startSpeech} className={`whitespace-normal break-words rounded-lg border px-4 py-3 text-sm font-black ${isSpeechActive ? 'border-amber-200 bg-amber-50 text-amber-700' : 'border-slate-200 bg-slate-50 text-slate-700 hover:bg-slate-100'}`}>
                {isSpeechActive ? '음성 입력 중지' : '음성으로 말하기'}
              </button>
            </div>
          </div>

          <div className="border-t border-slate-100 p-4">
            <div className="mb-3 flex items-end justify-between">
              <div className="min-w-0">
                <h3 className="text-sm font-black text-slate-800">자주쓰는 문장</h3>
                <p className="mt-1 text-xs font-semibold text-slate-500">문장을 선택하면 답변창에 입력됩니다. 상담원이 수정한 뒤 전송하세요.</p>
              </div>
              <span className="text-xs font-black text-slate-400">{QUICK_REPLIES.length}개</span>
            </div>
            <div className="grid max-h-[260px] gap-2 overflow-y-auto pr-1 sm:grid-cols-2 xl:grid-cols-1">
              {QUICK_REPLIES.map((reply) => (
                <button
                  key={reply.title}
                  onClick={() => setReplyInput(reply.text)}
                  className="min-w-0 rounded-lg border border-slate-200 bg-slate-50 p-3 text-left hover:border-blue-300 hover:bg-blue-50"
                >
                  <p className="break-words text-sm font-black text-slate-900">{reply.title}</p>
                  <p className="mt-1 break-words text-sm font-semibold leading-relaxed text-slate-600">{reply.text}</p>
                </button>
              ))}
            </div>
          </div>

          <div className="mt-auto border-t border-slate-100 p-4">
            <div className="grid gap-2">
              <label className="text-sm font-black text-slate-700">업무 분류</label>
              <select value={taskType} onChange={(event) => setTaskType(event.target.value)} className="h-11 min-w-0 rounded-lg border border-slate-200 bg-white px-3 text-sm font-black outline-none">
                <option>복지카드 재발급</option>
                <option>분실 접수</option>
                <option>본인 확인</option>
                <option>수수료 면제</option>
              </select>
              <div className="grid gap-2 sm:grid-cols-4 xl:grid-cols-2">
                <button onClick={() => addNote('처리', `${taskType} 분실 접수 진행`)} className="whitespace-normal rounded-lg border border-slate-200 px-3 py-2 text-sm font-black hover:bg-slate-50">분실 접수</button>
                <button onClick={() => addNote('확인', '신분증 본인 확인 완료')} className="whitespace-normal rounded-lg border border-slate-200 px-3 py-2 text-sm font-black hover:bg-slate-50">본인 확인</button>
                <button onClick={() => addNote('처리', '재발급 수수료 면제 적용')} className="whitespace-normal rounded-lg border border-slate-200 px-3 py-2 text-sm font-black hover:bg-slate-50">면제 적용</button>
                <button onClick={() => addNote('문의', '등기우편 수령 안내 완료')} className="whitespace-normal rounded-lg border border-slate-200 px-3 py-2 text-sm font-black hover:bg-slate-50">수령 안내</button>
              </div>
              <label className="flex items-center justify-between gap-3 rounded-lg border border-emerald-200 bg-emerald-50 px-3 py-2 text-sm font-black text-emerald-800">
                <span className="min-w-0 break-words">민원인 이해 확인</span>
                <input type="checkbox" checked={understood} onChange={(event) => setUnderstood(event.target.checked)} className="h-5 w-5 accent-emerald-600" />
              </label>
            </div>
          </div>
        </section>

        <section className="min-w-0 rounded-lg border border-slate-200 bg-white p-4 md:col-span-2 xl:col-span-3">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-base font-black">시나리오 초안</h2>
            <span className="text-xs font-black text-slate-400">{SCENARIO_STEPS.length}단계</span>
          </div>
          <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-4">
            {SCENARIO_STEPS.map((step) => (
              <div key={step.code} className="rounded-lg border border-slate-100 bg-slate-50 p-3">
                <p className="text-xs font-black text-blue-600">{step.code}</p>
                <p className="mt-1 break-words text-sm font-black text-slate-900">{step.sign}</p>
                <p className="mt-1 break-words text-sm font-semibold text-slate-500">{step.text}</p>
              </div>
            ))}
          </div>
        </section>
      </main>

      {showEndConfirm && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/50 p-4 backdrop-blur-sm">
          <div className="w-full max-w-sm rounded-2xl bg-white p-6 shadow-2xl">
            <h3 className="text-xl font-black text-slate-950">상담을 종료하시겠습니까?</h3>
            <p className="mt-2 text-sm font-semibold leading-relaxed text-slate-500">작성한 민원 메모와 상담 내용이 기록에 저장됩니다.</p>
            <div className="mt-6 grid grid-cols-2 gap-2">
              <button onClick={() => setShowEndConfirm(false)} className="rounded-lg bg-slate-100 px-4 py-3 text-sm font-black text-slate-600 hover:bg-slate-200">취소</button>
              <button onClick={handleConfirmEnd} className="rounded-lg bg-red-500 px-4 py-3 text-sm font-black text-white hover:bg-red-600">종료 및 저장</button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
