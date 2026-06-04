import { useState, useEffect, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useKioskSignLanguage } from '../hooks/useKioskSignLanguage'
import { useSessionPoller } from '../hooks/useSessionPoller'
import { useMessagePoller } from '../hooks/useMessagePoller'
import { useWebRTC } from '../hooks/useWebRTC'
import { postMessage, getCitizenSession } from '../api/session'
import { socket, registerRole } from '../socket'
import ChatMessage from '../components/ChatMessage'
import type { ChatMessage as ChatMessageType } from '../types'

async function clearSessionData() {
  try {
    await fetch('/api/citizen-session', { method: 'DELETE', credentials: 'include' })
    await fetch('/api/messages',        { method: 'DELETE', credentials: 'include' })
  } catch { /* 무시 */ }
}

export default function KioskSessionPage() {
  const navigate = useNavigate()
  const chatEndRef = useRef<HTMLDivElement>(null)

  // ── 소켓 role 등록 + 민원인 도착 알림 (순서 보장) ─────
  // registerRole 후 소켓 서버에 kiosk가 등록된 다음 citizen_arrived를 보내야
  // 상담원 화면에서 WebRTC offer가 정상적으로 시작됨
  useEffect(() => {
    async function registerAndNotify() {
      // 1. kiosk role 등록
      registerRole('kiosk')

      // 2. 소켓 서버가 register를 처리할 시간을 잠깐 대기
      await new Promise(resolve => setTimeout(resolve, 300))

      // 3. 민원인 정보 가져와서 상담원에게 알림
      try {
        const res = await getCitizenSession()
        if (res.citizenData) {
          socket.emit('citizen_arrived', { citizenData: res.citizenData })
        }
      } catch { /* 무시 */ }

    }
    void registerAndNotify()
  }, [])

  // ── 세션 종료 감지 ────────────────────────────────────
  useSessionPoller({ interval: 4000, onEndedPath: '/kiosk/standby' })

  useEffect(() => {
    const handleSessionEnd = async () => {
      await clearSessionData()
      navigate('/kiosk/standby', { replace: true })
    }
    socket.on('session_end',   handleSessionEnd)
    socket.on('session_reset', handleSessionEnd)
    return () => {
      socket.off('session_end',   handleSessionEnd)
      socket.off('session_reset', handleSessionEnd)
    }
  }, [navigate])

  // ── 수어 인식 훅 ──────────────────────────────────────
  const [localMessages, setLocalMessages] = useState<ChatMessageType[]>([])

  const handleNewMessage = useCallback(async (msg: ChatMessageType) => {
    setLocalMessages((prev) => [...prev, msg])
    try { await postMessage(msg) } catch { /* 무시 */ }
    socket.emit('chat_message', msg)
  }, [])

  const {
    videoRef, canvasRef, landmarkCanvasRef,
    isRunning, currentPrediction,
    videoDevices, selectedDeviceId, setSelectedDeviceId,
    startCamera, stopCamera, getPredictionStatus,
    getActiveStream,
  } = useKioskSignLanguage(handleNewMessage)

  // ── WebRTC — 카메라 스트림 공유 (문제 3 수정) ─────────
  // kiosk가 offer를 먼저 보내는 구조 (AgentDashboard가 answer)
  const { localRef, isConnected, isConnecting, startCall } = useWebRTC({
    role: 'kiosk',
    getExternalStream: getActiveStream,
  })

  // ── 상담원 메시지 폴링 ────────────────────────────────
  const { messages: polledMessages, resetMessages } = useMessagePoller({ interval: 3000 })

  // 소켓 채팅 메시지 수신
  const [socketMessages, setSocketMessages] = useState<ChatMessageType[]>([])
  useEffect(() => {
    const handleChatMessage = (msg: ChatMessageType) => {
      setSocketMessages((prev) => [...prev, { ...msg, timestamp: new Date(msg.timestamp) }])
    }
    socket.on('chat_message', handleChatMessage)
    return () => { socket.off('chat_message', handleChatMessage) }
  }, [])

  // ── 메시지 합산 & 중복 제거 ──────────────────────────
  const allMessages = [...localMessages, ...polledMessages, ...socketMessages].sort(
    (a, b) => a.timestamp.getTime() - b.timestamp.getTime(),
  )
  const seen = new Set<string>()
  const dedupedMessages = allMessages.filter((m) => {
    if (seen.has(m.id)) return false
    seen.add(m.id)
    return true
  })

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [dedupedMessages.length])

  // 카메라 자동 시작
  useEffect(() => {
    void startCamera()
    return () => stopCamera()
  }, [startCamera, stopCamera])

  // 카메라가 켜진 후 WebRTC offer 전송
  useEffect(() => {
    if (!isRunning) return
    const timer = setTimeout(() => { void startCall() }, 500)
    return () => clearTimeout(timer)
  }, [isRunning, startCall])

  // ── 세션 종료 ────────────────────────────────────────
  const handleEndSession = useCallback(async () => {
    stopCamera()
    resetMessages()
    await clearSessionData()
    socket.emit('session_end')
    navigate('/kiosk/standby', { replace: true })
  }, [stopCamera, resetMessages, navigate])

  const predStatus = getPredictionStatus(currentPrediction)
  const hasHand = currentPrediction?.has_hand ?? false

  return (
    <div className="min-h-screen bg-gray-950 flex flex-col">

      {/* ── 헤더 ── */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-gray-800 bg-gray-900/80">
        <div className="flex items-center gap-3">
          <div className={`w-3 h-3 rounded-full ${isRunning ? 'bg-teal-400 animate-pulse' : 'bg-gray-600'}`} />
          <span className="text-white font-semibold text-lg">수어 상담</span>
          <span className={`text-xs px-2 py-0.5 rounded-full border ${
            isConnected
              ? 'bg-green-500/10 border-green-500/30 text-green-400'
              : isConnecting
                ? 'bg-yellow-500/10 border-yellow-500/30 text-yellow-400'
                : 'bg-gray-700/50 border-gray-600 text-gray-400'
          }`}>
            {isConnected ? '상담원 연결됨' : isConnecting ? '연결 중...' : '상담원 대기'}
          </span>
        </div>

        {videoDevices.length > 1 && (
          <select
            value={selectedDeviceId}
            onChange={(e) => setSelectedDeviceId(e.target.value)}
            className="text-sm bg-gray-800 border border-gray-600 text-gray-300 rounded-lg px-3 py-2"
          >
            {videoDevices.map((d) => (
              <option key={d.deviceId} value={d.deviceId}>
                {d.label || `카메라 ${d.deviceId.slice(0, 6)}`}
              </option>
            ))}
          </select>
        )}

        <button
          onClick={handleEndSession}
          className="flex items-center gap-2 px-4 py-2 rounded-xl
                     bg-gray-800 hover:bg-gray-700 border border-gray-600
                     text-gray-300 text-sm font-medium transition-colors"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
            <polyline points="16 17 21 12 16 7"/>
            <line x1="21" y1="12" x2="9" y2="12"/>
          </svg>
          상담 종료
        </button>
      </header>

      {/* ── 메인 ── */}
      <main className="flex-1 flex overflow-hidden">

        <section className="relative flex-1 bg-black flex items-center justify-center">
          {/* WebRTC용 숨김 video (카메라 스트림 공유) */}
          <video ref={localRef} autoPlay playsInline muted className="hidden" />

          {/* 수어 인식용 카메라 */}
          <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" style={{ transform: 'scaleX(-1)' }} />
          <canvas ref={canvasRef} className="hidden" />

          {/* 랜드마크 오버레이 (문제 4 수정) */}
          <canvas
            ref={landmarkCanvasRef}
            className="absolute inset-0 w-full h-full pointer-events-none"
          />

          <div className="absolute bottom-6 left-6 right-6 flex items-center justify-between">
            <div className={`
              flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium backdrop-blur-sm
              ${hasHand
                ? 'bg-teal-500/20 border border-teal-500/40 text-teal-300'
                : 'bg-gray-800/60 border border-gray-600/40 text-gray-400'}
            `}>
              <div className={`w-2 h-2 rounded-full ${hasHand ? 'bg-teal-400 animate-pulse' : 'bg-gray-500'}`} />
              {predStatus}
            </div>

            <button
              onClick={isRunning ? stopCamera : startCamera}
              className={`
                px-4 py-2 rounded-full text-sm font-medium backdrop-blur-sm transition-colors
                ${isRunning
                  ? 'bg-red-500/20 border border-red-500/40 text-red-300 hover:bg-red-500/30'
                  : 'bg-teal-500/20 border border-teal-500/40 text-teal-300 hover:bg-teal-500/30'}
              `}
            >
              {isRunning ? '카메라 중지' : '카메라 시작'}
            </button>
          </div>
        </section>

        <section className="w-96 flex flex-col bg-gray-900 border-l border-gray-800">
          <div className="px-5 py-4 border-b border-gray-800">
            <h2 className="text-white font-semibold">대화</h2>
            <p className="text-gray-400 text-xs mt-0.5">수어 인식 결과와 상담원 메시지</p>
          </div>

          <div className="flex-1 overflow-y-auto px-4 py-4 space-y-4">
            {dedupedMessages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full gap-3 text-center">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#4b5563" strokeWidth="1.5">
                  <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
                </svg>
                <p className="text-gray-500 text-sm">
                  수어로 말씀해 주세요.<br />인식된 내용이 여기에 표시됩니다.
                </p>
              </div>
            ) : (
              dedupedMessages.map((msg) => (
                <ChatMessage key={msg.id} message={msg} dark />
              ))
            )}
            <div ref={chatEndRef} />
          </div>
        </section>
      </main>

      <footer className="px-6 py-3 bg-gray-900/60 border-t border-gray-800 flex items-center justify-center">
        <p className="text-gray-500 text-sm text-center">
          카메라 앞에서 수어를 표현하면 자동으로 인식됩니다
        </p>
      </footer>
    </div>
  )
}
