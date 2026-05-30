import { useCallback, useEffect, useRef, useState } from 'react'
import type { ReactElement } from 'react'
import { Navigate, Route, Routes, useLocation } from 'react-router-dom'

// ─── 전역 프론트엔드 에러 리포터 ─────────────────────────────────────────────
/**
 * 앱 전체에서 발생하는 JS 런타임 에러와 unhandled promise rejection을
 * 백엔드 로그 저장소(/api/logs/frontend)로 전송합니다.
 * 로그 뷰가 열려 있지 않아도 항상 동작합니다.
 */
function FrontendErrorReporter() {
  useEffect(() => {
    const post = (level: string, source: string, message: string) => {
      // 로그 전송 실패는 조용히 무시 (무한루프 방지)
      fetch('/api/logs/frontend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ level, source, message }),
      }).catch(() => {})
    }

    const handleError = (event: ErrorEvent) => {
      const file = event.filename?.split('/').pop() ?? ''
      post('error', 'Frontend Error', `${event.message}${file ? ` (${file}:${event.lineno})` : ''}`)
    }

    const handleRejection = (event: PromiseRejectionEvent) => {
      post('error', 'Frontend Promise', `Unhandled rejection: ${String(event.reason)}`)
    }

    window.addEventListener('error', handleError)
    window.addEventListener('unhandledrejection', handleRejection)

    return () => {
      window.removeEventListener('error', handleError)
      window.removeEventListener('unhandledrejection', handleRejection)
    }
  }, [])

  return null
}
import type { ChatMessage } from './types'
import LoginPage from './pages/LoginPage'
import AdminHome from './pages/AdminHome'
import AgentLaunchScreen from './pages/AgentLaunchScreen'
import AgentDashboard from './pages/AgentDashboard'
import KakaoCallback from './pages/KakaoCallback'
import KioskLaunchScreen from './pages/KioskLaunchScreen'
import CitizenKiosk from './pages/CitizenKiosk'
import { MEDIAPIPE_MODE_STORAGE_KEY, SCENARIO_MODE_STORAGE_KEY } from './hooks/useSignLanguage'

const CHANNEL_NAME = 'sign-lang-chat'
const STORAGE_KEY = 'sign-lang-messages'
const SESSION_KEY = 'sign-lang-session'
const AUTH_KEY = 'ksl-admin-authenticated'

type IncomingChatMessage = Omit<ChatMessage, 'timestamp'> & {
  timestamp: Date | string | number
}

function RequireAuth({ isAuthenticated, children }: { isAuthenticated: boolean; children: ReactElement }) {
  const location = useLocation()

  if (!isAuthenticated) {
    return <Navigate to="/" replace state={{ from: `${location.pathname}${location.search}${location.hash}` }} />
  }

  return children
}

const normalizeTimestamp = (timestamp: Date | string | number | null | undefined) => {
  const date = timestamp instanceof Date ? timestamp : new Date(timestamp ?? Date.now())
  return Number.isNaN(date.getTime()) ? new Date() : date
}

const normalizeMessage = (message: IncomingChatMessage): ChatMessage => ({
  ...message,
  timestamp: normalizeTimestamp(message.timestamp),
})

const serializeMessage = (message: ChatMessage) => ({
  ...message,
  timestamp: normalizeTimestamp(message.timestamp).toISOString(),
})

const serializeMessages = (items: ChatMessage[]) => JSON.stringify(items.map(serializeMessage))

export default function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [sessionEnded, setSessionEnded] = useState(false)
  const [isAuthenticated, setIsAuthenticated] = useState(() => localStorage.getItem(AUTH_KEY) === 'true')
  const seenIds = useRef<Set<string>>(new Set())
  const channelRef = useRef<BroadcastChannel | null>(null)

  useEffect(() => {
    const mode = new URLSearchParams(window.location.search).get('mp')
    if (mode === 'client' || mode === 'server') {
      localStorage.setItem(MEDIAPIPE_MODE_STORAGE_KEY, mode)
    }
    const scenario = new URLSearchParams(window.location.search).get('scenario')
    if (scenario) {
      const enabled = ['1', 'true', 'yes', 'resident'].includes(scenario.toLowerCase())
      localStorage.setItem(SCENARIO_MODE_STORAGE_KEY, enabled ? 'resident' : 'off')
    }
  }, [])

  // 1. 초기 데이터 로드
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) {
        const parsed = JSON.parse(stored) as IncomingChatMessage[]
        const restored = parsed.map(normalizeMessage)
        restored.forEach((message) => seenIds.current.add(message.id))
        setMessages(restored)
      }
      setSessionEnded(localStorage.getItem(SESSION_KEY) === 'ended')
    } catch {
      localStorage.removeItem(STORAGE_KEY)
    }
  }, [])

  // 💡 2. 디바운스(Debounce)를 활용한 스토리지 저장 최적화
  useEffect(() => {
    if (messages.length === 0) return // 빈 배열일 때는 불필요한 저장 생략

    const timerId = setTimeout(() => {
      localStorage.setItem(STORAGE_KEY, serializeMessages(messages))
    }, 500) // 500ms 동안 새 메시지가 없으면 최종본 1회 저장

    // 500ms가 지나기 전에 새 메시지가 오면 기존 타이머 취소 (금고 넣기 보류)
    return () => clearTimeout(timerId)
  }, [messages])

  // 3. 브로드캐스트 채널 설정
  useEffect(() => {
    const channel = new BroadcastChannel(CHANNEL_NAME)
    channelRef.current = channel

    channel.onmessage = (event) => {
      const { type, payload } = event.data ?? {}
      if (type === 'new_message') {
        const incoming = normalizeMessage(payload as IncomingChatMessage)
        if (seenIds.current.has(incoming.id)) return
        seenIds.current.add(incoming.id)
        
        // 💡 동기식 localStorage.setItem 제거 (상태만 업데이트)
        setMessages((prev) => [...prev, incoming])
      }
      if (type === 'session_end') {
        setSessionEnded(true)
        localStorage.setItem(SESSION_KEY, 'ended')
      }
      if (type === 'session_reset') {
        seenIds.current.clear()
        setMessages([])
        setSessionEnded(false)
        localStorage.removeItem(STORAGE_KEY)
        localStorage.removeItem(SESSION_KEY)
      }
    }

    return () => channel.close()
  }, [])

  const handleLogin = () => {
    localStorage.setItem(AUTH_KEY, 'true')
    setIsAuthenticated(true)
  }

  const handleLogout = async () => {
    try {
      await fetch('/api/logout', { method: 'POST', credentials: 'include' })
    } catch {}
    localStorage.removeItem(AUTH_KEY)
    setIsAuthenticated(false)
  }

  const handleNewMessage = useCallback((message: ChatMessage) => {
    const normalized = normalizeMessage(message as IncomingChatMessage)
    if (seenIds.current.has(normalized.id)) return
    seenIds.current.add(normalized.id)

    // 💡 동기식 localStorage.setItem 제거
    setMessages((prev) => [...prev, normalized])
    
    channelRef.current?.postMessage({ type: 'new_message', payload: serializeMessage(normalized) })

    fetch('/api/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(serializeMessage(normalized)),
    }).catch(() => {})
  }, [])

  const handleSessionEnd = useCallback(() => {
    setSessionEnded(true)
    localStorage.setItem(SESSION_KEY, 'ended')
    channelRef.current?.postMessage({ type: 'session_end', payload: null })
  }, [])

  const handleSessionReset = useCallback(() => {
    seenIds.current.clear()
    setMessages([])
    setSessionEnded(false)
    localStorage.removeItem(STORAGE_KEY)
    localStorage.removeItem(SESSION_KEY)
    channelRef.current?.postMessage({ type: 'session_reset', payload: null })
  }, [])

  return (
    <>
    <FrontendErrorReporter />
    <Routes>
      <Route path="/" element={<LoginPage onLogin={handleLogin} />} />
      <Route
        path="/admin"
        element={
          <RequireAuth isAuthenticated={isAuthenticated}>
            <AdminHome onLogout={handleLogout} onSessionReset={handleSessionReset} />
          </RequireAuth>
        }
      />
      <Route
        path="/agent/launch"
        element={
          <RequireAuth isAuthenticated={isAuthenticated}>
            <AgentLaunchScreen onSessionReset={handleSessionReset} />
          </RequireAuth>
        }
      />
      <Route
        path="/agent"
        element={
          <RequireAuth isAuthenticated={isAuthenticated}>
            <AgentDashboard
              messages={messages}
              onNewMessage={handleNewMessage}
              onSessionEnd={handleSessionEnd}
              onSessionReset={handleSessionReset}
            />
          </RequireAuth>
        }
      />
      <Route
        path="/kiosk"
        element={
          <RequireAuth isAuthenticated={isAuthenticated}>
            <KioskLaunchScreen />
          </RequireAuth>
        }
      />
      <Route
        path="/kiosk/session"
        element={
          <RequireAuth isAuthenticated={isAuthenticated}>
            <CitizenKiosk
              messages={messages}
              onNewMessage={handleNewMessage}
              onSessionReset={handleSessionReset}
              sessionEnded={sessionEnded}
            />
          </RequireAuth>
        }
      />
      <Route path="/citizen" element={<Navigate to="/kiosk" replace />} />
      <Route path="/citizen/session" element={<Navigate to="/kiosk/session" replace />} />
      <Route path="/kakao/callback" element={<KakaoCallback />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
    </>
  )
}
