import { useCallback, useEffect, useRef, useState } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import type { ChatMessage } from './types'
import LoginPage from './pages/LoginPage'
import AdminHome from './pages/AdminHome'
import DoctorLaunchScreen from './pages/DoctorLaunchScreen'
import DoctorDashboard from './pages/DoctorDashboard'
import KakaoCallback from './pages/KakaoCallback'
import KioskLaunchScreen from './pages/KioskLaunchScreen'
import PatientKiosk from './pages/PatientKiosk'

const CHANNEL_NAME = 'sign-lang-chat'
const STORAGE_KEY = 'sign-lang-messages'
const SESSION_KEY = 'sign-lang-session'
const AUTH_KEY = 'ksl-admin-authenticated'

type IncomingChatMessage = Omit<ChatMessage, 'timestamp'> & {
  timestamp: Date | string | number
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

  useEffect(() => {
    const channel = new BroadcastChannel(CHANNEL_NAME)
    channelRef.current = channel

    channel.onmessage = (event) => {
      const { type, payload } = event.data ?? {}
      if (type === 'new_message') {
        const incoming = normalizeMessage(payload as IncomingChatMessage)
        if (seenIds.current.has(incoming.id)) return
        seenIds.current.add(incoming.id)
        setMessages((prev) => {
          const next = [...prev, incoming]
          localStorage.setItem(STORAGE_KEY, serializeMessages(next))
          return next
        })
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

  useEffect(() => {
    let cancelled = false

    const pollServerState = async () => {
      try {
        const [messagesRes, sessionRes] = await Promise.all([
          fetch('/api/messages', { credentials: 'include' }),
          fetch('/api/session-state', { credentials: 'include' }),
        ])

        if (messagesRes.ok) {
          const data = await messagesRes.json().catch(() => ({}))
          const incomingMessages = Array.isArray(data.messages) ? data.messages : []
          const unseen = incomingMessages
            .map((item: IncomingChatMessage) => normalizeMessage(item))
            .filter((item: ChatMessage) => !seenIds.current.has(item.id))

          if (!cancelled && unseen.length > 0) {
            unseen.forEach((item: ChatMessage) => seenIds.current.add(item.id))
            setMessages((prev) => {
              const next = [...prev, ...unseen]
              localStorage.setItem(STORAGE_KEY, serializeMessages(next))
              return next
            })
          }
        }

        if (sessionRes.ok) {
          const data = await sessionRes.json().catch(() => ({}))
          if (!cancelled && data.ended === true) {
            setSessionEnded(true)
            localStorage.setItem(SESSION_KEY, 'ended')
          }
        }
      } catch {}
    }

    void pollServerState()
    const id = setInterval(pollServerState, 1000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
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

    setMessages((prev) => {
      const next = [...prev, normalized]
      localStorage.setItem(STORAGE_KEY, serializeMessages(next))
      return next
    })
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
    fetch('/api/session-state', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body: JSON.stringify({ ended: true }),
    }).catch(() => {})
  }, [])

  const handleSessionReset = useCallback(() => {
    seenIds.current.clear()
    setMessages([])
    setSessionEnded(false)
    localStorage.removeItem(STORAGE_KEY)
    localStorage.removeItem(SESSION_KEY)
    channelRef.current?.postMessage({ type: 'session_reset', payload: null })
    fetch('/api/messages', { method: 'DELETE', credentials: 'include' }).catch(() => {})
    fetch('/api/session-state', { method: 'DELETE', credentials: 'include' }).catch(() => {})
  }, [])

  return (
    <Routes>
      <Route path="/" element={<LoginPage onLogin={handleLogin} />} />
      <Route
        path="/admin"
        element={
          isAuthenticated ? (
            <AdminHome onLogout={handleLogout} onSessionReset={handleSessionReset} />
          ) : (
            <Navigate to="/" replace />
          )
        }
      />
      <Route path="/doctor/launch" element={<DoctorLaunchScreen onSessionReset={handleSessionReset} />} />
      <Route
        path="/doctor"
        element={
          <DoctorDashboard
            messages={messages}
            onNewMessage={handleNewMessage}
            onSessionEnd={handleSessionEnd}
            onSessionReset={handleSessionReset}
          />
        }
      />
      <Route path="/kiosk" element={<KioskLaunchScreen />} />
      <Route
        path="/kiosk/session"
        element={
          <PatientKiosk
            messages={messages}
            onNewMessage={handleNewMessage}
            onSessionReset={handleSessionReset}
            sessionEnded={sessionEnded}
          />
        }
      />
      <Route path="/patient" element={<Navigate to="/kiosk" replace />} />
      <Route path="/patient/session" element={<Navigate to="/kiosk/session" replace />} />
      <Route path="/kakao/callback" element={<KakaoCallback />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
