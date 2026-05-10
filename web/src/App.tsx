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
        const parsed = JSON.parse(stored) as Array<Omit<ChatMessage, 'timestamp'> & { timestamp: string }>
        const restored = parsed.map((message) => ({ ...message, timestamp: new Date(message.timestamp) }))
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
        const incoming = payload as Omit<ChatMessage, 'timestamp'> & { timestamp: string }
        if (seenIds.current.has(incoming.id)) return
        seenIds.current.add(incoming.id)
        setMessages((prev) => {
          const next = [...prev, { ...incoming, timestamp: new Date(incoming.timestamp) }]
          localStorage.setItem(STORAGE_KEY, JSON.stringify(next.map((msg) => ({ ...msg, timestamp: msg.timestamp.toISOString() }))))
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
    if (seenIds.current.has(message.id)) return
    seenIds.current.add(message.id)

    setMessages((prev) => {
      const next = [...prev, message]
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next.map((msg) => ({ ...msg, timestamp: msg.timestamp.toISOString() }))))
      return next
    })
    channelRef.current?.postMessage({ type: 'new_message', payload: { ...message, timestamp: message.timestamp.toISOString() } })

    fetch('/api/messages', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...message, timestamp: message.timestamp.toISOString() }),
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
