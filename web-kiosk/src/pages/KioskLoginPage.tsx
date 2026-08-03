import { useState, useCallback, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { login, logout } from '../api/auth'
import { getSessionState, getCitizenSession } from '../api/session'
import { ApiError } from '../api/client'

export default function KioskLoginPage() {
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const [checking, setChecking] = useState(true)

  useEffect(() => {
    let cancelled = false

    async function checkSession() {
      try {
        // 1. 세션 유효한지 확인 (401이면 catch로)
        await getSessionState()

        // 2. 세션 유효 → 백엔드 실제 응답 { waiting, citizenData } 확인
        const res = await getCitizenSession()

        if (cancelled) return

        if (res.waiting) {
          // 상담 진행 중 → 대기화면으로
          navigate('/kiosk/standby', { replace: true })
        } else {
          // 상담원 대기 중 → 로그아웃 후 로그인 폼
          try { await logout() } catch { /* 무시 */ }
          if (!cancelled) setChecking(false)
        }
      } catch {
        // 401 또는 네트워크 오류 → 로그인 폼 표시
        if (!cancelled) setChecking(false)
      }
    }

    void checkSession()
    return () => { cancelled = true }
  }, [navigate])

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault()
    if (!username.trim() || !password) return

    setError('')
    setLoading(true)
    try {
      await login(username.trim(), password)
      navigate('/kiosk/standby', { replace: true })
    } catch (err) {
      if (err instanceof ApiError) {
        setError(err.message || '아이디 또는 비밀번호가 올바르지 않습니다.')
      } else {
        setError('서버에 연결할 수 없습니다. 잠시 후 다시 시도해 주세요.')
      }
    } finally {
      setLoading(false)
    }
  }, [username, password, navigate])

  if (checking) {
    return (
      <div className="min-h-screen bg-gray-950 flex items-center justify-center">
        <svg className="animate-spin w-8 h-8 text-teal-500" viewBox="0 0 24 24" fill="none">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
        </svg>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center p-8">
      <div className="w-full max-w-md">
        <div className="text-center mb-12">
          <div className="inline-flex items-center justify-center w-20 h-20 rounded-2xl bg-teal-500/10 border border-teal-500/30 mb-6">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#14b8a6" strokeWidth="1.5">
              <path d="M18 8h1a4 4 0 0 1 0 8h-1"/>
              <path d="M2 8h16v9a4 4 0 0 1-4 4H6a4 4 0 0 1-4-4V8z"/>
              <line x1="6" y1="1" x2="6" y2="4"/>
              <line x1="10" y1="1" x2="10" y2="4"/>
              <line x1="14" y1="1" x2="14" y2="4"/>
            </svg>
          </div>
          <h1 className="text-3xl font-bold text-white mb-2">수어 상담 키오스크</h1>
          <p className="text-gray-400">관리자 로그인 후 서비스를 시작합니다</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-5">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">아이디</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="관리자 아이디"
              autoComplete="username"
              disabled={loading}
              className="w-full px-4 py-4 text-lg bg-gray-800 border border-gray-600 rounded-xl text-white placeholder-gray-500
                         focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent
                         disabled:opacity-50"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">비밀번호</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="비밀번호"
              autoComplete="current-password"
              disabled={loading}
              className="w-full px-4 py-4 text-lg bg-gray-800 border border-gray-600 rounded-xl text-white placeholder-gray-500
                         focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-transparent
                         disabled:opacity-50"
            />
          </div>

          {error && (
            <div className="flex items-center gap-3 px-4 py-3 bg-red-900/30 border border-red-700/50 rounded-xl text-red-300 text-sm">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="flex-shrink-0">
                <circle cx="12" cy="12" r="10"/>
                <line x1="12" y1="8" x2="12" y2="12"/>
                <line x1="12" y1="16" x2="12.01" y2="16"/>
              </svg>
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading || !username.trim() || !password}
            className="w-full py-4 text-lg font-semibold rounded-xl
                       bg-teal-500 hover:bg-teal-400 active:bg-teal-600
                       text-white transition-colors
                       disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {loading ? (
              <span className="flex items-center justify-center gap-2">
                <svg className="animate-spin w-5 h-5" viewBox="0 0 24 24" fill="none">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8z"/>
                </svg>
                로그인 중...
              </span>
            ) : '로그인 및 서비스 시작'}
          </button>
        </form>
      </div>
    </div>
  )
}
