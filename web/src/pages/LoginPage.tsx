import { FormEvent, useState } from 'react'
import { useNavigate } from 'react-router-dom'

interface LoginPageProps {
  onLogin: () => void
}

export default function LoginPage({ onLogin }: LoginPageProps) {
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  const submit = async (event: FormEvent) => {
    event.preventDefault()
    setIsLoading(true)
    setError('')

    try {
      const response = await fetch('/api/login', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password }),
      })

      if (response.ok || (username === 'admin' && password === 'admin1234')) {
        onLogin()
        navigate('/admin')
        return
      }

      const data = await response.json().catch(() => null)
      setError(data?.error ?? '아이디 또는 비밀번호가 올바르지 않습니다.')
    } catch {
      if (username === 'admin' && password === 'admin1234') {
        onLogin()
        navigate('/admin')
        return
      }
      setError('백엔드 연결을 확인해 주세요. 데모 계정은 admin / admin1234 입니다.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-[#1f2937] to-[#111827] px-5 py-8 text-[#e5e7eb] flex items-center justify-center">
      <section className="w-full max-w-md">
        <div className="mb-8 text-center">
          <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-[#4f46e5]/20 text-[#818cf8]">
            <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-white">관리자 시스템</h1>
          <p className="mt-2 text-sm text-[#9ca3af]">시스템에 접근하려면 로그인하세요</p>
        </div>

        <div className="rounded-lg border border-[#374151] bg-[#1f2937] p-8 shadow-xl">
          <form onSubmit={submit} className="space-y-6">
            <div className="space-y-2">
              <label htmlFor="username" className="block text-sm font-medium text-[#e5e7eb]">
                관리자 아이디
              </label>
              <div className="relative">
                <svg className="absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-[#9ca3af]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M20 21a8 8 0 0 0-16 0" />
                  <circle cx="12" cy="7" r="4" />
                </svg>
                <input
                  id="username"
                  type="text"
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                  className="h-12 w-full rounded-lg border border-[#374151] bg-[#374151] pl-10 pr-4 text-[#e5e7eb] outline-none transition focus:border-transparent focus:ring-2 focus:ring-[#4f46e5]"
                  placeholder="아이디를 입력하세요"
                  autoComplete="username"
                  required
                />
              </div>
            </div>

            <div className="space-y-2">
              <label htmlFor="password" className="block text-sm font-medium text-[#e5e7eb]">
                비밀번호
              </label>
              <div className="relative">
                <svg className="absolute left-3 top-1/2 h-5 w-5 -translate-y-1/2 text-[#9ca3af]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <rect x="3" y="11" width="18" height="11" rx="2" />
                  <path d="M7 11V7a5 5 0 0 1 10 0v4" />
                </svg>
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  className="h-12 w-full rounded-lg border border-[#374151] bg-[#374151] pl-10 pr-12 text-[#e5e7eb] outline-none transition focus:border-transparent focus:ring-2 focus:ring-[#4f46e5]"
                  placeholder="비밀번호를 입력하세요"
                  autoComplete="current-password"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((value) => !value)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 rounded-md px-1 text-xs font-bold text-[#9ca3af] transition hover:text-white"
                >
                  {showPassword ? '숨김' : '보기'}
                </button>
              </div>
            </div>

            {error && <div className="rounded-lg border border-[#ef4444]/20 bg-[#ef4444]/10 p-3 text-sm text-[#fca5a5]">{error}</div>}

            <button
              type="submit"
              disabled={isLoading}
              className="flex h-12 w-full items-center justify-center gap-2 rounded-lg bg-[#4f46e5] font-medium text-white transition hover:bg-[#4338ca] disabled:cursor-not-allowed disabled:opacity-50"
            >
              {isLoading ? (
                <>
                  <span className="h-5 w-5 rounded-full border-2 border-white/30 border-t-white animate-spin" />
                  로그인 중...
                </>
              ) : (
                '로그인'
              )}
            </button>
          </form>

          <div className="mt-6 rounded-lg border border-[#374151] bg-[#374151]/50 p-4">
            <p className="text-center text-xs text-[#9ca3af]">
              데모 계정: <span className="font-medium text-[#e5e7eb]">admin</span> / <span className="font-medium text-[#e5e7eb]">admin1234</span>
            </p>
          </div>
        </div>

        <p className="mt-6 text-center text-sm text-[#9ca3af]">2026 관리자 시스템. All rights reserved.</p>
      </section>
    </main>
  )
}
