import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useSessionPoller } from '../hooks/useSessionPoller'
import { logout } from '../api/auth'

export default function KioskStandbyPage() {
  const navigate = useNavigate()
  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false)
  const [loggingOut, setLoggingOut] = useState(false)

  useSessionPoller({ interval: 8000, onEndedPath: '/kiosk/standby' })

  // 화면 터치 → 민원인 정보 입력으로
  const handleScreenTap = useCallback(() => {
    if (showLogoutConfirm) return
    navigate('/kiosk/intake')
  }, [showLogoutConfirm, navigate])

  const handleLogout = useCallback(async () => {
    setLoggingOut(true)
    try { await logout() } catch { /* 무시 */ }
    navigate('/kiosk', { replace: true })
  }, [navigate])

  return (
    <div
      className="min-h-screen bg-gray-950 flex flex-col items-center justify-center select-none cursor-pointer relative overflow-hidden"
      onClick={handleScreenTap}
    >
      {/* 배경 장식 */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
                        w-[600px] h-[600px] rounded-full
                        bg-teal-500/5 border border-teal-500/10 animate-pulse" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
                        w-[400px] h-[400px] rounded-full
                        bg-teal-500/8 border border-teal-500/15" />
      </div>

      {/* 메인 콘텐츠 */}
      <div className="relative z-10 flex flex-col items-center gap-8 text-center px-12">
        <div className="w-28 h-28 rounded-full bg-teal-500/15 border-2 border-teal-500/40
                        flex items-center justify-center mb-4">
          <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="#14b8a6" strokeWidth="1.2">
            <path d="M7 11.5V7a5 5 0 0 1 10 0v4.5"/>
            <path d="M2 11.5h20"/>
            <path d="M5 11.5v6a7 7 0 0 0 14 0v-6"/>
            <circle cx="12" cy="18" r="1.5"/>
          </svg>
        </div>

        <div>
          <h1 className="text-4xl font-bold text-white mb-4 leading-tight">
            수어 상담을 시작하려면<br />화면을 터치하세요
          </h1>
          <p className="text-xl text-gray-400">
            Touch the screen to start sign language consultation
          </p>
        </div>

        <div className="mt-4 flex flex-col items-center gap-2 text-teal-400/60">
          <svg className="animate-bounce" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M12 5v14M5 12l7 7 7-7"/>
          </svg>
          <span className="text-sm">터치</span>
        </div>
      </div>

      {/* 로그아웃 버튼 */}
      <button
        onClick={(e) => { e.stopPropagation(); setShowLogoutConfirm(true) }}
        className="absolute top-5 right-5 z-20
                   flex items-center gap-2 px-4 py-2 rounded-xl
                   bg-gray-800/80 hover:bg-gray-700 border border-gray-600/60
                   text-gray-400 hover:text-gray-200 text-sm font-medium
                   transition-colors backdrop-blur-sm"
      >
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
          <polyline points="16 17 21 12 16 7"/>
          <line x1="21" y1="12" x2="9" y2="12"/>
        </svg>
        로그아웃
      </button>

      {/* 로그아웃 확인 모달 */}
      {showLogoutConfirm && (
        <div
          className="absolute inset-0 bg-black/70 z-30 flex items-center justify-center"
          onClick={(e) => { e.stopPropagation(); setShowLogoutConfirm(false) }}
        >
          <div
            className="bg-gray-900 border border-gray-700 rounded-2xl p-8 flex flex-col items-center gap-6 w-80"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex flex-col items-center gap-3 text-center">
              <div className="w-12 h-12 rounded-full bg-red-500/10 border border-red-500/30 flex items-center justify-center">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#f87171" strokeWidth="2">
                  <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/>
                  <polyline points="16 17 21 12 16 7"/>
                  <line x1="21" y1="12" x2="9" y2="12"/>
                </svg>
              </div>
              <p className="text-white text-lg font-semibold">로그아웃 하시겠어요?</p>
              <p className="text-gray-400 text-sm">서비스가 종료되고 로그인 화면으로 이동합니다.</p>
            </div>
            <div className="flex flex-col gap-3 w-full">
              <button
                onClick={handleLogout}
                disabled={loggingOut}
                className="w-full py-3 rounded-xl bg-red-600 hover:bg-red-500 text-white font-semibold transition-colors disabled:opacity-50"
              >
                {loggingOut ? '로그아웃 중...' : '로그아웃'}
              </button>
              <button
                onClick={() => setShowLogoutConfirm(false)}
                className="w-full py-3 rounded-xl bg-gray-800 hover:bg-gray-700 border border-gray-600 text-gray-300 font-medium transition-colors"
              >
                취소
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 하단 상태 */}
      <div className="absolute bottom-8 left-0 right-0 flex justify-center">
        <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-gray-800/60 border border-gray-700/40">
          <div className="w-2 h-2 rounded-full bg-teal-400 animate-pulse" />
          <span className="text-gray-400 text-sm">서비스 준비 완료</span>
        </div>
      </div>
    </div>
  )
}
