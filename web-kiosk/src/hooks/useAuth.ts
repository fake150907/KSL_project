import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { AUTH_EXPIRED_EVENT } from '../api/client'

/**
 * 앱 루트에서 한 번만 마운트.
 * apiFetch가 401을 받으면 AUTH_EXPIRED_EVENT를 발행하고,
 * 이 훅이 로그인 화면으로 리다이렉트한다.
 */
export function useAuthExpiredRedirect() {
  const navigate = useNavigate()

  useEffect(() => {
    const handler = () => navigate('/kiosk', { replace: true })
    window.addEventListener(AUTH_EXPIRED_EVENT, handler)
    return () => window.removeEventListener(AUTH_EXPIRED_EVENT, handler)
  }, [navigate])
}
