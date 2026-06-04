import { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { getSessionState } from '../api/session'
import { ApiError } from '../api/client'

interface Options {
  /** 폴링 간격 (ms). 기본 5000 */
  interval?: number
  /** ended:true 감지 시 이동할 경로. 기본 '/kiosk/standby' */
  onEndedPath?: string
  /** 폴링 활성화 여부 */
  enabled?: boolean
}

/**
 * GET /api/session-state 를 주기적으로 폴링한다.
 * - ended: true  → onEndedPath 로 이동 (기본: /kiosk/standby)
 * - 401           → /kiosk 로 이동 (useAuth 이벤트로 처리됨)
 */
export function useSessionPoller({
  interval = 5000,
  onEndedPath = '/kiosk/standby',
  enabled = true,
}: Options = {}) {
  const navigate = useNavigate()
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (!enabled) return

    let cancelled = false

    async function poll() {
      try {
        const state = await getSessionState()
        if (!cancelled && state.ended) {
          navigate(onEndedPath, { replace: true })
        }
      } catch (err) {
        // 401은 useAuth가 처리. 그 외 네트워크 오류는 무시하고 계속 폴링
        if (err instanceof ApiError && err.status !== 401) {
          console.warn('[useSessionPoller] 오류:', err.message)
        }
      } finally {
        if (!cancelled) {
          timerRef.current = setTimeout(poll, interval)
        }
      }
    }

    poll()

    return () => {
      cancelled = true
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [enabled, interval, navigate, onEndedPath])
}
