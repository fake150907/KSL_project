import { useEffect, useRef, useState } from 'react'

// ─── 타입 ────────────────────────────────────────────────────────────────────

export type LogLevel = 'error' | 'warning' | 'info' | 'success'

export interface LogEntry {
  id: number
  timestamp: string
  level: LogLevel
  source: string
  message: string
  status?: number
  method?: string
  path?: string
}

// ─── 상수 ────────────────────────────────────────────────────────────────────

const MAX_CLIENT_LOGS = 1000
const RECONNECT_DELAY_MS = 3_000

/**
 * 서버는 메시지가 없을 때 25초마다 ping을 전송합니다.
 * 35초 안에 아무 이벤트도 수신되지 않으면 연결 끊김으로 판단합니다.
 */
const HEARTBEAT_TIMEOUT_MS = 35_000
const HEARTBEAT_CHECK_INTERVAL_MS = 5_000

// ─── hook ────────────────────────────────────────────────────────────────────

/**
 * 백엔드 SSE 스트림(`/api/logs/stream`)에 항상 연결을 유지합니다.
 * - 연결이 끊기면 3초 후 자동으로 재연결합니다.
 * - 35초 동안 이벤트가 없으면 연결 끊김으로 판단합니다.
 */
export function useLogStream() {
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [connected, setConnected] = useState(false)

  const seenIdsRef = useRef<Set<number>>(new Set())
  const esRef = useRef<EventSource | null>(null)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const heartbeatTimerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const lastEventAtRef = useRef<number>(0)
  const unmountedRef = useRef(false)

  useEffect(() => {
    unmountedRef.current = false

    // 하트비트 감시
    heartbeatTimerRef.current = setInterval(() => {
      if (unmountedRef.current) return
      const last = lastEventAtRef.current
      if (last > 0 && Date.now() - last > HEARTBEAT_TIMEOUT_MS) {
        setConnected(false)
      }
    }, HEARTBEAT_CHECK_INTERVAL_MS)

    const connect = () => {
      if (unmountedRef.current) return

      const es = new EventSource('/api/logs/stream')
      esRef.current = es

      es.onopen = () => {
        if (unmountedRef.current) return
        lastEventAtRef.current = Date.now()
        setConnected(true)
      }

      es.onmessage = (event: MessageEvent<string>) => {
        if (unmountedRef.current) return
        lastEventAtRef.current = Date.now()
        setConnected(true)

        try {
          const parsed: Record<string, unknown> = JSON.parse(event.data)
          // ping 메시지는 연결 확인만 하고 무시
          if (!parsed.id) return

          const entry = parsed as unknown as LogEntry
          if (seenIdsRef.current.has(entry.id)) return
          seenIdsRef.current.add(entry.id)

          setLogs((prev) => {
            const next = [...prev, entry]
            return next.length > MAX_CLIENT_LOGS
              ? next.slice(next.length - MAX_CLIENT_LOGS)
              : next
          })
        } catch {
          // JSON 파싱 오류 무시
        }
      }

      es.onerror = () => {
        if (unmountedRef.current) return
        setConnected(false)
        lastEventAtRef.current = 0
        es.close()
        reconnectTimerRef.current = setTimeout(connect, RECONNECT_DELAY_MS)
      }
    }

    connect()

    return () => {
      unmountedRef.current = true
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
      if (heartbeatTimerRef.current) clearInterval(heartbeatTimerRef.current)
      esRef.current?.close()
    }
  }, [])

  /** 화면 표시 로그를 지웁니다. 서버 측 누적 데이터는 유지됩니다. */
  const clearDisplay = () => {
    setLogs([])
    seenIdsRef.current.clear()
  }

  return { logs, connected, clearDisplay }
}
