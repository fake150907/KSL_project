import { useEffect, useRef, useState } from 'react'
import { fetchMessages } from '../api/session'
import type { ChatMessage } from '../types'

interface Options {
  interval?: number
  enabled?: boolean
}

/**
 * GET /api/messages 를 주기적으로 폴링해 최신 메시지 목록을 반환한다.
 * 상담원이 보낸 메시지를 키오스크 화면에 실시간으로 반영하는 데 사용.
 */
export function useMessagePoller({ interval = 3000, enabled = true }: Options = {}) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const seenIds = useRef<Set<string>>(new Set())
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (!enabled) return

    let cancelled = false

    async function poll() {
      try {
        const fetched = await fetchMessages()
        if (cancelled) return

        // 새 메시지만 누적 (중복 방지)
        const next: ChatMessage[] = []
        for (const msg of fetched) {
          if (!seenIds.current.has(msg.id)) {
            seenIds.current.add(msg.id)
            next.push(msg)
          }
        }
        if (next.length > 0) {
          setMessages((prev) => [...prev, ...next])
        }
      } catch {
        // 401은 useAuth가 처리. 네트워크 오류는 무시
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
  }, [enabled, interval])

  const resetMessages = () => {
    seenIds.current.clear()
    setMessages([])
  }

  return { messages, resetMessages }
}
