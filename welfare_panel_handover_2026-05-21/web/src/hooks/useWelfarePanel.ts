/**
 * Standalone welfare-panel hook.
 *
 * Usage in any page/component:
 *
 *   const { welfarePanel, dismiss } = useWelfarePanel(lookupKey)
 *   {welfarePanel.length > 0 && <WelfarePanel items={welfarePanel} onClose={dismiss} />}
 *
 * The hook owns its own state and HTTP call to `GET /api/welfare_panel`.
 * It does NOT touch any other hook, type, or response shape, so it can be
 * dropped into a page without coupling to the recognition pipeline.
 *
 * Trigger contract:
 *   - Pass `lookupKey` (string | undefined). Pass the same `lookup_key`
 *     that the backend recognition pipeline already returns.
 *   - The backend (`/api/welfare_panel`) decides whether the key is a
 *     welfare-card scenario and returns either the 3 cards or [].
 *   - The hook re-fetches whenever `lookupKey` changes to a new non-empty value.
 *   - Once `welfarePanel` is set, it stays visible until `dismiss()` is called
 *     or `useWelfarePanel(undefined)` is invoked.
 */

import { useCallback, useEffect, useRef, useState } from 'react'
import type { WelfarePanelItem } from '../components/WelfarePanel'

interface UseWelfarePanelResult {
  welfarePanel: WelfarePanelItem[]
  dismiss: () => void
}

export function useWelfarePanel(
  lookupKey: string | null | undefined,
): UseWelfarePanelResult {
  const [welfarePanel, setWelfarePanel] = useState<WelfarePanelItem[]>([])
  const lastFetchedKeyRef = useRef<string | null>(null)
  const dismissedKeysRef = useRef<Set<string>>(new Set())

  useEffect(() => {
    const key = (lookupKey || '').trim()
    if (!key) return
    if (lastFetchedKeyRef.current === key) return
    if (dismissedKeysRef.current.has(key)) return

    lastFetchedKeyRef.current = key
    const controller = new AbortController()
    fetch(`/api/welfare_panel?lookup_key=${encodeURIComponent(key)}`, {
      signal: controller.signal,
    })
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!data) return
        const panel = Array.isArray(data.welfare_panel) ? data.welfare_panel : []
        if (panel.length > 0) setWelfarePanel(panel)
      })
      .catch((err) => {
        if ((err as Error).name !== 'AbortError') {
          console.warn('[useWelfarePanel] fetch failed:', err)
        }
      })

    return () => controller.abort()
  }, [lookupKey])

  const dismiss = useCallback(() => {
    const key = lastFetchedKeyRef.current
    if (key) dismissedKeysRef.current.add(key)
    setWelfarePanel([])
  }, [])

  return { welfarePanel, dismiss }
}
