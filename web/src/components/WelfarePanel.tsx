import { useEffect, useState } from 'react'

export interface WelfarePanelItem {
  serv_id: string
  title: string
  summary: string
  agency: string
  phone: string
  website: string
  detail_link: string
  apply_steps: string[]
}

type WelfarePanelTheme = 'light' | 'dark'

interface Props {
  items: WelfarePanelItem[]
  intervalMs?: number
  onClose?: () => void
  compact?: boolean
  theme?: WelfarePanelTheme
}

const PANEL_THEME = {
  light: {
    panel: {
      background: 'linear-gradient(160deg, #ffffff 0%, #f3f8ff 52%, #eefcf8 100%)',
      border: '1px solid rgba(96, 165, 250, 0.28)',
      boxShadow: '0 18px 38px rgba(15, 23, 42, 0.08)',
    },
    eyebrow: '#2563eb',
    counter: '#64748b',
    title: '#172033',
    body: '#475569',
    muted: '#64748b',
    link: '#0f766e',
    pillBg: 'rgba(219, 234, 254, 0.82)',
    pillBorder: 'rgba(96, 165, 250, 0.38)',
    stepBg: 'rgba(204, 251, 241, 0.78)',
    dotInactive: 'rgba(100, 116, 139, 0.24)',
    closeBorder: 'rgba(100, 116, 139, 0.18)',
    closeHover: '#e2e8f0',
  },
  dark: {
    panel: {
      background: 'linear-gradient(160deg, #142033 0%, #17263c 54%, #123334 100%)',
      border: '1px solid rgba(125, 211, 252, 0.18)',
      boxShadow: '0 20px 42px rgba(2, 6, 23, 0.38)',
    },
    eyebrow: '#7dd3fc',
    counter: '#8fa3bd',
    title: '#f8fafc',
    body: '#cbd7e6',
    muted: '#94a3b8',
    link: '#99f6e4',
    pillBg: 'rgba(14, 165, 233, 0.16)',
    pillBorder: 'rgba(125, 211, 252, 0.26)',
    stepBg: 'rgba(45, 212, 191, 0.13)',
    dotInactive: 'rgba(203, 213, 225, 0.2)',
    closeBorder: 'rgba(203, 213, 225, 0.16)',
    closeHover: 'rgba(255, 255, 255, 0.08)',
  },
} satisfies Record<WelfarePanelTheme, Record<string, string | Record<string, string>>>

export function WelfarePanel({ items, intervalMs = 7000, onClose, compact = false, theme = 'light' }: Props) {
  const [idx, setIdx] = useState(0)
  const colors = PANEL_THEME[theme]

  useEffect(() => {
    setIdx(0)
  }, [items])

  useEffect(() => {
    if (items.length <= 1) return
    const id = window.setInterval(() => {
      setIdx((prev) => (prev + 1) % items.length)
    }, intervalMs)
    return () => window.clearInterval(id)
  }, [items, intervalMs])

  if (items.length === 0) return null

  const card = items[idx]
  const steps = card.apply_steps.slice(0, 3)

  return (
    <div
      className={`w-full ${compact ? 'flex h-full min-h-[220px] flex-col rounded-lg p-3 sm:min-h-[260px] sm:p-3.5' : 'rounded-2xl p-5'}`}
      style={colors.panel}
    >
      <div className={`flex items-start justify-between gap-2 ${compact ? 'mb-2' : 'mb-3'}`}>
        <span className="min-w-0 flex-1 break-keep text-[10px] font-black leading-snug tracking-wide sm:text-[11px]" style={{ color: colors.eyebrow }}>
          상담 중 확인할 수 있는 복지 서비스
        </span>
        <div className="flex shrink-0 items-center gap-2">
          <span className="text-xs font-semibold" style={{ color: colors.counter }}>
            {idx + 1} / {items.length}
          </span>
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              aria-label="닫기"
              className="flex h-6 w-6 items-center justify-center rounded-full text-sm font-bold transition-colors"
              style={{
                color: colors.muted,
                border: `1px solid ${colors.closeBorder}`,
              }}
              onMouseEnter={(event) => {
                event.currentTarget.style.background = colors.closeHover
              }}
              onMouseLeave={(event) => {
                event.currentTarget.style.background = 'transparent'
              }}
            >
              X
            </button>
          )}
        </div>
      </div>

      <h3 className={`${compact ? 'break-keep text-sm leading-snug sm:text-base' : 'text-lg'} mb-1 font-black`} style={{ color: colors.title }}>
        {card.title}
      </h3>
      <p className={`${compact ? 'mb-3 line-clamp-3 text-xs leading-relaxed sm:line-clamp-4 sm:text-sm' : 'mb-4 text-sm leading-relaxed'}`} style={{ color: colors.body }}>
        {card.summary}
      </p>

      <div className={`flex min-w-0 items-center gap-2 sm:gap-3 ${compact ? 'mb-2' : 'mb-4'}`}>
        <div
          className={`${compact ? 'shrink-0 px-2 py-1 text-sm sm:px-2.5 sm:py-1.5 sm:text-base' : 'px-3 py-2 text-xl'} rounded-lg font-black`}
          style={{
            background: colors.pillBg,
            border: `1px solid ${colors.pillBorder}`,
            color: colors.eyebrow,
          }}
        >
          전화 {card.phone}
        </div>
        <div className="min-w-0 break-keep text-[11px] font-semibold leading-snug sm:text-xs" style={{ color: colors.muted }}>
          {card.agency}
        </div>
      </div>

      {steps.length > 0 && (
        <ol className={`${compact ? 'mb-2 space-y-1' : 'mb-4 space-y-1.5'}`}>
          {steps.map((step, i) => (
            <li key={i} className="flex gap-2 text-xs" style={{ color: colors.body }}>
              <span
                className="flex h-4 w-4 flex-shrink-0 items-center justify-center rounded-full text-[11px] font-black"
                style={{
                  background: colors.stepBg,
                  color: colors.eyebrow,
                }}
              >
                {i + 1}
              </span>
              <span className={`${compact ? 'line-clamp-1 sm:line-clamp-2' : ''} min-w-5 break-keep leading-snug`}>{step}</span>
            </li>
          ))}
        </ol>
      )}

      <div className="mt-auto flex min-w-0 items-center justify-between gap-2">
        <a href={card.detail_link} target="_blank" rel="noreferrer" className="min-w-0 break-keep text-xs font-bold underline" style={{ color: colors.link }}>
          복지로에서 자세히 보기
        </a>
        {items.length > 1 && (
          <div className="flex shrink-0 gap-1.5">
            {items.map((_, i) => (
              <button
                key={i}
                onClick={() => setIdx(i)}
                className="h-2 w-2 rounded-full transition-all"
                style={{
                  background: i === idx ? colors.eyebrow : colors.dotInactive,
                }}
                aria-label={`슬라이드 ${i + 1}`}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
