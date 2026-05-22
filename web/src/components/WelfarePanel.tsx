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

interface Props {
  items: WelfarePanelItem[]
  intervalMs?: number
  onClose?: () => void
}

export function WelfarePanel({ items, intervalMs = 7000, onClose }: Props) {
  const [idx, setIdx] = useState(0)

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
      className="w-full rounded-2xl p-5"
      style={{
        background: 'linear-gradient(160deg, rgba(34,211,238,0.08), rgba(99,102,241,0.06))',
        border: '1px solid rgba(34,211,238,0.25)',
      }}
    >
      <div className="flex items-center justify-between mb-3">
        <span
          className="text-[11px] font-bold uppercase tracking-wider"
          style={{ color: '#22d3ee' }}
        >
          📌 기다리는 동안 — 청각장애인 지원 서비스
        </span>
        <div className="flex items-center gap-2">
          <span className="text-xs" style={{ color: '#64748b' }}>
            {idx + 1} / {items.length}
          </span>
          {onClose && (
            <button
              type="button"
              onClick={onClose}
              aria-label="닫기"
              className="w-6 h-6 rounded-full flex items-center justify-center text-sm font-bold transition-colors hover:bg-white/10"
              style={{
                color: '#94a3b8',
                border: '1px solid rgba(255,255,255,0.12)',
              }}
            >
              ✕
            </button>
          )}
        </div>
      </div>

      <h3 className="text-lg font-bold mb-1" style={{ color: '#e2e8f0' }}>
        {card.title}
      </h3>
      <p className="text-sm mb-4 leading-relaxed" style={{ color: '#cbd5e1' }}>
        {card.summary}
      </p>

      <div className="flex items-center gap-3 mb-4">
        <div
          className="px-3 py-2 rounded-lg font-black text-xl"
          style={{
            background: 'rgba(34,211,238,0.15)',
            border: '1px solid rgba(34,211,238,0.4)',
            color: '#22d3ee',
          }}
        >
          ☎ {card.phone}
        </div>
        <div className="text-xs" style={{ color: '#94a3b8' }}>
          {card.agency}
        </div>
      </div>

      {steps.length > 0 && (
        <ol className="space-y-1.5 mb-4">
          {steps.map((step, i) => (
            <li
              key={i}
              className="text-xs flex gap-2"
              style={{ color: '#cbd5e1' }}
            >
              <span
                className="flex-shrink-0 w-4 h-4 rounded-full flex items-center justify-center font-bold text-[10px]"
                style={{
                  background: 'rgba(34,211,238,0.2)',
                  color: '#22d3ee',
                }}
              >
                {i + 1}
              </span>
              <span className="leading-snug">{step}</span>
            </li>
          ))}
        </ol>
      )}

      <div className="flex items-center justify-between">
        <a
          href={card.detail_link}
          target="_blank"
          rel="noreferrer"
          className="text-xs underline"
          style={{ color: '#7dd3fc' }}
        >
          복지로에서 자세히 보기 →
        </a>
        {items.length > 1 && (
          <div className="flex gap-1.5">
            {items.map((_, i) => (
              <button
                key={i}
                onClick={() => setIdx(i)}
                className="w-2 h-2 rounded-full transition-all"
                style={{
                  background: i === idx ? '#22d3ee' : 'rgba(255,255,255,0.15)',
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
