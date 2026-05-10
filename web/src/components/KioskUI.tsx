import { Step } from './hangul'

const STEP_LABELS = ['이름', '생년월일', '성별', '연락처', '정보 확인']
const STEP_KEYS = ['name', 'dob', 'gender', 'phone', 'confirm']

export function StepBar({ current }: { current: string }) {
  const idx = STEP_KEYS.indexOf(current)
  if (idx === -1) return null
  return (
    <div className="flex items-center gap-1.5">
      {STEP_LABELS.map((label, i) => {
        const done = i < idx
        const active = i === idx
        return (
          <div key={label} className="flex items-center gap-1.5">
            <div className="flex items-center gap-1.5">
              <div
                className="w-5 h-5 rounded-full flex items-center justify-center text-[9px] font-black transition-all duration-300"
                style={{
                  background: done ? '#22d3ee' : active ? 'rgba(34,211,238,0.12)' : 'rgba(255,255,255,0.04)',
                  border: `1px solid ${done ? '#22d3ee' : active ? 'rgba(34,211,238,0.5)' : 'rgba(255,255,255,0.08)'}`,
                  color: done ? '#0b1120' : active ? '#22d3ee' : '#334155',
                }}
              >
                {done ? '✓' : i + 1}
              </div>
              <span
                className="text-xs font-semibold transition-colors duration-300"
                style={{ color: active ? '#22d3ee' : done ? '#475569' : '#1e293b' }}
              >
                {label}
              </span>
            </div>
            {i < STEP_LABELS.length - 1 && (
              <div
                className="w-6 h-px transition-all duration-300"
                style={{ background: done ? 'rgba(34,211,238,0.4)' : 'rgba(255,255,255,0.06)' }}
              />
            )}
          </div>
        )
      })}
    </div>
  )
}

export function Header({ step }: { step: string }) {
  return (
    <header className="relative flex-shrink-0 flex items-center justify-between px-8 py-4 border-b border-white/[0.04]">
      <div className="flex items-center gap-2.5">
        <div className="w-6 h-6 rounded-lg bg-cyan-400/10 border border-cyan-400/20 flex items-center justify-center">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#22d3ee" strokeWidth="2.5">
            <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
          </svg>
        </div>
        <span className="text-[11px] font-bold tracking-[0.18em] text-slate-600 uppercase">수어 통역 시스템</span>
      </div>
      <StepBar current={step} />
    </header>
  )
}

interface NavProps {
  onPrev?: () => void
  onNext?: () => void
  nextLabel?: string
  nextDisabled?: boolean
  nextGreen?: boolean
}

export function Nav({ onPrev, onNext, nextLabel = '다음', nextDisabled = false, nextGreen = false }: NavProps) {
  return (
    <div className="flex gap-2.5 w-full max-w-[340px]">
      {onPrev && (
        <button
          onClick={onPrev}
          className="flex-1 h-14 rounded-2xl text-sm font-bold transition-all active:scale-[0.96]"
          style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.07)', color: '#475569' }}
        >이전</button>
      )}
      {onNext && (
        <button
          onClick={onNext}
          disabled={nextDisabled}
          className="flex-[2] h-14 rounded-2xl text-sm font-black text-slate-900 transition-all active:scale-[0.96] disabled:opacity-20 disabled:cursor-not-allowed"
          style={{
            background: nextGreen ? 'linear-gradient(135deg,#22c55e,#16a34a)' : 'linear-gradient(135deg,#22d3ee,#3b82f6)',
            boxShadow: nextDisabled ? 'none' : nextGreen ? '0 0 28px rgba(34,197,94,0.25)' : '0 0 28px rgba(34,211,238,0.2)',
          }}
        >{nextLabel}</button>
      )}
    </div>
  )
}