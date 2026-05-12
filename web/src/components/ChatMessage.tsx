import type { ChatMessage as ChatMessageType } from '../types'

interface ChatMessageProps {
  message: ChatMessageType
  /** 'patient' messages align left, 'doctor' messages align right */
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isDoctor = message.sender === 'doctor'
  const time = message.timestamp.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })

  return (
    <div className={`flex items-end gap-2 ${isDoctor ? 'flex-row-reverse' : 'flex-row'}`}>
      {/* Avatar */}
      <div className={`
        flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-[10px] font-bold border
        ${isDoctor
          ? 'bg-[rgba(59,130,246,0.15)] border-[rgba(59,130,246,0.3)] text-blue-300'
          : 'bg-[rgba(34,197,94,0.12)] border-[rgba(34,197,94,0.25)] text-green-400'}
      `}>
        {isDoctor
          ? <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
          : <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
        }
      </div>

      <div className={`flex flex-col gap-1 max-w-[75%] ${isDoctor ? 'items-end' : 'items-start'}`}>
        {/* Label */}
        {message.label && (
          <span className="text-[10px] font-semibold text-[#94a3b8] px-1">
            {message.label}
          </span>
        )}
        {/* Bubble */}
        <div className={`
          px-3 py-2 rounded-2xl text-sm font-medium leading-relaxed
          ${isDoctor
            ? 'bg-[rgba(37,99,235,0.75)] text-white rounded-tr-sm border border-[rgba(96,165,250,0.3)]'
            : 'bg-[#22c55e] text-[#052e16] rounded-tl-sm'}
        `}>
          {message.text}
        </div>
        {/* Time */}
        <span className="text-[10px] text-[#64748b] px-1">{time}</span>
      </div>
    </div>
  )
}