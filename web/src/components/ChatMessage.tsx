import type { ChatMessage as ChatMessageType } from '../types'

interface ChatMessageProps {
  message: ChatMessageType
  textClassName?: string
  dark?: boolean
  /** 'citizen' messages align left, 'agent' messages align right */
}

export default function ChatMessage({ message, textClassName = 'text-sm', dark = false }: ChatMessageProps) {
  const isAgent = message.sender === 'agent'
  const time = message.timestamp.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })

  return (
    <div className={`flex items-end gap-2 ${isAgent ? 'flex-row-reverse' : 'flex-row'}`}>
      {/* Avatar */}
      <div className={`
        flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center text-[10px] font-bold border
        ${isAgent
          ? dark ? 'bg-[#172b4a] border-[#2f5f9f] text-[#93c5fd]' : 'bg-[#eff6ff] border-[#bfdbfe] text-[#2563eb]'
          : dark ? 'bg-[#123334] border-[#1f5f59] text-[#5eead4]' : 'bg-[#ecfdf5] border-[#a7f3d0] text-[#0f766e]'}
      `}>
        {isAgent
          ? <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
          : <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
        }
      </div>

      <div className={`flex min-w-0 max-w-[75%] flex-col gap-1 ${isAgent ? 'items-end' : 'items-start'}`}>
        {/* Label */}
        {message.label && (
          <span className={`px-1 text-[10px] font-semibold ${dark ? 'text-[#8fa3bd]' : 'text-[#64748b]'}`}>
            {message.label}
          </span>
        )}
        {/* Bubble */}
        <div className={`
          break-words px-3 py-2 rounded-2xl font-medium leading-relaxed ${textClassName}
          ${isAgent
            ? dark ? 'bg-[#2563eb] text-white rounded-tr-sm border border-[#60a5fa]/30 shadow-sm shadow-blue-950/20' : 'bg-[#2563eb] text-white rounded-tr-sm border border-[#93c5fd] shadow-sm shadow-blue-100'
            : dark ? 'bg-[#14b8a6] text-[#042f2e] rounded-tl-sm shadow-sm shadow-teal-950/20' : 'bg-[#ccfbf1] text-[#134e4a] rounded-tl-sm border border-[#99f6e4] shadow-sm shadow-teal-50'}
        `}>
          {message.text}
        </div>
        {/* Time */}
        <span className={`px-1 text-[10px] ${dark ? 'text-[#8fa3bd]' : 'text-[#64748b]'}`}>{time}</span>
      </div>
    </div>
  )
}
