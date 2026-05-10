import { useState, useEffect } from 'react'
import * as Hangul from 'hangul-js'

interface HangulKeyboardProps {
  /** 완성된 이름 문자열 (부모 컴포넌트의 상태) */
  value: string
  /** 글자가 추가/삭제될 때마다 호출 */
  onChange: (name: string) => void
}

const QWERTY_NORMAL = [
  ['ㅂ','ㅈ','ㄷ','ㄱ','ㅅ','ㅛ','ㅕ','ㅑ','ㅐ','ㅔ'],
  ['ㅁ','ㄴ','ㅇ','ㄹ','ㅎ','ㅗ','ㅓ','ㅏ','ㅣ'],
  ['Shift', 'ㅋ','ㅌ','ㅊ','ㅍ','ㅠ','ㅜ','ㅡ', '지우기']
]

const QWERTY_SHIFT = [
  ['ㅃ','ㅉ','ㄸ','ㄲ','ㅆ','ㅛ','ㅕ','ㅑ','ㅒ','ㅖ'],
  ['ㅁ','ㄴ','ㅇ','ㄹ','ㅎ','ㅗ','ㅓ','ㅏ','ㅣ'],
  ['Shift', 'ㅋ','ㅌ','ㅊ','ㅍ','ㅠ','ㅜ','ㅡ', '지우기']
]

export default function HangulKeyboard({ value, onChange }: HangulKeyboardProps) {
  const [jamoBuffer, setJamoBuffer] = useState<string[]>([])
  const [isShift, setIsShift] = useState(false)

  useEffect(() => {
    if (value && jamoBuffer.length === 0) {
      setJamoBuffer(Hangul.disassemble(value))
    }
  }, [])

  const handleKeyPress = (key: string) => {
    if (key === 'Shift') {
      setIsShift(!isShift)
      return
    }

    if (key === '지우기') {
      setJamoBuffer(prev => {
        const newBuffer = prev.slice(0, -1)
        onChange(Hangul.assemble(newBuffer))
        return newBuffer
      })
      setIsShift(false)
      return
    }

    setJamoBuffer(prev => {
      const newBuffer = [...prev, key]
      onChange(Hangul.assemble(newBuffer))
      return newBuffer
    })
    setIsShift(false)
  }

  const layout = isShift ? QWERTY_SHIFT : QWERTY_NORMAL

  return (
    <div className="flex flex-col gap-4 w-full max-w-[720px] items-center">
      {/* 🖥️ 입력 디스플레이 영역 (화이트 모드) */}
      <div className="w-full h-[80px] px-6 flex items-center rounded-2xl mb-2 bg-slate-50 border border-slate-200 shadow-inner">
        <span className="text-3xl font-black tracking-widest text-slate-900">
          {value}
        </span>
        {!value && (
          <span className="text-slate-300 text-2xl font-semibold">홍길동</span>
        )}
        <span className="w-[3px] h-8 bg-blue-500 animate-pulse rounded-full ml-1" />
      </div>

      {/* ⌨️ 키보드 자판 영역 */}
      <div className="flex flex-col gap-2.5 w-full p-4 rounded-3xl bg-slate-50 border border-slate-200">
        {layout.map((row, rowIdx) => (
          <div key={rowIdx} className="flex justify-center gap-2 w-full">
            {row.map((key, keyIdx) => {
              const isAction = key === 'Shift' || key === '지우기'
              const isActiveShift = key === 'Shift' && isShift

              return (
                <button
                  key={`${rowIdx}-${keyIdx}`}
                  onClick={() => handleKeyPress(key)}
                  className={`
                    flex items-center justify-center rounded-xl font-bold transition-all duration-100 active:scale-95 select-none border-2
                    ${isAction ? 'px-4 text-sm' : 'w-14 h-16 text-2xl'}
                  `}
                  style={{
                    background: isActiveShift ? '#3B82F6' : '#FFFFFF',
                    borderColor: isActiveShift ? '#2563EB' : '#E2E8F0',
                    color: isActiveShift ? '#FFFFFF' : isAction ? '#64748B' : '#1E293B',
                    boxShadow: isActiveShift ? '0 4px 12px rgba(59,130,246,0.3)' : '0 2px 4px rgba(0,0,0,0.02)',
                    flex: isAction ? 1 : 'none'
                  }}
                >
                  {key}
                </button>
              )
            })}
          </div>
        ))}
      </div>
    </div>
  )
}