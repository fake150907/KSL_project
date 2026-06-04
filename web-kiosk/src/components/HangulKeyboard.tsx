import { useState, useEffect } from 'react'
import * as Hangul from 'hangul-js'

interface HangulKeyboardProps {
  value: string
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
    if (!value) {
      setJamoBuffer([])
    } else {
      // 💡 현재 버퍼를 조합한 결과와 외부에서 들어온 value를 비교합니다.
      const assembled = Hangul.assemble(jamoBuffer)
      
      // 두 값이 다를 때만(외부 주입 또는 초기 렌더링 시) 분해하여 버퍼에 넣습니다.
      if (value !== assembled) {
        setJamoBuffer(Hangul.disassemble(value))
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]) // 💡 jamoBuffer 관련 항목을 제거하여 내부 상태 변경에 의한 루프를 차단합니다.

  const handleKeyPress = (key: string) => {
    if (key === 'Shift') { setIsShift(!isShift); return }
    
    if (key === '지우기') {
      const newBuffer = jamoBuffer.slice(0, -1)
      setJamoBuffer(newBuffer)
      onChange(Hangul.assemble(newBuffer))
      setIsShift(false)
      return
    }
    
    const newBuffer = [...jamoBuffer, key]
    setJamoBuffer(newBuffer)
    onChange(Hangul.assemble(newBuffer))
    setIsShift(false)
  }

  const layout = isShift ? QWERTY_SHIFT : QWERTY_NORMAL

  return (
    <div className="flex flex-col gap-3 w-full max-w-[720px] items-center">
      {/* 화면 렌더링 코드는 기존과 동일하므로 생략 없이 유지합니다. */}
      <div className="w-full h-[60px] md:h-[80px] px-4 md:px-6 flex items-center rounded-xl md:rounded-2xl mb-1 md:mb-2 bg-slate-50 border border-slate-200 shadow-inner">
        <span className="text-xl md:text-3xl font-black tracking-widest text-slate-900">
          {value}
        </span>
        {!value && (
          <span className="text-slate-300 text-lg md:text-2xl font-semibold">홍길동</span>
        )}
        <span className="w-[2px] md:w-[3px] h-6 md:h-8 bg-blue-500 animate-pulse rounded-full ml-1" />
      </div>

      <div className="flex flex-col gap-1.5 md:gap-2.5 w-full p-2 md:p-4 rounded-2xl md:rounded-3xl bg-slate-50 border border-slate-200">
        {layout.map((row, rowIdx) => (
          <div key={rowIdx} className="flex justify-center gap-1 md:gap-2 w-full">
            {row.map((key, keyIdx) => {
              const isAction = key === 'Shift' || key === '지우기'
              const isActiveShift = key === 'Shift' && isShift

              return (
                <button
                  key={`${rowIdx}-${keyIdx}`}
                  onClick={() => handleKeyPress(key)}
                  className={`
                    flex items-center justify-center rounded-lg md:rounded-xl font-bold transition-all duration-100 active:scale-95 select-none border-2 flex-shrink-0
                    ${isAction ? 'px-2 md:px-4 text-[10px] md:text-sm' : 'w-8 h-10 sm:w-10 sm:h-12 md:w-14 md:h-16 text-sm md:text-2xl'}
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