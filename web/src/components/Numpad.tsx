interface NumpadProps {
  /** 키 눌림 이벤트. '지우기' | '전체삭제' | '0'~'9' 중 하나가 전달됩니다 */
  onPress: (val: string) => void
}

const KEYS = ['1','2','3','4','5','6','7','8','9','지우기','0','전체삭제']

export default function Numpad({ onPress }: NumpadProps) {
  return (
    <div className="grid grid-cols-3 gap-3 w-full max-w-[340px]">
      {KEYS.map((k) => {
        const isAction = k === '지우기' || k === '전체삭제'
        
        return (
          <button
            key={k}
            onClick={() => onPress(k)}
            className={`
              h-[72px] rounded-2xl font-black transition-all duration-100 select-none border-2 active:scale-[0.91]
              ${isAction 
                ? 'text-sm bg-slate-100 border-slate-200 text-slate-500 hover:bg-slate-200' 
                : 'text-2xl bg-white border-slate-200 text-slate-800 hover:border-blue-400 hover:bg-blue-50 hover:text-blue-600'
              }
            `}
            style={{
              boxShadow: isAction ? 'none' : '0 2px 4px rgba(0,0,0,0.02)'
            }}
          >
            {k}
          </button>
        )
      })}
    </div>
  )
}