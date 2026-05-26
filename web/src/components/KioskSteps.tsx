import HangulKeyboard from './HangulKeyboard'
import Numpad from './Numpad'
import SignLanguageLogo from './SignLanguageLogo'
import { Header, Nav } from './KioskUI'
import { formatPhone, maskName, maskPhone } from './hangul'
import type { CitizenData, Step } from './hangul'

export interface StepProps {
  data: CitizenData
  setData: React.Dispatch<React.SetStateAction<CitizenData>>
  go: (s: Step) => void
  onFinish?: () => void
}

const handleNumpadLogic = (field: 'dob' | 'phone', val: string, max: number, minGuard: number, setData: React.Dispatch<React.SetStateAction<CitizenData>>) => {
  setData(prev => {
    const cur = prev[field].replace(/\D/g, '')
    if (val === '지우기') return cur.length <= minGuard ? prev : { ...prev, [field]: cur.slice(0, -1) }
    if (val === '전체삭제') return { ...prev, [field]: field === 'phone' ? '010' : '' }
    if (cur.length < max) return { ...prev, [field]: cur + val }
    return prev
  })
}

const isValidDate = (dobStr: string) => {
  if (dobStr.length !== 6) return false
  const yy = parseInt(dobStr.substring(0, 2), 10); const mm = parseInt(dobStr.substring(2, 4), 10); const dd = parseInt(dobStr.substring(4, 6), 10)
  if (mm < 1 || mm > 12) return false
  const monthDays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
  const fullYear = yy > 24 ? 1900 + yy : 2000 + yy
  if ((fullYear % 4 === 0 && fullYear % 100 !== 0) || fullYear % 400 === 0) monthDays[1] = 29
  return dd >= 1 && dd <= monthDays[mm - 1]
}

// 1. 시작 화면
export function StepStart({ go }: Pick<StepProps, 'go'>) {
  return (
    <div className="h-full w-full flex flex-col items-center justify-center bg-white text-slate-900 overflow-hidden relative px-4">
      <div className="relative z-10 flex flex-col items-center gap-10 w-full max-w-lg">
        <SignLanguageLogo className="h-20 w-20" />
        <div className="text-center">
          <h1 className="text-4xl font-black tracking-tight text-slate-800 mb-3">수어 통역 시스템</h1>
          <p className="text-slate-500 text-sm font-medium">민원인 접수 키오스크</p>
        </div>
        <button 
          onClick={() => go('name')} 
          className="group flex flex-row items-center gap-5 px-10 py-6 rounded-[22px] transition-all duration-200 active:scale-[0.97] shadow-lg bg-gradient-to-r from-[#0d6fb8] to-[#3cd3c1] w-full"
        >
          <SignLanguageLogo className="h-12 w-12 shadow-none" />
          <div className="text-left text-white">
            <div className="text-xl font-black">민원 상담 접수하기</div>
            <div className="text-sm font-semibold opacity-90 mt-0.5">개인정보 입력 후 대기실로 이동합니다</div>
          </div>
        </button>
      </div>
    </div>
  )
}

// 2. 이름 입력
export function StepName({ data, setData, go }: StepProps) {
  return (
    <div className="h-full w-full flex flex-col bg-white text-slate-900 overflow-hidden relative">
      <Header step="name" />
      <div className="flex-1 flex flex-col items-center justify-center gap-6 px-10 py-6 overflow-y-auto">
        <h2 className="text-2xl font-black text-slate-800">이름을 입력해주세요</h2>
        <HangulKeyboard value={data.name} onChange={(name) => setData(p => ({ ...p, name }))} />
        <div className="flex gap-3 w-full max-w-[680px] mt-2">
          <button onClick={() => { setData(p => ({ ...p, name: '' })); go('start') }} className="flex-1 h-14 rounded-2xl text-sm font-bold bg-slate-100 border border-slate-200 text-slate-600">처음으로</button>
          <button onClick={() => go('dob')} disabled={!data.name.trim()} className="flex-[2] h-14 rounded-2xl text-sm font-black text-white bg-blue-600 disabled:opacity-30 shadow-sm">다음 단계 →</button>
        </div>
      </div>
    </div>
  )
}

// 3. 생년월일
export function StepDob({ data, setData, go }: StepProps) {
  const digits = data.dob.replace(/\D/g, ''); const isComplete = digits.length === 6; const isDateOk = isComplete ? isValidDate(digits) : false
  return (
    <div className="h-full w-full flex flex-col bg-white text-slate-900 overflow-hidden relative">
      <Header step="dob" />
      <div className="flex-1 flex flex-col items-center justify-center gap-6 px-10 py-6 overflow-y-auto">
        <div className="text-center h-16">
          <h2 className="text-2xl font-black text-slate-800 mb-1">생년월일 6자리</h2>
          {isComplete && !isDateOk ? <p className="text-red-500 text-sm font-bold animate-pulse">존재하지 않는 날짜입니다.</p> : <p className="text-slate-500 text-sm">예시 → 900101</p>}
        </div>
        <div className="flex items-center gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className={`w-14 h-16 rounded-xl flex shrink-0 items-center justify-center text-3xl font-black border-2 ${digits[i] ? (isComplete && !isDateOk ? 'border-red-400 bg-red-50 text-red-500' : 'border-blue-500 text-slate-900') : 'border-slate-200 text-slate-300'}`}>
              {digits[i] || (i === digits.length ? <span className="w-0.5 h-7 bg-blue-500 animate-pulse" /> : '')}
            </div>
          ))}
        </div>
        <Numpad onPress={(v) => handleNumpadLogic('dob', v, 6, 0, setData)} />
        <Nav onPrev={() => { setData(p => ({ ...p, dob: '' })); go('name') }} onNext={() => go('gender')} nextDisabled={!isDateOk} />
      </div>
    </div>
  )
}

// 4. 성별
export function StepGender({ data, setData, go }: StepProps) {
  return (
    <div className="h-full w-full flex flex-col bg-white text-slate-900 overflow-hidden relative">
      <Header step="gender" />
      <div className="flex-1 flex flex-col items-center justify-center gap-10 px-10 py-6 overflow-y-auto">
        <h2 className="text-2xl font-black text-slate-800">성별을 선택해주세요</h2>
        <div className="flex gap-6">
          {['남성', '여성'].map(v => (
            <button key={v} onClick={() => setData(p => ({ ...p, gender: v }))} className={`w-52 h-52 rounded-[32px] flex flex-col items-center justify-center gap-5 border-4 transition-all shrink-0 ${data.gender === v ? 'border-blue-600 bg-blue-50 text-blue-700' : 'border-slate-100 bg-slate-50 text-slate-400'}`}>
              <span className="text-2xl font-black">{v}</span>
            </button>
          ))}
        </div>
        <Nav onPrev={() => { setData(p => ({ ...p, gender: '' })); go('dob') }} onNext={() => go('phone')} nextDisabled={!data.gender} />
      </div>
    </div>
  )
}

// 5. 연락처
export function StepPhone({ data, setData, go }: StepProps) {
  return (
    <div className="h-full w-full flex flex-col bg-white text-slate-900 overflow-hidden relative">
      <Header step="phone" />
      <div className="flex-1 flex flex-col items-center justify-center gap-8 px-10 py-6 overflow-y-auto">
        <h2 className="text-2xl font-black text-slate-800">연락처를 입력해주세요</h2>
        <div className="flex items-center justify-center w-full max-w-[340px] h-[72px] rounded-2xl text-2xl font-black tracking-widest bg-slate-50 border border-slate-200">
          <span className={data.phone.length >= 11 ? 'text-slate-900' : 'text-slate-400'}>{formatPhone(data.phone)}</span>
          <span className="ml-1 w-[2px] h-8 bg-blue-500 animate-pulse" />
        </div>
        <Numpad onPress={(v) => handleNumpadLogic('phone', v, 11, 3, setData)} />
        <Nav onPrev={() => { setData(p => ({ ...p, phone: '010' })); go('gender') }} onNext={() => go('confirm')} nextDisabled={data.phone.replace(/\D/g, '').length < 11} />
      </div>
    </div>
  )
}

// 6. 확인
export function StepConfirm({ data, go, onFinish }: StepProps) {
  const maskedName = maskName(data.name)
  const maskedPhone = maskPhone(data.phone)

  return (
    <div className="h-full w-full flex flex-col bg-white text-slate-900 overflow-hidden relative">
      <Header step="confirm" />
      <div className="flex-1 flex flex-col items-center justify-center gap-8 px-10 py-6 overflow-y-auto">
        <h2 className="text-2xl font-black text-slate-800">입력하신 정보가 맞습니까?</h2>
        <div className="w-full max-w-[400px] flex flex-col gap-5 p-8 rounded-3xl bg-slate-50 border border-slate-200 shadow-sm">
          {[ ['이름', maskedName], ['생년월일', data.dob], ['성별', data.gender], ['연락처', maskedPhone] ].map(([l, v]) => (
            <div key={l} className="flex justify-between items-center border-b border-slate-200 pb-4 last:border-0 last:pb-0">
              <span className="text-sm font-bold text-slate-500">{l}</span>
              <span className={`text-xl font-black ${l === '연락처' ? 'text-blue-600' : 'text-slate-800'}`}>{v}</span>
            </div>
          ))}
        </div>
        <Nav onPrev={() => go('phone')} onNext={onFinish} nextLabel="맞습니다, 상담 시작하기" nextGreen />
      </div>
    </div>
  )
}

// 7. 대기
export function StepWaiting({ data }: StepProps) {
  return (
    <div className="h-full w-full flex flex-col items-center justify-center bg-white text-slate-900 overflow-hidden relative px-4">
      <div className="relative w-28 h-28 flex items-center justify-center mb-8">
        <div className="absolute inset-0 rounded-full border-4 border-slate-100 border-t-blue-500 animate-spin" />
        <svg className="w-10 h-10" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" strokeWidth="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
      </div>
      <h2 className="text-3xl font-black text-slate-800 mb-2">{maskName(data.name)}님, 환영합니다</h2>
      <p className="text-slate-500 text-sm font-medium">상담원이 곧 입장할 예정입니다</p>
      <div className="mt-8 px-5 py-2.5 rounded-full bg-emerald-50 border border-emerald-100 text-emerald-600 font-bold text-sm">상담원 대기실에 알림 전송됨</div>
    </div>
  )
}
