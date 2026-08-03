import { useState } from 'react'
import HangulKeyboard from './HangulKeyboard'
import Numpad from './Numpad'
import SignLanguageLogo from './SignLanguageLogo'
import { Header, Nav } from './KioskUI'
import { formatPhone, maskName, maskPhone, containsSensitiveInfo } from './hangul'
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
          onClick={() => go('consent')}
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

// 1-b. 개인정보 동의
export function StepConsent({ go }: Pick<StepProps, 'go'>) {
  const [checked, setChecked] = useState(false)

  return (
    <div className="h-full w-full flex flex-col items-center justify-center bg-slate-100 px-4 py-8 overflow-y-auto">
      <div className="w-full max-w-[680px] bg-white border border-slate-200 rounded-3xl shadow-xl p-8 flex flex-col gap-5">
        {/* 헤더 */}
        <div>
          <span className="inline-block px-3 py-1.5 rounded-full bg-slate-100 text-slate-500 text-sm font-bold mb-4">Pierna Privacy Notice</span>
          <h1 className="text-3xl font-black tracking-tight leading-snug">피어나(Pierna)<br />서비스 이용 동의</h1>
          <p className="mt-3 text-base text-slate-600 leading-relaxed">
            피어나는 주민센터 창구에서 수어 통역 및 상담 지원 서비스를 제공하기 위해
            필요한 최소한의 개인정보와 외부 AI 서비스를 이용합니다.
          </p>
        </div>

        {/* 주요 안내 */}
        <div className="bg-slate-50 border border-slate-200 rounded-2xl px-5 py-4">
          <strong className="block text-base mb-3">주요 안내</strong>
          <ul className="list-disc pl-5 space-y-2 text-[15px] text-slate-700 leading-snug">
            <li>이름과 전화번호만 수집합니다.</li>
            <li>원본 수어 영상과 JPEG 이미지 프레임은 저장하지 않습니다.</li>
            <li>수어 영상은 실시간 랜드마크(관절 좌표) 추출 후 즉시 삭제됩니다.</li>
            <li>수어 인식 결과의 자연어 변환 및 상담 요약을 위해 <strong>Anthropic Claude (Messages API)</strong>를 이용할 수 있습니다.</li>
            <li>상담 요약 및 안내 발송을 위해 카카오 알림톡을 이용할 수 있습니다.</li>
          </ul>
        </div>

        {/* 안내 박스 */}
        <div className="border border-slate-300 rounded-2xl px-4 py-3 text-[15px] text-slate-600 leading-relaxed">
          AI는 통역 및 상담 요약 기능만 수행하며, 행정 처리·판단·의사결정 권한은 담당 공무원에게 있습니다.
          자세한 개인정보 처리 기준은 개인정보처리방침에서 확인할 수 있습니다.
        </div>

        {/* 개인정보처리방침 링크 버튼 */}
        <button
          type="button"
          onClick={() => go('policy')}
          className="flex items-center justify-center gap-2 w-full py-3 border border-slate-200 rounded-2xl bg-white text-slate-800 text-[15px] font-bold hover:bg-slate-50 active:scale-[0.98] transition-transform"
        >
          <svg className="w-4 h-4 text-slate-500" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4zm2 6a1 1 0 011-1h6a1 1 0 110 2H7a1 1 0 01-1-1zm1 3a1 1 0 000 2h6a1 1 0 100-2H7z" clipRule="evenodd"/></svg>
          개인정보처리방침 보기
        </button>

        {/* 동의 체크 */}
        <div
          className="border-2 border-slate-900 rounded-2xl p-4 flex items-start gap-3 cursor-pointer select-none"
          onClick={() => setChecked(v => !v)}
        >
          <div className={`w-7 h-7 rounded-lg border-2 border-slate-900 flex-shrink-0 flex items-center justify-center transition-colors ${checked ? 'bg-slate-900' : 'bg-white'}`}>
            {checked && <svg className="w-4 h-4 text-white" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414L8.414 15l-5.121-5.121a1 1 0 011.414-1.414L8.414 12.172l6.879-6.879a1 1 0 011.414 0z" clipRule="evenodd"/></svg>}
          </div>
          <span className="text-[17px] font-bold leading-snug">위 내용을 확인하였으며, 피어나 서비스 이용에 동의합니다.</span>
        </div>

        {/* 버튼 */}
        <div className="grid grid-cols-2 gap-3">
          <button
            type="button"
            onClick={() => go('start')}
            className="py-4 rounded-2xl text-base font-bold bg-slate-200 text-slate-800 active:scale-[0.97] transition-transform"
          >
            동의하지 않음
          </button>
          <button
            type="button"
            disabled={!checked}
            onClick={() => go('name')}
            className="py-4 rounded-2xl text-base font-bold bg-slate-900 text-white disabled:opacity-30 active:scale-[0.97] transition-transform"
          >
            동의하고 시작하기
          </button>
        </div>

        {/* 푸터 */}
        <p className="text-xs text-slate-400 leading-relaxed">
          서비스명: 피어나(Pierna) · 운영 주체: 오동가영 팀<br />
          본 화면은 태블릿 창구 상담 시작 전 이용자 동의를 위한 안내 화면입니다.
        </p>
      </div>
    </div>
  )
}

// 1-c. 개인정보처리방침
export function StepPrivacyPolicy({ go }: Pick<StepProps, 'go'>) {
  return (
    <div className="h-full w-full flex flex-col bg-slate-100 overflow-hidden">
      {/* 상단 바 */}
      <div className="flex-shrink-0 flex items-center gap-3 px-5 py-4 bg-white border-b border-slate-200">
        <button
          type="button"
          onClick={() => go('consent')}
          className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-100 text-slate-700 text-sm font-bold hover:bg-slate-200 active:scale-[0.97] transition-transform"
        >
          <svg className="w-4 h-4" viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M9.707 16.707a1 1 0 01-1.414 0l-6-6a1 1 0 010-1.414l6-6a1 1 0 011.414 1.414L5.414 9H17a1 1 0 110 2H5.414l4.293 4.293a1 1 0 010 1.414z" clipRule="evenodd"/></svg>
          동의 화면으로 돌아가기
        </button>
        <span className="text-sm font-bold text-slate-500">개인정보처리방침</span>
      </div>

      {/* 본문 스크롤 영역 */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="w-full max-w-[900px] mx-auto bg-white border border-slate-200 rounded-3xl shadow-sm overflow-hidden">
          {/* 헤더 */}
          <div className="px-8 pt-10 pb-8 border-b border-slate-200 bg-gradient-to-b from-white to-slate-50">
            <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-100 text-slate-500 text-sm font-bold mb-4">Pierna Privacy Policy</span>
            <h1 className="text-4xl font-black tracking-tight leading-tight">피어나(Pierna)<br />개인정보처리방침</h1>
            <p className="mt-4 text-slate-600 text-[17px] max-w-2xl">
              오동가영 팀은 피어나(Pierna) 서비스 이용자의 개인정보를 중요하게 생각하며,
              개인정보 보호 관련 법령을 준수하기 위해 다음과 같이 개인정보처리방침을 안내합니다.
            </p>
            <div className="grid grid-cols-3 gap-3 mt-6">
              {[['서비스명','피어나(Pierna)'],['운영 주체','오동가영 팀'],['시행일','2026년 ○월 ○일']].map(([k,v])=>(
                <div key={k} className="border border-slate-200 rounded-2xl bg-white p-4">
                  <strong className="block text-xs text-slate-400 mb-1">{k}</strong>
                  <span className="text-base font-bold">{v}</span>
                </div>
              ))}
            </div>
          </div>

          {/* 본문 섹션들 */}
          <div className="px-8 py-6 space-y-0 divide-y divide-slate-100">
            {/* 요약 안내 */}
            <div className="py-5">
              <div className="border border-slate-200 rounded-2xl bg-slate-50 px-5 py-4 text-slate-600 text-[15px] leading-relaxed">
                피어나는 이름과 전화번호만 최소한으로 수집합니다. 수어 영상은 저장하지 않으며,
                실시간 랜드마크(관절 좌표) 추출 후 즉시 삭제합니다.
              </div>
            </div>

            {/* 제1조 */}
            <div className="py-6">
              <h2 className="text-xl font-black mb-3">제1조 개인정보의 처리 목적</h2>
              <p className="text-[15px] text-slate-600 mb-2">오동가영 팀은 다음 목적을 위해 개인정보를 처리합니다.</p>
              <ol className="list-decimal pl-5 space-y-1 text-[15px] text-slate-700">
                <li>수어 통역 및 민원 상담 지원</li>
                <li>상담 요약 및 안내 메시지 발송</li>
                <li>민원 상담 이력 관리</li>
                <li>서비스 품질 개선</li>
                <li>이용자 문의 및 민원 처리</li>
              </ol>
              <p className="mt-2 text-[15px] text-slate-600">수집된 개인정보는 위 목적 외의 용도로 이용되지 않습니다.</p>
            </div>

            {/* 제2조 */}
            <div className="py-6">
              <h2 className="text-xl font-black mb-3">제2조 수집하는 개인정보 항목</h2>
              <div className="overflow-x-auto border border-slate-200 rounded-2xl">
                <table className="w-full text-[14px]">
                  <tbody>
                    {[
                      ['필수 항목','이름, 전화번호'],
                      ['자동 처리 정보','서비스 이용 기록, 접속 일시, 시스템 로그'],
                      ['수집하지 않는 정보','주민등록번호, 장애등록번호, 복지카드 번호, 건강정보 등 민감정보'],
                    ].map(([k,v],i,arr)=>(
                      <tr key={k} className={i < arr.length-1 ? 'border-b border-slate-100' : ''}>
                        <th className="w-1/4 px-4 py-3 bg-slate-50 font-bold text-left whitespace-nowrap">{k}</th>
                        <td className="px-4 py-3 text-slate-700">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* 제3조 */}
            <div className="py-6">
              <h2 className="text-xl font-black mb-3">제3조 수어 영상 처리</h2>
              <p className="text-[15px] text-slate-600 mb-3">피어나는 수어 인식을 위해 카메라 영상을 실시간으로 처리할 수 있습니다.</p>
              <div className="grid grid-cols-2 gap-3">
                {[
                  ['원본 영상 미저장','수어 영상은 실시간 좌표 추출에만 사용되며, 서버 또는 단말기에 저장하지 않습니다.'],
                  ['즉시 삭제','수어 영상은 랜드마크(관절 좌표) 추출 후 즉시 삭제됩니다.'],
                ].map(([t,d])=>(
                  <div key={t} className="border border-slate-200 rounded-2xl bg-white p-4">
                    <div className="font-bold mb-2 text-[15px]">{t}</div>
                    <p className="text-[14px] text-slate-600">{d}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* 제4조 */}
            <div className="py-6">
              <h2 className="text-xl font-black mb-3">제4조 외부 AI 서비스 이용</h2>
              <p className="text-[15px] text-slate-600 mb-3">
                피어나는 수어 인식 결과를 자연스러운 한국어 문장으로 변환하고 상담 내용을 요약하기 위해
                외부 AI 서비스인 <strong>Anthropic Claude(Messages API)</strong>를 활용할 수 있습니다.
              </p>
              <div className="overflow-x-auto border border-slate-200 rounded-2xl mb-3">
                <table className="w-full text-[14px]">
                  <tbody>
                    {[
                      ['이용 서비스','Anthropic Claude (Messages API)'],
                      ['이용 목적','수어 인식 결과(GLOSS)의 자연어 변환, 상담 내용 요약 생성, 의사소통 지원 품질 향상'],
                      ['전송 정보','수어 인식 결과(GLOSS), 상담 텍스트'],
                      ['전송하지 않는 정보','원본 수어 영상, 주민등록번호, 장애등록번호, 복지카드 번호, 기타 민감정보'],
                      ['보호 조치','이름 및 전화번호는 필요한 경우 마스킹 또는 분리 처리 후 전송합니다.'],
                    ].map(([k,v],i,arr)=>(
                      <tr key={k} className={i < arr.length-1 ? 'border-b border-slate-100' : ''}>
                        <th className="w-1/4 px-4 py-3 bg-slate-50 font-bold text-left whitespace-nowrap">{k}</th>
                        <td className="px-4 py-3 text-slate-700">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="border border-slate-200 rounded-2xl px-4 py-3 bg-slate-50 font-bold text-[14px] text-slate-700">
                AI는 통역 및 요약 기능만 수행하며, 행정 처리 및 의사결정 권한은 담당 공무원에게 있습니다.
              </div>
            </div>

            {/* 제5조 */}
            <div className="py-6">
              <h2 className="text-xl font-black mb-3">제5조 개인정보의 보유 및 이용 기간</h2>
              <div className="overflow-x-auto border border-slate-200 rounded-2xl">
                <table className="w-full text-[14px]">
                  <tbody>
                    {[
                      ['이름·전화번호','상담 목적 달성 후 최대 30일 보관 후 파기'],
                      ['수어 영상','저장하지 않음'],
                      ['랜드마크 좌표','실시간 처리 후 삭제'],
                      ['상담 요약 데이터','운영 목적 범위 내 최소 기간 보관 후 파기'],
                    ].map(([k,v],i,arr)=>(
                      <tr key={k} className={i < arr.length-1 ? 'border-b border-slate-100' : ''}>
                        <th className="w-1/4 px-4 py-3 bg-slate-50 font-bold text-left whitespace-nowrap">{k}</th>
                        <td className="px-4 py-3 text-slate-700">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="mt-3 text-[14px] text-slate-600">이용자가 삭제를 요청하는 경우 지체 없이 파기합니다. 법령에 따라 보관이 필요한 경우에는 해당 기간 동안 보관할 수 있습니다.</p>
            </div>

            {/* 제6조 */}
            <div className="py-6">
              <h2 className="text-xl font-black mb-3">제6조 개인정보의 제3자 제공</h2>
              <p className="text-[15px] text-slate-600 mb-3">오동가영 팀은 원칙적으로 개인정보를 제3자에게 제공하지 않습니다. 다만 다음의 경우 제공될 수 있습니다.</p>
              <div className="overflow-x-auto border border-slate-200 rounded-2xl">
                <table className="w-full text-[14px]">
                  <tbody>
                    {[
                      ['제공받는 자','카카오 또는 알림톡 발송 대행사'],
                      ['제공 항목','전화번호'],
                      ['제공 목적','상담 요약 및 안내 메시지 발송'],
                      ['보유 기간','발송 목적 달성 시까지'],
                    ].map(([k,v],i,arr)=>(
                      <tr key={k} className={i < arr.length-1 ? 'border-b border-slate-100' : ''}>
                        <th className="w-1/4 px-4 py-3 bg-slate-50 font-bold text-left whitespace-nowrap">{k}</th>
                        <td className="px-4 py-3 text-slate-700">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* 제7조 */}
            <div className="py-6">
              <h2 className="text-xl font-black mb-3">제7조 개인정보 처리의 위탁</h2>
              <p className="text-[15px] text-slate-600">
                서비스 운영 과정에서 일부 업무를 외부 전문 업체에 위탁할 수 있습니다.
                위탁이 발생하는 경우 관련 법령에 따라 계약을 체결하고 개인정보 보호조치를 적용합니다.
              </p>
            </div>

            {/* 제8조 */}
            <div className="py-6">
              <h2 className="text-xl font-black mb-3">제8조 개인정보의 파기 절차 및 방법</h2>
              <ul className="list-disc pl-5 space-y-1.5 text-[15px] text-slate-700">
                <li>처리 목적이 달성된 개인정보는 지체 없이 파기합니다.</li>
                <li>전자 파일은 복구 또는 재생되지 않도록 안전하게 삭제합니다.</li>
                <li>출력물은 분쇄 또는 소각 방식으로 파기합니다.</li>
              </ul>
            </div>

            {/* 제9조 */}
            <div className="py-6">
              <h2 className="text-xl font-black mb-3">제9조 안전성 확보 조치</h2>
              <div className="grid grid-cols-2 gap-3">
                {[
                  ['보안','AES-256 암호화 저장'],
                  ['접근','접근 권한 최소화'],
                  ['화면','개인정보 마스킹 처리'],
                  ['차단','민감정보 입력 차단'],
                  ['로그','보안 로그 관리'],
                  ['영상','수어 영상 미저장 정책 적용'],
                ].map(([badge,text])=>(
                  <div key={badge} className="border border-slate-200 rounded-2xl bg-white px-4 py-3 flex items-center gap-2 text-[14px]">
                    <span className="inline-block px-2 py-0.5 rounded-full bg-slate-100 text-slate-600 text-xs font-bold">{badge}</span>
                    {text}
                  </div>
                ))}
              </div>
            </div>

            {/* 제10조 */}
            <div className="py-6">
              <h2 className="text-xl font-black mb-3">제10조 정보주체의 권리</h2>
              <p className="text-[15px] text-slate-600 mb-2">이용자는 언제든지 다음 권리를 행사할 수 있습니다.</p>
              <ul className="list-disc pl-5 space-y-1.5 text-[15px] text-slate-700">
                <li>개인정보 열람 요청</li>
                <li>개인정보 정정 요청</li>
                <li>개인정보 삭제 요청</li>
                <li>개인정보 처리정지 요청</li>
              </ul>
            </div>

            {/* 제11조 */}
            <div className="py-6">
              <h2 className="text-xl font-black mb-3">제11조 개인정보 보호책임자</h2>
              <div className="overflow-x-auto border border-slate-200 rounded-2xl">
                <table className="w-full text-[14px]">
                  <tbody>
                    {[
                      ['운영 주체','오동가영 팀'],
                      ['서비스명','피어나(Pierna)'],
                      ['문의처','추후 운영 연락처 기재'],
                    ].map(([k,v],i,arr)=>(
                      <tr key={k} className={i < arr.length-1 ? 'border-b border-slate-100' : ''}>
                        <th className="w-1/4 px-4 py-3 bg-slate-50 font-bold text-left whitespace-nowrap">{k}</th>
                        <td className="px-4 py-3 text-slate-700">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* 제12조 */}
            <div className="py-6">
              <h2 className="text-xl font-black mb-3">제12조 개인정보처리방침의 변경</h2>
              <p className="text-[15px] text-slate-600">
                본 개인정보처리방침은 법령, 서비스 정책 또는 운영상 필요에 따라 변경될 수 있습니다.
                변경 시 서비스 화면 또는 홈페이지를 통해 안내합니다.
              </p>
            </div>
          </div>

          {/* 푸터 */}
          <div className="px-8 py-5 bg-slate-50 border-t border-slate-200 text-slate-400 text-xs leading-relaxed">
            본 문서는 피어나(Pierna) 서비스의 개인정보 처리 기준을 안내하기 위한 초안입니다.
            실제 서비스 운영 전 법률 및 개인정보보호 전문가 검토를 권장합니다.
          </div>
        </div>

        {/* 하단 돌아가기 버튼 */}
        <div className="mt-4 flex justify-center">
          <button
            type="button"
            onClick={() => go('consent')}
            className="px-8 py-4 rounded-2xl bg-slate-900 text-white font-bold text-base active:scale-[0.97] transition-transform"
          >
            동의 화면으로 돌아가기
          </button>
        </div>
      </div>
    </div>
  )
}

// 2. 이름 입력
export function StepName({ data, setData, go }: StepProps) {
  const handleNameChange = (name: string) => {
    if (containsSensitiveInfo(name)) {
      alert('민감정보(주민번호 등)는 입력할 수 없습니다.')
      setData(p => ({ ...p, name: '*'.repeat(name.length) }))
      return
    }
    setData(p => ({ ...p, name }))
  }
  return (
    <div className="h-full w-full flex flex-col bg-white text-slate-900 overflow-hidden relative">
      <Header step="name" />
      <div className="flex-1 flex flex-col items-center justify-center gap-6 px-10 py-6 overflow-y-auto">
        <h2 className="text-2xl font-black text-slate-800">이름을 입력해주세요</h2>
        <HangulKeyboard value={data.name} onChange={handleNameChange} />
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
