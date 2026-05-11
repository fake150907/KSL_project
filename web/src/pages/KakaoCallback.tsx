import { useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'

type Status = 'authenticating' | 'summarizing' | 'sending' | 'done' | 'error'

const STATUS_MESSAGE: Record<Status, string> = {
  authenticating: '카카오 인증을 처리하고 있습니다...',
  summarizing: '진료 내용을 AI로 요약하고 있습니다...',
  sending: '카카오톡으로 전송하고 있습니다...',
  done: '전송 완료! 잠시 후 화면이 이동됩니다.',
  error: '',
}

interface PatientData {
  name: string
  phone: string
}

export default function KakaoCallback() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const [status, setStatus] = useState<Status>('authenticating')
  const [errorMsg, setErrorMsg] = useState('')

  useEffect(() => {
    const code = searchParams.get('code')
    const error = searchParams.get('error')

    if (error) {
      setErrorMsg(`카카오 인증이 취소되었거나 실패했습니다: ${error}`)
      setStatus('error')
      return
    }
    if (!code) {
      setErrorMsg('카카오 인가 코드가 없습니다. OAuth URL을 다시 확인해주세요.')
      setStatus('error')
      return
    }

    const run = async () => {
      const tokenRes = await fetch('/api/kakao/token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, redirect_uri: `${window.location.origin}/kakao/callback` }),
      })
      const tokenData = await tokenRes.json().catch(() => ({}))
      if (!tokenRes.ok || !tokenData.access_token) {
        throw new Error(tokenData.error || '카카오 토큰 발급에 실패했습니다.')
      }

      const accessToken: string = tokenData.access_token
      const refreshToken: string = tokenData.refresh_token || ''
      localStorage.setItem('KAKAO_ACCESS_TOKEN', accessToken)
      if (refreshToken) localStorage.setItem('KAKAO_REFRESH_TOKEN', refreshToken)

      const returnPatientRaw = sessionStorage.getItem('kakao_return_patient')
      sessionStorage.removeItem('kakao_return_url')
      sessionStorage.removeItem('kakao_return_patient')

      if (!returnPatientRaw) {
        setStatus('done')
        setTimeout(() => navigate('/kiosk'), 2000)
        return
      }

      let patientData: PatientData | null = null
      try {
        patientData = JSON.parse(returnPatientRaw) as PatientData
      } catch {}

      setStatus('summarizing')
      const conversation: string[] = []

      try {
        const records = JSON.parse(localStorage.getItem('medical_records') || '[]')
        const patientDigits = (patientData?.phone || '').replace(/[^0-9]/g, '')
        const matched = Array.isArray(records)
          ? records.find((record: any) => {
              const recordDigits = String(record?.patientPhone || '').replace(/[^0-9]/g, '')
              return recordDigits && patientDigits && recordDigits === patientDigits
            }) || records[0]
          : null

        if (matched) {
          const notes = Array.isArray(matched.notes) ? matched.notes : []
          const noteLines = notes
            .map((note: any) => {
              const tag = String(note?.tag || '메모')
              const text = String(note?.text || '').trim()
              return text ? `- ${tag}: ${text}` : ''
            })
            .filter(Boolean)

          conversation.push(
            `[환자 정보] 이름: ${matched.patientName || patientData?.name || ''}, 연락처: ${matched.patientPhone || patientData?.phone || ''}`,
            noteLines.length > 0 ? `[의사 메모/처방]\n${noteLines.join('\n')}` : '[의사 메모/처방]\n- 기록 없음',
          )
        }
      } catch {}

      try {
        const messages = JSON.parse(localStorage.getItem('sign-lang-messages') || '[]')
        if (Array.isArray(messages)) {
          messages.forEach((message: any) => {
            if (!message?.text) return
            const speaker = message.sender === 'doctor' ? '의사' : '환자'
            conversation.push(`${speaker}: ${message.text}`)
          })
        }
      } catch {}

      if (conversation.length === 0) conversation.push('대화 기록 없음')

      let summaryText = conversation.join('\n')
      try {
        const summaryRes = await fetch('/api/summary', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify({ conversation }),
        })
        if (summaryRes.ok) {
          const summaryData = await summaryRes.json().catch(() => ({}))
          summaryText = summaryData.summary || summaryText
        }
      } catch {}

      setStatus('sending')
      const notifyRes = await fetch('/api/notify/kakao', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ access_token: accessToken, refresh_token: refreshToken, summary: summaryText }),
      })
      const notifyData = await notifyRes.json().catch(() => ({}))
      if (!notifyRes.ok) {
        throw new Error(notifyData.error || '카카오톡 전송에 실패했습니다.')
      }

      localStorage.removeItem('KAKAO_ACCESS_TOKEN')
      localStorage.removeItem('KAKAO_REFRESH_TOKEN')

      setStatus('done')
      setTimeout(() => navigate('/kiosk'), 2500)
    }

    run().catch((err) => {
      const msg = err instanceof Error ? err.message : String(err)
      setErrorMsg(msg && msg !== '[object Object]' ? msg : JSON.stringify(err) || '오류가 발생했습니다.')
      setStatus('error')
    })
  }, [searchParams, navigate])

  const isProcessing = status === 'authenticating' || status === 'summarizing' || status === 'sending'

  return (
    <div className="min-h-screen bg-slate-100 px-6 py-10 text-slate-900">
      <div className="mx-auto flex min-h-[70vh] max-w-2xl flex-col items-center justify-center rounded-3xl bg-white p-10 text-center shadow-xl">
        <div className={`mb-6 flex h-20 w-20 items-center justify-center rounded-full ${
          status === 'done' ? 'bg-emerald-500' : status === 'error' ? 'bg-red-100' : 'bg-[#FEE500]'
        }`}>
          {status === 'done' ? (
            <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3.5">
              <polyline points="20 6 9 17 4 12" />
            </svg>
          ) : status === 'error' ? (
            <span className="text-4xl font-black text-red-500">!</span>
          ) : (
            <span className="text-4xl font-black text-[#3C1E1E]">K</span>
          )}
        </div>

        <h1 className="mb-4 text-3xl font-black">카카오 인증</h1>

        {isProcessing && (
          <div className="mb-6 flex flex-col items-center gap-3">
            <div className="h-6 w-6 animate-spin rounded-full border-4 border-slate-200 border-t-yellow-400" />
            <p className="text-xl font-bold text-slate-600">{STATUS_MESSAGE[status]}</p>
          </div>
        )}

        {status === 'done' && (
          <p className="mb-8 text-xl font-bold leading-relaxed text-slate-600">{STATUS_MESSAGE.done}</p>
        )}

        {status === 'error' && (
          <>
            <p className="mb-8 text-xl font-bold leading-relaxed text-red-500">{errorMsg}</p>
            <button
              type="button"
              onClick={() => navigate('/kiosk')}
              className="h-14 rounded-2xl bg-blue-600 px-8 text-lg font-black text-white shadow-lg shadow-blue-100 transition hover:bg-blue-700"
            >
              환자 시작 화면으로 이동
            </button>
          </>
        )}
      </div>
    </div>
  )
}
