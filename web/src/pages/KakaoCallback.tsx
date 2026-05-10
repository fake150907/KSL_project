import { useEffect, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'

export default function KakaoCallback() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const [message, setMessage] = useState('카카오 인증을 처리하고 있습니다.')
  const [isDone, setIsDone] = useState(false)

  useEffect(() => {
    const code = searchParams.get('code')
    const error = searchParams.get('error')

    if (error) {
      setMessage(`카카오 인증이 취소되었거나 실패했습니다: ${error}`)
      setIsDone(true)
      return
    }

    if (!code) {
      setMessage('카카오 인가 코드가 없습니다. OAuth URL을 다시 확인해주세요.')
      setIsDone(true)
      return
    }

    const exchangeToken = async () => {
      try {
        const res = await fetch('/api/kakao/token', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            code,
            redirect_uri: `${window.location.origin}/kakao/callback`,
          }),
        })
        const data = await res.json().catch(() => ({}))

        if (!res.ok || !data.access_token) {
          throw new Error(data.error || '카카오 access token 발급에 실패했습니다.')
        }

        localStorage.setItem('KAKAO_ACCESS_TOKEN', data.access_token)
        if (data.refresh_token) {
          localStorage.setItem('KAKAO_REFRESH_TOKEN', data.refresh_token)
        }
        setMessage('카카오 access token을 저장했습니다. 이제 카카오톡으로 받기를 다시 눌러주세요.')
      } catch (err) {
        setMessage(err instanceof Error ? err.message : '카카오 access token 저장에 실패했습니다.')
      } finally {
        setIsDone(true)
      }
    }

    void exchangeToken()
  }, [searchParams])

  return (
    <div className="min-h-screen bg-slate-100 px-6 py-10 text-slate-900">
      <div className="mx-auto flex min-h-[70vh] max-w-2xl flex-col items-center justify-center rounded-3xl bg-white p-10 text-center shadow-xl">
        <div className="mb-6 flex h-20 w-20 items-center justify-center rounded-full bg-[#FEE500]">
          <span className="text-4xl font-black text-[#3C1E1E]">K</span>
        </div>
        <h1 className="mb-4 text-3xl font-black">카카오 인증</h1>
        <p className="mb-8 text-xl font-bold leading-relaxed text-slate-600">{message}</p>
        <button
          type="button"
          disabled={!isDone}
          onClick={() => navigate('/kiosk')}
          className="h-14 rounded-2xl bg-blue-600 px-8 text-lg font-black text-white shadow-lg shadow-blue-100 transition hover:bg-blue-700 disabled:bg-slate-200 disabled:text-slate-400"
        >
          환자 시작 화면으로 이동
        </button>
      </div>
    </div>
  )
}
