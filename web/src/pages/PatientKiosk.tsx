import { useRef, useEffect, useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import type { ChatMessage as ChatMessageType } from '../types'
import VideoFeed from '../components/VideoFeed'
import ChatMessage from '../components/ChatMessage'
import { useSignLanguage, validationDemoScenarios, type DemoScenario } from '../hooks/useSignLanguage'
import { socket } from '../socket'

interface PatientKioskProps {
  messages: ChatMessageType[]
  onNewMessage: (msg: ChatMessageType) => void
  onSessionReset?: () => void
  roomLabel?: string
  sessionEnded?: boolean
  patientName?: string 
  patientPhone?: string
}

const KIOSK_W = 1080
const KIOSK_H = 1920

export default function PatientKiosk({
  messages,
  onNewMessage,
  onSessionReset,
  roomLabel = '진료실 3번',
  sessionEnded = false,
  patientName = '환자',
  patientPhone = '01000000000',
}: PatientKioskProps) {
  const navigate = useNavigate()
  const location = useLocation()
  const navState = location.state as { patientData?: { name: string, phone: string } } | null
  
  const actualPatientName = navState?.patientData?.name || patientName
  const actualPatientPhone = navState?.patientData?.phone || patientPhone

  const chatEndRef = useRef<HTMLDivElement>(null)
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null)
  
  const [scale, setScale] = useState(1)
  const [isWaitingForDoctor, setIsWaitingForDoctor] = useState(sessionEnded)
  const [sendStatus, setSendStatus] = useState<'idle' | 'sending' | 'sent' | 'error'>('idle')
  const [sendError, setSendError] = useState('')
  const [showPopup, setShowPopup] = useState(false)
  const [showDemoList, setShowDemoList] = useState(false)
  const [predictionLog, setPredictionLog] = useState<Array<{ label: string; confidence: number; timestamp: number }>>([])

  const {
    videoRef, canvasRef, landmarkCanvasRef,
    isRunning, isDemoMode, activeDemoLabel, activeDemoClipLabel, currentPrediction,
    videoDevices, selectedDeviceId, setSelectedDeviceId,
    startCamera, stopCamera, startDemoScenario, handleDemoVideoEnded, getPredictionStatus,
  } = useSignLanguage(onNewMessage)

  useEffect(() => {
    const recalc = () => setScale(Math.min(window.innerWidth / KIOSK_W, window.innerHeight / KIOSK_H))
    recalc()
    window.addEventListener('resize', recalc)
    return () => window.removeEventListener('resize', recalc)
  }, [])

  useEffect(() => {
    const parent = chatEndRef.current?.parentElement;
    if (parent) parent.scrollTo({ top: parent.scrollHeight, behavior: 'smooth' });
  }, [messages])

  useEffect(() => {
    if (!sessionEnded) setIsWaitingForDoctor(false)
  }, [sessionEnded])

  useEffect(() => {
    if (sessionEnded && !isWaitingForDoctor) {
      stopCamera()
      setShowPopup(true)
      setSendStatus('idle')
    }
  }, [sessionEnded, isWaitingForDoctor, stopCamera])

  useEffect(() => {
    if (!isRunning || !videoRef.current?.srcObject) return;
    const pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
    peerConnectionRef.current = pc;
    const stream = videoRef.current.srcObject as MediaStream;
    stream.getTracks().forEach((track) => pc.addTrack(track, stream));
    pc.onicecandidate = (event) => { if (event.candidate) socket.emit('webrtc_ice_candidate', { target: 'doctor', candidate: event.candidate }); };
    const handleAnswer = async (data: any) => { if (pc.signalingState !== 'closed') await pc.setRemoteDescription(new RTCSessionDescription(data.answer)); };
    const handleCandidate = async (data: any) => { if (data.candidate && pc.signalingState !== 'closed') await pc.addIceCandidate(new RTCIceCandidate(data.candidate)); };
    socket.on('webrtc_answer', handleAnswer);
    socket.on('webrtc_ice_candidate', handleCandidate);
    const createOffer = async () => {
      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      socket.emit('webrtc_offer', { target: 'doctor', offer });
    };
    createOffer();
    return () => {
      pc.close();
      socket.off('webrtc_answer', handleAnswer);
      socket.off('webrtc_ice_candidate', handleCandidate);
    };
  }, [isRunning, videoRef]);

  const [clock, setClock] = useState(() => new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' }))
  useEffect(() => {
    const id = setInterval(() => setClock(new Date().toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })), 10000)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    if (!currentPrediction?.window_filled) return
    const top = currentPrediction.top_predictions?.[0]
    const label = currentPrediction.label || top?.label
    const confidence = currentPrediction.confidence || top?.confidence || 0
    if (!label) return

    setPredictionLog((prev) => {
      const last = prev[0]
      if (last && last.label === label && Math.abs(last.confidence - confidence) < 0.001) return prev
      return [{ label, confidence, timestamp: Date.now() }, ...prev].slice(0, 4)
    })
  }, [currentPrediction?.timestamp, currentPrediction?.window_filled, currentPrediction?.label, currentPrediction?.confidence, currentPrediction?.top_predictions])

  const predictionStatus = currentPrediction ? getPredictionStatus(currentPrediction) : ''
  const isRecognized = !!currentPrediction?.window_filled && !!currentPrediction?.label && (currentPrediction?.confidence ?? 0) >= 0.30
  const bannerLabel = isRecognized ? predictionStatus.replace('인식 중... ', '') : predictionStatus

  const buildChatText = () => {
    if (messages.length === 0) return '대화 내역이 없습니다.'
    return messages.map((m) => `[${m.timestamp.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })}] ${m.sender === 'doctor' ? '의사' : '환자'}: ${m.text}`).join('\n')
  }

  const getLatestMedicalRecordText = () => {
    try {
      const records = JSON.parse(localStorage.getItem('medical_records') || '[]')
      if (!Array.isArray(records)) return ''
      const patientDigits = actualPatientPhone.replace(/[^0-9]/g, '')
      const matched = records.find((record: any) => {
        const recordDigits = String(record?.patientPhone || '').replace(/[^0-9]/g, '')
        return recordDigits && patientDigits && recordDigits === patientDigits
      }) || records[0]
      if (!matched) return ''

      const notes = Array.isArray(matched.notes) ? matched.notes : []
      const noteLines = notes
        .map((note: any) => {
          const tag = String(note?.tag || '메모')
          const text = String(note?.text || '').trim()
          return text ? `- ${tag}: ${text}` : ''
        })
        .filter(Boolean)

      return [
        `[환자 정보] 이름: ${matched.patientName || actualPatientName}, 연락처: ${matched.patientPhone || actualPatientPhone}`,
        noteLines.length > 0 ? `[의사 메모/처방]\n${noteLines.join('\n')}` : '[의사 메모/처방]\n- 기록 없음',
      ].join('\n')
    } catch {
      return ''
    }
  }

  const buildClinicalSummaryInput = () => {
    const conversation = messages.map((m) => {
      const speaker = m.sender === 'doctor' ? '의사' : '환자'
      return `${speaker}: ${m.text}`
    })
    const recordText = getLatestMedicalRecordText()
    if (recordText) conversation.unshift(recordText)
    if (conversation.length === 0) conversation.push('대화 기록 없음')
    return conversation
  }

  const formatPhone = (val: string) => {
    const digits = val.replace(/[^0-9]/g, '').slice(0, 11)
    if (digits.length <= 3) return digits
    if (digits.length <= 7) return `${digits.slice(0, 3)}-${digits.slice(3)}`
    return `${digits.slice(0, 3)}-${digits.slice(3, 7)}-${digits.slice(7)}`
  }

  const handleSendKakao = async () => {
    const cleaned = actualPatientPhone.replace(/[^0-9]/g, '')
    if (cleaned.length < 10) return
    setSendStatus('sending')
    const chatText = buildChatText()
    try {
      const res = await fetch('/api/send-kakao', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ phone: cleaned, message: chatText }) })
      if (res.ok) { setSendStatus('sent'); return; }
    } catch {}
    try {
      await navigator.clipboard.writeText(`📋 진료 대화 내역\n전화번호: ${cleaned}\n\n${chatText}`)
      setSendStatus('sent')
    } catch { setSendStatus('sent') }
  }

  const handleSendKakaoSafe = async () => {
    const cleaned = actualPatientPhone.replace(/[^0-9]/g, '')
    if (cleaned.length < 10) return
    setSendStatus('sending')
    setSendError('')
    const chatText = buildChatText()
    const accessToken = localStorage.getItem('KAKAO_ACCESS_TOKEN') || ''

    try {
      if (!accessToken) {
        throw new Error('카카오 access_token이 없어 실제 카카오톡 전송은 실행되지 않았습니다.')
      }
      const res = await fetch('/api/notify/kakao', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ access_token: accessToken, summary: chatText }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        throw new Error(data.error || '카카오톡 전송에 실패했습니다.')
      }
      setSendStatus('sent')
      window.setTimeout(() => navigate('/kiosk'), 1200)
      return
    } catch (err) {
      setSendError(err instanceof Error ? err.message : '카카오톡 전송에 실패했습니다.')
    }

    try {
      await navigator.clipboard.writeText(`[수어 진료 대화 내역]\n전화번호: ${cleaned}\n\n${chatText}`)
    } catch {}
    setSendStatus('error')
  }

  const handleSendKakaoSummary = async () => {
    const cleaned = actualPatientPhone.replace(/[^0-9]/g, '')
    if (cleaned.length < 10) return
    setSendStatus('sending')
    setSendError('')

    const conversation = messages.map((m) => `${m.sender === 'doctor' ? '의사' : '환자'}: ${m.text}`)
    const chatText = buildChatText()
    const accessToken = localStorage.getItem('KAKAO_ACCESS_TOKEN') || ''
    let summaryText = chatText

    try {
      if (conversation.length > 0) {
        const summaryRes = await fetch('/api/summary', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify({ conversation }),
        })
        if (summaryRes.ok) {
          const summaryData = await summaryRes.json().catch(() => ({}))
          summaryText = summaryData.summary || chatText
        }
      }

      const res = await fetch('/api/notify/kakao', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ access_token: accessToken, summary: summaryText }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        throw new Error(data.error || '카카오톡 전송에 실패했습니다.')
      }
      setSendStatus('sent')
      return
    } catch (err) {
      setSendError(err instanceof Error ? err.message : '카카오톡 전송에 실패했습니다.')
    }

    try {
      await navigator.clipboard.writeText(`[수어 진료 요약본]\n전화번호: ${cleaned}\n\n${summaryText}`)
    } catch {}
    setSendStatus('error')
  }

  const handleSendKakaoSummaryV2 = async () => {
    const cleaned = actualPatientPhone.replace(/[^0-9]/g, '')
    if (cleaned.length < 10) return
    setSendStatus('sending')
    setSendError('')

    const conversation = buildClinicalSummaryInput()
    const chatText = buildChatText()
    const accessToken = localStorage.getItem('KAKAO_ACCESS_TOKEN') || ''
    const refreshToken = localStorage.getItem('KAKAO_REFRESH_TOKEN') || ''
    let summaryText = chatText

    try {
      const summaryRes = await fetch('/api/summary', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ conversation }),
      })
      if (summaryRes.ok) {
        const summaryData = await summaryRes.json().catch(() => ({}))
        summaryText = summaryData.summary || chatText
      }

      const res = await fetch('/api/notify/kakao', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ access_token: accessToken, refresh_token: refreshToken, summary: summaryText }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        throw new Error(data.error || '카카오톡 전송에 실패했습니다.')
      }
      if (data.access_token) localStorage.setItem('KAKAO_ACCESS_TOKEN', data.access_token)
      if (data.refresh_token) localStorage.setItem('KAKAO_REFRESH_TOKEN', data.refresh_token)
      setSendStatus('sent')
      window.setTimeout(() => navigate('/kiosk'), 1200)
      return
    } catch (err) {
      setSendError(err instanceof Error ? err.message : '카카오톡 전송에 실패했습니다.')
    }

    try {
      await navigator.clipboard.writeText(`[수어 진료 요약본]\n전화번호: ${cleaned}\n\n${summaryText}`)
    } catch {}
    setSendStatus('error')
  }

  const handleClosePopup = () => { setShowPopup(false); navigate('/kiosk') }
  const handleDemoSelect = (scenario: DemoScenario) => {
    setShowDemoList(false)
    void startDemoScenario(scenario)
  }

  return (
    <div className="fixed inset-0 bg-slate-100 overflow-hidden flex items-center justify-center">
      <div
        className="flex flex-col bg-white text-slate-900 flex-shrink-0 relative overflow-visible shadow-[0_20px_80px_rgba(0,0,0,0.1)]"
        style={{ width: KIOSK_W, height: KIOSK_H, transform: `scale(${scale})`, transformOrigin: 'center center' }}
      >
        {/* 의사 대기 화면 (화이트) */}
        {isWaitingForDoctor && (
          <div className="absolute inset-0 z-[100] flex flex-col items-center justify-center bg-white/95 backdrop-blur-md">
            <div className="relative w-32 h-32 mb-10">
              <div className="absolute inset-0 border-[6px] border-slate-100 rounded-full"></div>
              <div className="absolute inset-0 border-[6px] border-blue-500 rounded-full border-t-transparent animate-spin"></div>
              <div className="absolute inset-0 flex items-center justify-center">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" strokeWidth="2.5">
                  <path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83"/>
                </svg>
              </div>
            </div>
            <h2 className="text-[48px] font-black text-slate-800 mb-6 tracking-tight">진료실을 준비하고 있습니다</h2>
            <p className="text-[28px] text-slate-500 font-medium">의사 선생님이 이전 진료를 정리 중입니다. 잠시만 기다려주세요.</p>
          </div>
        )}

        {/* 헤더 */}
        <header className="flex-shrink-0 flex items-center justify-between px-10 h-[160px] border-b border-slate-100 bg-white">
          <div className="flex items-center gap-5">
            <div className="flex flex-col gap-1">
              <span className="text-[32px] font-black tracking-tight text-slate-900">수어 통역 키오스크</span>
              <span className="text-[24px] text-blue-600 font-bold bg-blue-50 px-3 py-1 rounded-lg inline-block">{roomLabel} - {actualPatientName}님</span>
            </div>
          </div>
          <div className="flex flex-col items-end gap-1">
            <span className="text-[42px] font-black tabular-nums text-slate-800">{clock}</span>
            <span className={`flex items-center gap-2 text-[20px] font-bold ${isRunning ? 'text-emerald-600' : 'text-slate-400'}`}>
              <span className={`rounded-full shrink-0 w-3 h-3 ${isRunning ? 'bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.4)]' : 'bg-slate-300'}`} />
              {isRunning ? '카메라 활성화됨' : '카메라 대기 중'}
            </span>
          </div>
        </header>

        {/* 비디오 피드 영역 */}
        <div className="flex-shrink-0 bg-slate-50 h-[810px] border-b border-slate-100 p-6">
          <div className="w-full h-full rounded-[40px] overflow-hidden shadow-inner border-4 border-white bg-black">
            <VideoFeed videoRef={videoRef} canvasRef={canvasRef} landmarkCanvasRef={landmarkCanvasRef} isRunning={isRunning} currentPrediction={currentPrediction} predictionStatus={predictionStatus} onVideoEnded={handleDemoVideoEnded} />
          </div>
        </div>

        {/* 인식 상태 바 */}
        <div className={`flex-shrink-0 flex items-center px-10 gap-5 h-[120px] border-b border-slate-100 transition-colors duration-300 ${isRunning ? isRecognized ? 'bg-emerald-50' : 'bg-blue-50' : 'bg-white'}`}>
          {isRunning ? (
            <>
              <div className={`w-14 h-14 rounded-2xl flex items-center justify-center ${isRecognized ? 'bg-emerald-100' : 'bg-blue-100'}`}>
                <span className={`rounded-full shrink-0 w-4 h-4 ${isRecognized ? 'bg-emerald-500' : 'bg-blue-500'}`} />
              </div>
              <div className="flex flex-col">
                <span className={`text-[20px] font-bold ${isRecognized ? 'text-emerald-600' : 'text-blue-600'}`}>{isRecognized ? '수어 인식 성공' : '실시간 인식 중'}</span>
                <span className="text-[34px] font-black text-slate-800 leading-tight">{bannerLabel || '손을 카메라에 맞춰 보여주세요'}</span>
              </div>
              <div className="ml-auto flex max-w-[430px] flex-col items-end gap-2">
                <span className="text-[18px] font-black text-slate-400">최근 예측 단어</span>
                <div className="flex flex-wrap justify-end gap-2">
                  {predictionLog.length > 0 ? predictionLog.map((item) => (
                    <span key={`${item.timestamp}-${item.label}`} className="rounded-2xl bg-white px-4 py-2 text-[20px] font-black text-slate-800 shadow-sm border border-slate-100">
                      {item.label}
                      <span className="ml-2 text-[16px] text-blue-500">{Math.round(item.confidence * 100)}%</span>
                    </span>
                  )) : (
                    <span className="rounded-2xl bg-white/70 px-4 py-2 text-[20px] font-bold text-slate-400">아직 없음</span>
                  )}
                </div>
              </div>
            </>
          ) : (
            <span className="text-[28px] text-slate-400 font-bold mx-auto italic">카메라 시작 버튼을 누르면 수어 번역이 시작됩니다</span>
          )}
        </div>

        {/* 컨트롤 버튼 */}
        <div className="relative flex-shrink-0 flex flex-col justify-center gap-4 px-10 h-[260px] border-b border-slate-100">
          <div className="flex items-center gap-5">
            <button
              onClick={isRunning ? stopCamera : startCamera}
              disabled={sessionEnded}
              className={`flex flex-1 items-center justify-center gap-4 h-[82px] text-[28px] font-black rounded-3xl border-2 transition-all active:scale-95 shadow-sm ${
                sessionEnded ? 'bg-slate-50 border-slate-200 text-slate-300 cursor-not-allowed' : isRunning ? 'bg-red-50 border-red-200 text-red-600 hover:bg-red-100' : 'bg-blue-600 border-blue-700 text-white hover:bg-blue-700 shadow-blue-100'
              }`}
            >
              {sessionEnded ? '진료가 종료되었습니다' : isRunning ? '카메라 중지하기' : '카메라 시작하기'}
            </button>
            {videoDevices.length > 1 && (
              <select value={selectedDeviceId} onChange={(e) => setSelectedDeviceId(e.target.value)} className="h-[82px] min-w-[320px] px-8 text-[24px] font-bold bg-slate-50 border-2 border-slate-200 rounded-3xl text-slate-700 outline-none focus:border-blue-500 transition-colors">
                {videoDevices.map((d, i) => <option key={d.deviceId} value={d.deviceId}>{d.label || `카메라 ${i + 1}`}</option>)}
              </select>
            )}
          </div>
          <div className="relative flex items-center gap-3">
            <button
              onClick={() => validationDemoScenarios[0] && handleDemoSelect(validationDemoScenarios[0])}
              disabled={sessionEnded}
              className={`flex flex-1 items-center justify-center h-[82px] px-8 text-[28px] font-black rounded-3xl border-2 transition-all active:scale-95 shadow-sm ${
                sessionEnded ? 'bg-slate-50 border-slate-200 text-slate-300 cursor-not-allowed' : isDemoMode ? 'bg-emerald-50 border-emerald-200 text-emerald-700' : 'bg-slate-900 border-slate-950 text-white hover:bg-slate-800'
              }`}
            >
              {isDemoMode ? `데모 시연 중: ${activeDemoClipLabel || activeDemoLabel}` : '데모영상 시연'}
            </button>
            <button
              type="button"
              onClick={() => setShowDemoList((prev) => !prev)}
              disabled={sessionEnded}
              aria-label="데모 목록 열기"
              className="flex h-[82px] w-[96px] items-center justify-center rounded-3xl border-2 border-slate-200 bg-white text-slate-700 shadow-sm transition-all hover:bg-slate-50 active:scale-95 disabled:text-slate-300"
            >
              <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3">
                <path d="m6 9 6 6 6-6" />
              </svg>
            </button>
            {showDemoList && (
              <div className="absolute left-0 right-0 top-[92px] z-50 overflow-hidden rounded-3xl border-2 border-slate-200 bg-white shadow-[0_24px_70px_rgba(15,23,42,0.22)]">
                {validationDemoScenarios.map((scenario, index) => (
                  <button
                    key={scenario.label}
                    type="button"
                    onClick={() => handleDemoSelect(scenario)}
                    className="flex w-full items-center gap-5 border-b border-slate-100 px-8 py-6 text-left text-[24px] font-black text-slate-800 last:border-b-0 hover:bg-blue-50"
                  >
                    <span className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-blue-600 text-[22px] text-white">{index + 1}</span>
                    <span>{scenario.label}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* 대화 내역 */}
        <div className="flex-1 flex flex-col min-h-0 overflow-hidden bg-slate-50/30">
          <div className="shrink-0 flex items-center justify-between px-10 py-6 border-b border-slate-100 bg-white">
            <div className="flex items-center gap-3">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#64748b" strokeWidth="2.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
              <span className="text-[26px] font-black text-slate-700">실시간 대화 내역</span>
            </div>
            <span className="text-[22px] font-bold text-blue-500 bg-blue-50 px-4 py-1 rounded-full">{messages.length}</span>
          </div>
          <div className="flex-1 overflow-y-auto flex flex-col gap-6 px-10 py-8">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full gap-8 text-center opacity-40">
                <div className="w-[140px] h-[140px] rounded-full bg-slate-100 flex items-center justify-center">
                  <svg width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="#475569" strokeWidth="1.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                </div>
                <div>
                  <p className="text-[32px] font-black text-slate-600">대화 내역이 없습니다</p>
                  <p className="text-[24px] text-slate-500 mt-3 font-medium">진료가 시작되면 대화 내용이 여기에 기록됩니다</p>
                </div>
              </div>
            ) : (
              messages.map((msg) => <ChatMessage key={msg.id} message={msg} />)
            )}
            <div ref={chatEndRef} />
          </div>
        </div>

        <footer className="shrink-0 h-[100px] flex items-center justify-center border-t border-slate-100 bg-white shadow-[0_-10px_40px_rgba(0,0,0,0.02)]">
          <p className="text-[24px] text-slate-400 font-bold">수어로 증상을 표현해 주세요. 인공지능이 즉시 의사 선생님께 전달합니다.</p>
        </footer>

        {/* 진료 종료 팝업 (화이트) */}
        {showPopup && (
          <div className="absolute inset-0 flex items-center justify-center z-50 bg-slate-900/60 backdrop-blur-md">
            <div className="w-[860px] bg-white border border-slate-100 rounded-[60px] px-16 pt-20 pb-16 flex flex-col items-center shadow-[0_40px_100px_rgba(0,0,0,0.2)]">
              <div className="w-[130px] h-[130px] rounded-full bg-[#FEE500] flex items-center justify-center mb-8 shrink-0 shadow-xl shadow-yellow-100">
                <svg width="64" height="64" viewBox="0 0 24 24" fill="#3C1E1E"><path d="M12 3C6.477 3 2 6.477 2 10.8c0 2.8 1.718 5.253 4.286 6.72l-.857 3.2 3.715-2.457C10.012 18.41 10.99 18.6 12 18.6c5.523 0 10-3.477 10-7.8S17.523 3 12 3z"/></svg>
              </div>
              <h2 className="text-[52px] font-black text-slate-900 text-center leading-tight">진료를 마쳤습니다</h2>
              <p className="text-[30px] text-slate-500 mt-4 text-center font-medium">등록하신 휴대전화로 진료 대화 내역을 보내드릴까요?</p>
              
              <div className="flex flex-col items-center justify-center rounded-[40px] mt-14 mb-10 w-full py-12 bg-slate-50 border-2 border-slate-100 shadow-inner">
                <span className="text-[26px] text-slate-400 font-bold mb-4">{actualPatientName}님의 연락처</span>
                <div className="text-[64px] font-black text-slate-800 tracking-[6px]">{formatPhone(actualPatientPhone)}</div>
              </div>
              
              <div className="flex flex-col w-full gap-4">
                <button onClick={handleSendKakaoSummaryV2} disabled={sendStatus === 'sending' || actualPatientPhone.length < 10} className={`w-full h-[110px] rounded-[32px] text-[38px] font-black flex items-center justify-center gap-5 transition-all ${sendStatus === 'sending' || actualPatientPhone.length < 10 ? 'bg-slate-100 text-slate-300' : 'bg-[#FEE500] text-[#3C1E1E] hover:brightness-105 active:scale-[0.98] shadow-lg shadow-yellow-100'}`}>
                  {sendStatus === 'sending' ? '전송하는 중...' : '카카오톡으로 받기'}
                </button>
                {sendStatus === 'error' && (
                  <p className="rounded-3xl bg-red-50 px-8 py-5 text-center text-[22px] font-bold leading-snug text-red-600">
                    {sendError || '카카오톡 전송 설정이 필요합니다. 대화 내용은 클립보드에 복사했습니다.'}
                  </p>
                )}
                <button onClick={handleClosePopup} className="h-[90px] text-slate-400 text-[28px] font-bold hover:text-slate-600 transition-colors">아니요, 괜찮습니다</button>
              </div>

              {sendStatus === 'sent' && (
                <div className="absolute inset-0 bg-white rounded-[60px] flex flex-col items-center justify-center p-16 animate-in fade-in duration-500">
                  <div className="w-[160px] h-[140px] bg-emerald-500 rounded-full flex items-center justify-center mb-10 shadow-lg shadow-emerald-100">
                    <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3.5"><polyline points="20 6 9 17 4 12"/></svg>
                  </div>
                  <p className="text-[54px] font-black text-slate-900 mb-4">전송 완료!</p>
                  <p className="text-[30px] text-slate-500 text-center leading-relaxed font-medium">입력하신 번호로 카카오톡 메시지를 보내드렸습니다.<br/>진료실을 나가셔도 좋습니다.</p>
                  <button onClick={handleClosePopup} className="mt-16 w-full h-[110px] rounded-[32px] text-[38px] font-black bg-blue-600 text-white hover:bg-blue-700 shadow-xl shadow-blue-100 transition-all active:scale-95">확인 (메인으로)</button>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
