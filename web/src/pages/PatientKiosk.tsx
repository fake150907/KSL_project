import { useRef, useEffect, useState, useCallback } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import type { ChatMessage as ChatMessageType } from '../types'
import VideoFeed from '../components/VideoFeed'
import ChatMessage from '../components/ChatMessage'
import { useSignLanguage, validationDemoScenarios, type DemoScenario } from '../hooks/useSignLanguage'
import { socket, registerRole } from '../socket'

interface PatientKioskProps {
  messages: ChatMessageType[]
  onNewMessage: (msg: ChatMessageType) => void
  onSessionReset?: () => void
  roomLabel?: string
  sessionEnded?: boolean
  patientName?: string 
  patientPhone?: string
}

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
  
  const [isWaitingForDoctor, setIsWaitingForDoctor] = useState(sessionEnded)
  const [sendStatus, setSendStatus] = useState<'idle' | 'sending' | 'sent' | 'error'>('idle')
  const [sendError, setSendError] = useState('')
  const [showPopup, setShowPopup] = useState(false)
  const [showDemoList, setShowDemoList] = useState(false)
  const [predictionLog, setPredictionLog] = useState<Array<{ label: string; confidence: number; timestamp: number }>>([])

  // ✅ 수정됨: 수어 인식 메시지를 socket으로도 전송하는 래퍼
  const handleNewMessageFromKiosk = useCallback((msg: ChatMessageType) => {
    onNewMessage(msg)                    // 자기 화면 업데이트
    socket.emit('chat_message', msg)     // ✅ 의사에게 전송
  }, [onNewMessage])

  const {
    videoRef, canvasRef, landmarkCanvasRef,
    isRunning, isDemoMode, activeDemoLabel, activeDemoClipLabel, currentPrediction,
    videoDevices, selectedDeviceId, setSelectedDeviceId,
    startCamera, stopCamera, startDemoScenario, handleDemoVideoEnded, getPredictionStatus,
  } = useSignLanguage(handleNewMessageFromKiosk)

  // 키오스크 역할 등록
  useEffect(() => {
    registerRole('kiosk')
  }, [])

  // ✅ 수정됨: 의사가 보낸 채팅 메시지 수신
  useEffect(() => {
    const handleIncomingMessage = (msg: ChatMessageType) => {
      onNewMessage(msg)
    }
    socket.on('chat_message', handleIncomingMessage)
    return () => {
      socket.off('chat_message', handleIncomingMessage)
    }
  }, [onNewMessage])

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
    const handleSessionEnd = () => {
      stopCamera()
      setIsWaitingForDoctor(false)
      setShowPopup(true)
      setSendStatus('idle')
      setSendError('')
    }

    socket.on('session_end', handleSessionEnd)
    return () => {
      socket.off('session_end', handleSessionEnd)
    }
  }, [stopCamera])

  // WebRTC
  useEffect(() => {
    if (!isRunning || !videoRef.current?.srcObject) return;

    let iceQueue: RTCIceCandidateInit[] = [];
    let isRemoteDescriptionSet = false;

    const pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
    peerConnectionRef.current = pc;
    const stream = videoRef.current.srcObject as MediaStream;
    stream.getTracks().forEach((track) => {
      const sender = pc.addTrack(track, stream);
      if (track.kind === 'video') {
        const params = sender.getParameters();
        params.encodings = params.encodings?.length ? params.encodings : [{}];
        params.encodings[0].maxBitrate = 700_000;
        params.encodings[0].maxFramerate = 20;
        void sender.setParameters(params).catch((err) => {
          console.warn('WebRTC video parameter update failed:', err);
        });
      }
    });

    pc.onicecandidate = (event) => { 
      if (event.candidate) {
        socket.emit('webrtc_ice_candidate', { target: 'doctor', candidate: event.candidate }); 
      }
    };
    
    const handleAnswer = async (data: any) => { 
      if (pc.signalingState !== 'closed') {
        try {
          await pc.setRemoteDescription(new RTCSessionDescription(data.answer));
          isRemoteDescriptionSet = true;
          
          while(iceQueue.length > 0) {
             const candidate = iceQueue.shift();
             if (candidate) await pc.addIceCandidate(new RTCIceCandidate(candidate));
          }
        } catch (err) {
          console.error('Answer 처리 중 에러:', err);
        }
      }
    };
    
    const handleCandidate = async (data: any) => { 
      if (data.candidate && pc.signalingState !== 'closed') {
        try {
          if (isRemoteDescriptionSet) {
             await pc.addIceCandidate(new RTCIceCandidate(data.candidate));
          } else {
             iceQueue.push(data.candidate);
          }
        } catch (err) {
          console.error('ICE 처리 중 에러:', err);
        }
      }
    };
    
    socket.on('webrtc_answer', handleAnswer);
    socket.on('webrtc_ice_candidate', handleCandidate);
    
    const createOffer = async () => {
      try {
        const offer = await pc.createOffer();
        await pc.setLocalDescription(offer);
        socket.emit('webrtc_offer', { target: 'doctor', offer });
      } catch (err) {
        console.error('Offer 생성 에러:', err);
      }
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

  const formatPhone = (val: string) => {
    const digits = val.replace(/[^0-9]/g, '').slice(0, 11)
    if (digits.length <= 3) return digits
    if (digits.length <= 7) return `${digits.slice(0, 3)}-${digits.slice(3)}`
    return `${digits.slice(0, 3)}-${digits.slice(3, 7)}-${digits.slice(7)}`
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
      if (!res.ok) throw new Error(data.error || '카카오톡 전송에 실패했습니다.')
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

const handleClosePopup = () => { 
    setShowPopup(false)
    onSessionReset?.()
    navigate('/kiosk') 
  }
  const handleDemoSelect = (scenario: DemoScenario) => {
    setShowDemoList(false)
    void startDemoScenario(scenario)
  }

  return (
    <div className="fixed inset-0 bg-slate-100 flex flex-col overflow-hidden">
      
      {isWaitingForDoctor && (
        <div className="absolute inset-0 z-[100] flex flex-col items-center justify-center bg-white/95 backdrop-blur-md px-4 text-center">
          <div className="relative w-20 h-20 md:w-28 md:h-28 mb-6 md:mb-8">
            <div className="absolute inset-0 border-4 md:border-[6px] border-slate-100 rounded-full"></div>
            <div className="absolute inset-0 border-4 md:border-[6px] border-blue-500 rounded-full border-t-transparent animate-spin"></div>
            <div className="absolute inset-0 flex items-center justify-center">
              <svg className="w-8 h-8 md:w-10 md:h-10" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" strokeWidth="2.5">
                <path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83"/>
              </svg>
            </div>
          </div>
          <h2 className="text-2xl md:text-3xl font-black text-slate-800 mb-4 tracking-tight">진료실을 준비하고 있습니다</h2>
          <p className="text-sm md:text-lg text-slate-500 font-medium">의사 선생님이 이전 진료를 정리 중입니다. 잠시만 기다려주세요.</p>
        </div>
      )}

      <header className="flex-shrink-0 flex items-center justify-between px-6 py-3 border-b border-slate-200 bg-white z-10 shadow-sm">
        <div className="flex flex-col gap-0.5">
          <span className="text-xl font-black tracking-tight text-slate-900">수어 통역 키오스크</span>
          <span className="text-sm text-blue-600 font-bold bg-blue-50 px-2 py-0.5 rounded-md inline-block w-fit">{roomLabel} - {actualPatientName}님</span>
        </div>
        <div className="flex flex-col items-end gap-0.5">
          <span className="text-2xl font-black tabular-nums text-slate-800">{clock}</span>
          <span className={`flex items-center gap-1.5 text-xs font-bold ${isRunning ? 'text-emerald-600' : 'text-slate-400'}`}>
            <span className={`rounded-full shrink-0 w-2 h-2 ${isRunning ? 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.4)]' : 'bg-slate-300'}`} />
            {isRunning ? '카메라 활성' : '카메라 대기'}
          </span>
        </div>
      </header>

      <div className="flex flex-row flex-1 min-h-0 overflow-hidden">
        <div className="w-[480px] shrink-0 flex flex-col border-r border-slate-200 bg-white overflow-hidden">
          <div className="flex-1 min-h-0 p-3 bg-slate-50 flex justify-center items-center">
            <div className="w-full h-full rounded-2xl overflow-hidden shadow-inner border-4 border-white bg-black relative">
              <VideoFeed videoRef={videoRef} canvasRef={canvasRef} landmarkCanvasRef={landmarkCanvasRef} isRunning={isRunning} currentPrediction={currentPrediction} predictionStatus={predictionStatus} onVideoEnded={handleDemoVideoEnded} />
            </div>
          </div>

          <div className={`flex shrink-0 items-center px-4 py-2.5 gap-3 border-t border-slate-100 transition-colors duration-300 ${isRunning ? isRecognized ? 'bg-emerald-50' : 'bg-blue-50' : 'bg-white'}`}>
            {isRunning ? (
              <>
                <div className={`w-9 h-9 rounded-lg flex shrink-0 items-center justify-center ${isRecognized ? 'bg-emerald-100' : 'bg-blue-100'}`}>
                  <span className={`rounded-full shrink-0 w-2.5 h-2.5 ${isRecognized ? 'bg-emerald-500' : 'bg-blue-500'}`} />
                </div>
                <div className="flex flex-col min-w-0">
                  <span className={`text-xs font-bold ${isRecognized ? 'text-emerald-600' : 'text-blue-600'}`}>{isRecognized ? '수어 인식 성공' : '실시간 인식 중'}</span>
                  <span className="text-sm font-black text-slate-800 leading-tight truncate">{bannerLabel || '손을 카메라에 맞춰 보여주세요'}</span>
                </div>
                <div className="ml-auto hidden min-w-0 flex-col items-end gap-1 xl:flex">
                  <span className="text-[10px] font-black text-slate-400">최근 예측</span>
                  <div className="flex max-w-[180px] flex-wrap justify-end gap-1">
                    {predictionLog.length > 0 ? predictionLog.map((item) => (
                      <span key={`${item.timestamp}-${item.label}`} className="rounded-md border border-white/70 bg-white px-1.5 py-0.5 text-[10px] font-black text-slate-700 shadow-sm">
                        {item.label}
                        <span className="ml-1 text-blue-500">{Math.round(item.confidence * 100)}%</span>
                      </span>
                    )) : (
                      <span className="rounded-md bg-white/70 px-1.5 py-0.5 text-[10px] font-bold text-slate-400">없음</span>
                    )}
                  </div>
                </div>
              </>
            ) : (
              <span className="text-xs text-slate-400 font-bold mx-auto italic text-center">카메라 시작 버튼을 누르면 수어 번역이 시작됩니다</span>
            )}
          </div>

          <div className="flex shrink-0 flex-col gap-2 px-4 py-3 border-t border-slate-100 bg-white">
            <div className="flex gap-2">
              <button
                onClick={isRunning ? stopCamera : startCamera}
                disabled={sessionEnded}
                className={`flex-1 py-2.5 text-sm font-black rounded-lg border-2 transition-all active:scale-95 shadow-sm ${
                  sessionEnded ? 'bg-slate-50 border-slate-200 text-slate-300 cursor-not-allowed' : isRunning ? 'bg-red-50 border-red-200 text-red-600 hover:bg-red-100' : 'bg-blue-600 border-blue-700 text-white hover:bg-blue-700 shadow-blue-100'
                }`}
              >
                {sessionEnded ? '진료 종료됨' : isRunning ? '카메라 중지' : '카메라 시작'}
              </button>

              <div className="relative flex items-stretch gap-2 flex-1">
                <button
                  onClick={() => validationDemoScenarios[0] && handleDemoSelect(validationDemoScenarios[0])}
                  disabled={sessionEnded}
                  className={`flex-1 flex items-center justify-center py-2.5 text-sm font-black rounded-lg border-2 transition-all active:scale-95 shadow-sm ${
                    sessionEnded ? 'bg-slate-50 border-slate-200 text-slate-300 cursor-not-allowed' : isDemoMode ? 'bg-emerald-50 border-emerald-200 text-emerald-700' : 'bg-slate-900 border-slate-950 text-white hover:bg-slate-800'
                  }`}
                >
                  {isDemoMode ? `데모: ${activeDemoClipLabel || activeDemoLabel}` : '데모 시연'}
                </button>
                <button
                  type="button"
                  onClick={() => setShowDemoList((prev) => !prev)}
                  disabled={sessionEnded}
                  className="flex w-10 items-center justify-center rounded-lg border-2 border-slate-200 bg-white text-slate-700 shadow-sm hover:bg-slate-50 active:scale-95 disabled:text-slate-300"
                >
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"><path d="m6 9 6 6 6-6" /></svg>
                </button>

                {showDemoList && (
                  <div className="absolute left-0 right-0 bottom-full mb-1 z-50 overflow-hidden rounded-xl border-2 border-slate-200 bg-white shadow-2xl">
                    {validationDemoScenarios.map((scenario, index) => (
                      <button
                        key={scenario.label}
                        onClick={() => handleDemoSelect(scenario)}
                        className="flex w-full items-center gap-2 border-b border-slate-100 px-3 py-2.5 text-left text-sm font-black text-slate-800 last:border-b-0 hover:bg-blue-50"
                      >
                        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-blue-600 text-white text-[10px]">{index + 1}</span>
                        <span className="truncate">{scenario.label}</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {videoDevices.length > 1 && (
              <select value={selectedDeviceId} onChange={(e) => setSelectedDeviceId(e.target.value)} className="w-full py-2 px-3 text-sm font-bold bg-slate-50 border-2 border-slate-200 rounded-lg text-slate-700 outline-none focus:border-blue-500">
                {videoDevices.map((d, i) => <option key={d.deviceId} value={d.deviceId}>{d.label || `카메라 ${i + 1}`}</option>)}
              </select>
            )}
          </div>
        </div>

        <div className="flex-1 flex flex-col min-w-0 min-h-0 bg-slate-50/30">
          <div className="shrink-0 flex items-center justify-between px-6 py-3 border-b border-slate-100 bg-white shadow-sm z-10">
            <div className="flex items-center gap-2">
              <svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="#64748b" strokeWidth="2.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
              <span className="text-base font-black text-slate-700">실시간 대화 내역</span>
            </div>
            <span className="text-xs font-bold text-blue-500 bg-blue-50 px-2 py-0.5 rounded-full">{messages.length}</span>
          </div>
          <div className="flex-1 overflow-y-auto flex flex-col gap-3 px-6 py-4">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full gap-3 text-center opacity-40">
                <div className="w-14 h-14 rounded-full bg-slate-100 flex items-center justify-center">
                  <svg className="w-7 h-7" viewBox="0 0 24 24" fill="none" stroke="#475569" strokeWidth="1.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                </div>
                <div>
                  <p className="text-base font-black text-slate-600">대화 내역이 없습니다</p>
                  <p className="text-xs text-slate-500 mt-1 font-medium">진료가 시작되면 대화 내용이 기록됩니다</p>
                </div>
              </div>
            ) : (
              messages.map((msg) => <ChatMessage key={msg.id} message={msg} />)
            )}
            <div ref={chatEndRef} />
          </div>
          <footer className="shrink-0 py-2.5 px-6 border-t border-slate-100 bg-white flex items-center justify-center">
            <p className="text-xs text-slate-400 font-bold text-center">수어로 증상을 표현해 주세요. 인공지능이 즉시 의사 선생님께 전달합니다.</p>
          </footer>
        </div>
      </div>

      {showPopup && (
        <div className="absolute inset-0 flex items-center justify-center z-50 bg-slate-900/60 backdrop-blur-sm p-4">
          <div className="w-full max-w-sm bg-white border border-slate-100 rounded-3xl px-6 pt-6 pb-6 flex flex-col items-center shadow-2xl relative">
            <div className="w-14 h-14 rounded-full bg-[#FEE500] flex items-center justify-center mb-3 shrink-0 shadow-lg shadow-yellow-100">
              <svg className="w-7 h-7" viewBox="0 0 24 24" fill="#3C1E1E"><path d="M12 3C6.477 3 2 6.477 2 10.8c0 2.8 1.718 5.253 4.286 6.72l-.857 3.2 3.715-2.457C10.012 18.41 10.99 18.6 12 18.6c5.523 0 10-3.477 10-7.8S17.523 3 12 3z"/></svg>
            </div>
            <h2 className="text-2xl font-black text-slate-900 text-center leading-tight mb-1">진료를 마쳤습니다</h2>
            <p className="text-sm text-slate-500 text-center font-medium">등록하신 휴대전화로<br/>진료 대화 내역을 보내드릴까요?</p>
            
            <div className="flex flex-col items-center justify-center rounded-xl mt-4 mb-5 w-full py-3 bg-slate-50 border-2 border-slate-100 shadow-inner">
              <span className="text-xs text-slate-400 font-bold mb-1">{actualPatientName}님의 연락처</span>
              <div className="text-xl font-black text-slate-800 tracking-wider">{formatPhone(actualPatientPhone)}</div>
            </div>
            
            <div className="flex flex-col w-full gap-2">
              <button onClick={handleSendKakaoSummary} disabled={sendStatus === 'sending' || actualPatientPhone.length < 10} className={`w-full py-3 rounded-xl text-sm md:text-base font-black flex items-center justify-center gap-2 transition-all ${sendStatus === 'sending' || actualPatientPhone.length < 10 ? 'bg-slate-100 text-slate-300' : 'bg-[#FEE500] text-[#3C1E1E] hover:brightness-105 active:scale-[0.98] shadow-md shadow-yellow-100'}`}>
                {sendStatus === 'sending' ? '전송하는 중...' : '카카오톡으로 받기'}
              </button>
              {sendStatus === 'error' && (
                <p className="rounded-lg bg-red-50 px-3 py-2 text-center text-[10px] md:text-xs font-bold leading-snug text-red-600">
                  {sendError || '설정이 필요합니다. 내용은 클립보드에 복사했습니다.'}
                </p>
              )}
              <button onClick={handleClosePopup} className="py-2 text-slate-400 text-xs md:text-sm font-bold hover:text-slate-600 transition-colors">아니요, 괜찮습니다</button>
            </div>

            {sendStatus === 'sent' && (
              <div className="absolute inset-0 bg-white rounded-3xl flex flex-col items-center justify-center p-6 animate-in fade-in duration-300">
                <div className="w-14 h-14 bg-emerald-500 rounded-full flex items-center justify-center mb-4 shadow-md shadow-emerald-100">
                  <svg className="w-7 h-7" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3.5"><polyline points="20 6 9 17 4 12"/></svg>
                </div>
                <p className="text-xl font-black text-slate-900 mb-2">전송 완료!</p>
                <p className="text-xs text-slate-500 text-center leading-relaxed font-medium">메시지를 성공적으로 보냈습니다.<br/>진료실을 나가셔도 좋습니다.</p>
                <button onClick={handleClosePopup} className="mt-6 w-full py-3 rounded-xl text-sm font-black bg-blue-600 text-white hover:bg-blue-700 shadow-md shadow-blue-100 transition-all active:scale-95">확인 (메인으로)</button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
