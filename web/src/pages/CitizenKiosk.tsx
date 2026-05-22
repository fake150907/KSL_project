import { useRef, useEffect, useState, useCallback } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import type { ChatMessage as ChatMessageType } from '../types'
import VideoFeed from '../components/VideoFeed'
import ChatMessage from '../components/ChatMessage'
import { useSignLanguage, validationDemoScenarios, type DemoScenario } from '../hooks/useSignLanguage'
import { useWelfarePanel } from '../hooks/useWelfarePanel'
import { WelfarePanel } from '../components/WelfarePanel'
import { socket, registerRole } from '../socket'

interface CitizenKioskProps {
  messages: ChatMessageType[]
  onNewMessage: (msg: ChatMessageType) => void
  onSessionReset?: () => void
  roomLabel?: string
  sessionEnded?: boolean
  citizenName?: string 
  citizenPhone?: string
}

export default function CitizenKiosk({
  messages,
  onNewMessage,
  onSessionReset,
  roomLabel = '상담실 3번',
  sessionEnded = false,
  citizenName = '민원인',
  citizenPhone = '01000000000',
}: CitizenKioskProps) {
  const navigate = useNavigate()
  const location = useLocation()
  const navState = location.state as { citizenData?: { name: string, phone: string } } | null
  
  const actualCitizenName = navState?.citizenData?.name || citizenName
  const actualCitizenPhone = navState?.citizenData?.phone || citizenPhone

  const chatEndRef = useRef<HTMLDivElement>(null)
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null)
  
  const [isWaitingForAgent, setIsWaitingForAgent] = useState(sessionEnded)
  const [sendStatus, setSendStatus] = useState<'idle' | 'sending' | 'sent' | 'error'>('idle')
  const [sendError, setSendError] = useState('')
  const [showPopup, setShowPopup] = useState(false)
  const [cachedSummary, setCachedSummary] = useState('')
  const [showDemoList, setShowDemoList] = useState(false)
  const [predictionLog, setPredictionLog] = useState<Array<{ label: string; confidence: number; timestamp: number }>>([])

  const handleNewMessageFromKiosk = useCallback((msg: ChatMessageType) => {
    onNewMessage(msg)                    // 자기 화면 업데이트
    socket.emit('chat_message', msg)     // ✅ 상담원에게 전송
  }, [onNewMessage])

  const {
    videoRef, canvasRef, landmarkCanvasRef,
    isRunning, isDemoMode, activeDemoLabel, activeDemoClipLabel, currentPrediction,
    lastLookupKey,
    videoDevices, selectedDeviceId, setSelectedDeviceId,
    startCamera, stopCamera, startDemoScenario, handleDemoVideoEnded, getPredictionStatus,
    camFps, sendFps,
    liveSegmentStatus,
  } = useSignLanguage(handleNewMessageFromKiosk)

  const { welfarePanel, dismiss: dismissWelfarePanel } = useWelfarePanel(
    currentPrediction?.scenario?.lookup_key || lastLookupKey
  )

  // 키오스크 역할 등록
  useEffect(() => {
    registerRole('kiosk')
  }, [])

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
    if (!sessionEnded) setIsWaitingForAgent(false)
  }, [sessionEnded])

  useEffect(() => {
    if (sessionEnded && !isWaitingForAgent) {
      stopCamera()
      setShowPopup(true)
      setSendStatus('idle')
    }
  }, [sessionEnded, isWaitingForAgent, stopCamera])

  useEffect(() => {
    const handleSessionEnd = () => {
      stopCamera()
      setIsWaitingForAgent(false)
      setSendStatus('idle')
      setSendError('')

      // ✅ 요약 생성 후 소켓으로 상담원 쪽에 전달 (카카오 전송 여부와 무관)
      const saveSummary = async () => {
        let summaryText = buildChatText()
        try {
          const res = await fetch('/api/summary', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ conversation: buildConsultationSummaryInput() }),
          })
          const data = await res.json().catch(() => ({}))
          
          if (res.ok && data.summary) {
             summaryText = data.summary
          } else {
             // API 응답이 정상이 아닐 경우 콘솔에 에러 출력
             console.error("요약 API 호출 실패:", data.error || res.statusText)
          }
        } catch (error) { 
          // 네트워크 오류 등 예외 발생 시 콘솔에 출력 후 폴백 텍스트 사용
          console.error("요약 요청 중 네트워크 예외 발생:", error)
        }

        setCachedSummary(summaryText)

        socket.emit('consultation_summary_saved', {
          citizenName: actualCitizenName,
          citizenPhone: actualCitizenPhone,
          consultationSummary: summaryText,
          isSent: false,
          deliveryStatus: 'pending',
        })

        setShowPopup(true)
      }
      void saveSummary()
    }

    socket.on('session_end', handleSessionEnd)
    return () => {
      socket.off('session_end', handleSessionEnd)
    }
  }, [stopCamera, actualCitizenPhone, actualCitizenName, messages])

  // WebRTC
  useEffect(() => {
    if (!isRunning || !videoRef.current?.srcObject) return;

    let iceQueue: RTCIceCandidateInit[] = [];
    let isRemoteDescriptionSet = false;

    const pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
    peerConnectionRef.current = pc;
    const stream = videoRef.current.srcObject as MediaStream;
    stream.getTracks().forEach((track) => pc.addTrack(track, stream));

    pc.onicecandidate = (event) => { 
      if (event.candidate) {
        socket.emit('webrtc_ice_candidate', { target: 'agent', candidate: event.candidate }); 
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
        socket.emit('webrtc_offer', { target: 'agent', offer });
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
    const label = currentPrediction.display_label || currentPrediction.label || top?.display_label || top?.label
    const confidence = currentPrediction.confidence || top?.confidence || 0
    if (!label) return

    setPredictionLog((prev) => {
      const last = prev[0]
      if (last && last.label === label && Math.abs(last.confidence - confidence) < 0.001) return prev
      return [{ label, confidence, timestamp: Date.now() }, ...prev].slice(0, 4)
    })
  }, [currentPrediction?.timestamp, currentPrediction?.window_filled, currentPrediction?.label, currentPrediction?.display_label, currentPrediction?.confidence, currentPrediction?.top_predictions])

  const predictionStatus = currentPrediction ? getPredictionStatus(currentPrediction) : ''
  const isRecognized = !!currentPrediction?.window_filled && !!currentPrediction?.label && (currentPrediction?.confidence ?? 0) >= 0.30
  const bannerLabel = isRecognized ? predictionStatus.replace('인식 중... ', '') : predictionStatus

  const buildChatText = () => {
    if (messages.length === 0) return '대화 내역이 없습니다.'
    return messages.map((m) => `[${m.timestamp.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })}] ${m.sender === 'agent' ? '상담원' : '민원인'}: ${m.text}`).join('\n')
  }

  const buildConsultationSummaryInput = () => {
    const conversation = messages.map((m) => {
      const speaker = m.sender === 'agent' ? 'agent' : 'citizen'
      const time = m.timestamp.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })
      return `[${time}] ${speaker}: ${m.text}`
    })

    conversation.unshift(`citizen_name: ${actualCitizenName}`)
    conversation.unshift(`citizen_phone: ${actualCitizenPhone}`)

    if (messages.length === 0) {
      conversation.push('no conversation messages')
    }

    return conversation
  }

  const formatPhone = (val: string) => {
    const digits = val.replace(/[^0-9]/g, '').slice(0, 11)
    if (digits.length <= 3) return digits
    if (digits.length <= 7) return `${digits.slice(0, 3)}-${digits.slice(3)}`
    return `${digits.slice(0, 3)}-${digits.slice(3, 7)}-${digits.slice(7)}`
  }

  const handleSendKakaoSummary = async () => {
    const cleaned = actualCitizenPhone.replace(/[^0-9]/g, '')
    if (cleaned.length < 10) return

    const accessToken = localStorage.getItem('KAKAO_ACCESS_TOKEN') || ''
    const refreshToken = localStorage.getItem('KAKAO_REFRESH_TOKEN') || ''

    if (!accessToken && !refreshToken) {
      const redirectUri = encodeURIComponent(window.location.origin + '/kakao/callback');
      const clientId = 'dbc36d320c333e45410fe1f7b642fd11'; 
      
      // 💡 [핵심 해결 코드] 기존 주소 맨 끝에 &prompt=login 을 추가합니다.
      // 이렇게 하면 브라우저에 쿠키가 남아있어도 무조건 카카오 계정 로그인 화면이 뜹니다.
      const loginUrl = `https://kauth.kakao.com/oauth/authorize?client_id=${clientId}&redirect_uri=${redirectUri}&response_type=code&prompt=login`;
      
      const popup = window.open(loginUrl, 'kakaoLogin', 'width=450,height=600');
      
      if (!popup) {
        setSendError('팝업 창이 차단되었습니다! 주소창 우측에서 팝업 차단을 허용해주세요.');
        setSendStatus('error');
        return;
      }
      
      const timer = setInterval(() => {
        if (popup.closed) {
          clearInterval(timer);
          if (localStorage.getItem('KAKAO_ACCESS_TOKEN')) {
            handleSendKakaoSummary(); 
          } else {
            setSendError('로그인이 취소되었습니다.');
            setSendStatus('error');
          }
        }
      }, 1000);
      return; 
    }

    // 💡 [STEP 2] 토큰이 있다면 전송 시작 (session_end 시 이미 생성된 요약 재사용)
    setSendStatus('sending')
    setSendError('')

    const summaryText = cachedSummary || buildChatText()

    try {
      // 실제 카카오톡 발송
      const res = await fetch('/api/notify/kakao', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ 
          access_token: accessToken, 
          refresh_token: refreshToken, 
          summary: summaryText 
        }),
      })
      const data = await res.json().catch(() => ({}))
      
      if (!res.ok) throw new Error(data.error || '전송 실패')

      // 전송 성공 시 UI 업데이트 (사용자는 여기서 '확인' 버튼을 보게 됩니다)
      setSendStatus('sent')

      // ✅ 소켓 emit → AgentLaunchScreen의 applyDiagnosisSummary 트리거
      const sentAt = new Date().toISOString()
      socket.emit('consultation_summary_saved', {
        citizenName: actualCitizenName,
        citizenPhone: actualCitizenPhone,
        consultationSummary: summaryText,
        isSent: true,
        deliveryStatus: 'kakao_sent',
        sentAt,
      })

      // ✅ localStorage 직접 패치 (모달 바로 열어도 반영되도록)
      const savedRecords = JSON.parse(localStorage.getItem('consultation_records') || '[]')
      const normalizedPhone = actualCitizenPhone.replace(/\D/g, '')
      const recIdx = savedRecords.findIndex((r: { citizenPhone?: string }) =>
        (r.citizenPhone || '').replace(/\D/g, '') === normalizedPhone
      )
      if (recIdx >= 0) {
        savedRecords[recIdx] = { ...savedRecords[recIdx], consultationSummary: summaryText, isSent: true, deliveryStatus: 'kakao_sent', sentAt }
        localStorage.setItem('consultation_records', JSON.stringify(savedRecords))
      }

    } catch (err) {
      // ✅ 실패 시 클립보드 복사 후 소켓/localStorage 업데이트
      try { await navigator.clipboard.writeText(summaryText) } catch { /* ignore */ }

      const sentAt = new Date().toISOString()
      socket.emit('consultation_summary_saved', {
        citizenName: actualCitizenName,
        citizenPhone: actualCitizenPhone,
        consultationSummary: summaryText,
        isSent: false,
        deliveryStatus: 'clipboard_copied',
        sentAt,
      })

      const savedRecords = JSON.parse(localStorage.getItem('consultation_records') || '[]')
      const normalizedPhone = actualCitizenPhone.replace(/\D/g, '')
      const recIdx = savedRecords.findIndex((r: { citizenPhone?: string }) =>
        (r.citizenPhone || '').replace(/\D/g, '') === normalizedPhone
      )
      if (recIdx >= 0) {
        savedRecords[recIdx] = { ...savedRecords[recIdx], consultationSummary: summaryText, isSent: false, deliveryStatus: 'clipboard_copied', sentAt }
        localStorage.setItem('consultation_records', JSON.stringify(savedRecords))
      }

      setSendError(err instanceof Error ? err.message : '전송 실패');
      setSendStatus('error');
    }
  }

  const handleClosePopup = () => {
    localStorage.removeItem('KAKAO_ACCESS_TOKEN');
    localStorage.removeItem('KAKAO_REFRESH_TOKEN');

    // 카카오 전송 완료 상태면 이미 kakao_sent로 저장됐으므로 덮어쓰지 않음
    if (sendStatus !== 'sent') {
      const summaryText = cachedSummary || buildChatText()
      const sentAt = new Date().toISOString()

      socket.emit('consultation_summary_saved', {
        citizenName: actualCitizenName,
        citizenPhone: actualCitizenPhone,
        consultationSummary: summaryText,
        isSent: false,
        deliveryStatus: 'pending',
        sentAt,
      })

      const savedRecords = JSON.parse(localStorage.getItem('consultation_records') || '[]')
      const normalizedPhone = actualCitizenPhone.replace(/\D/g, '')
      const recIdx = savedRecords.findIndex((r: { citizenPhone?: string }) =>
        (r.citizenPhone || '').replace(/\D/g, '') === normalizedPhone
      )
      if (recIdx >= 0) {
        savedRecords[recIdx] = {
          ...savedRecords[recIdx],
          consultationSummary: summaryText,
          isSent: false,
          deliveryStatus: 'pending',
          sentAt,
        }
        localStorage.setItem('consultation_records', JSON.stringify(savedRecords))
      }
    }

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
      
      {isWaitingForAgent && (
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
          <h2 className="text-2xl md:text-3xl font-black text-slate-800 mb-4 tracking-tight">상담실을 준비하고 있습니다</h2>
          <p className="text-sm md:text-lg text-slate-500 font-medium">상담원이 이전 상담를 정리 중입니다. 잠시만 기다려주세요.</p>
        </div>
      )}

      <header className="flex-shrink-0 flex items-center justify-between px-6 py-3 border-b border-slate-200 bg-white z-10 shadow-sm">
        <div className="flex flex-col gap-0.5">
          <span className="text-xl font-black tracking-tight text-slate-900">수어 통역 키오스크</span>
          <span className="text-sm text-blue-600 font-bold bg-blue-50 px-2 py-0.5 rounded-md inline-block w-fit">{roomLabel} - {actualCitizenName}님</span>
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
              <VideoFeed videoRef={videoRef} canvasRef={canvasRef} landmarkCanvasRef={landmarkCanvasRef} isRunning={isRunning} isDemoMode={isDemoMode} currentPrediction={currentPrediction} predictionStatus={predictionStatus} onVideoEnded={handleDemoVideoEnded} camFps={camFps} sendFps={sendFps} liveSegmentStatus={liveSegmentStatus} />
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
                {sessionEnded ? '상담 종료됨' : isRunning ? '카메라 중지' : '카메라 시작'}
              </button>

              <div className="relative flex items-stretch gap-2 flex-1">
                <button
                  onClick={() => {
                    const def =
                      validationDemoScenarios.find((s) => s.clips[0]?.id === 'resident_realz03_01_hello') ??
                      validationDemoScenarios[0]
                    if (def) handleDemoSelect(def)
                  }}
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
                        key={scenario.displayText}
                        onClick={() => handleDemoSelect(scenario)}
                        className="flex w-full items-center gap-2 border-b border-slate-100 px-3 py-2.5 text-left text-sm font-black text-slate-800 last:border-b-0 hover:bg-blue-50"
                      >
                        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-blue-600 text-white text-[10px]">{index + 1}</span>
                        <span className="truncate">{scenario.displayText}</span>
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
          {welfarePanel.length > 0 && (
            <div className="shrink-0 px-6 pt-4 pb-2 bg-slate-900">
              <WelfarePanel items={welfarePanel} onClose={dismissWelfarePanel} />
            </div>
          )}
          <div className="flex-1 overflow-y-auto flex flex-col gap-3 px-6 py-4">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full gap-3 text-center opacity-40">
                <div className="w-14 h-14 rounded-full bg-slate-100 flex items-center justify-center">
                  <svg className="w-7 h-7" viewBox="0 0 24 24" fill="none" stroke="#475569" strokeWidth="1.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                </div>
                <div>
                  <p className="text-base font-black text-slate-600">대화 내역이 없습니다</p>
                  <p className="text-xs text-slate-500 mt-1 font-medium">상담이 시작되면 대화 내용이 기록됩니다</p>
                </div>
              </div>
            ) : (
              messages.map((msg) => <ChatMessage key={msg.id} message={msg} />)
            )}
            <div ref={chatEndRef} />
          </div>
          <footer className="shrink-0 py-2.5 px-6 border-t border-slate-100 bg-white flex items-center justify-center">
            <p className="text-xs text-slate-400 font-bold text-center">수어로 증상을 표현해 주세요. 인공지능이 즉시 상담원께 전달합니다.</p>
          </footer>
        </div>
      </div>

      {showPopup && (
        <div className="absolute inset-0 flex items-center justify-center z-50 bg-slate-900/60 backdrop-blur-sm p-4">
          <div className="w-full max-w-sm bg-white border border-slate-100 rounded-3xl px-6 pt-6 pb-6 flex flex-col items-center shadow-2xl relative">
            <div className="w-14 h-14 rounded-full bg-[#FEE500] flex items-center justify-center mb-3 shrink-0 shadow-lg shadow-yellow-100">
              <svg className="w-7 h-7" viewBox="0 0 24 24" fill="#3C1E1E"><path d="M12 3C6.477 3 2 6.477 2 10.8c0 2.8 1.718 5.253 4.286 6.72l-.857 3.2 3.715-2.457C10.012 18.41 10.99 18.6 12 18.6c5.523 0 10-3.477 10-7.8S17.523 3 12 3z"/></svg>
            </div>
            <h2 className="text-2xl font-black text-slate-900 text-center leading-tight mb-1">상담을 마쳤습니다</h2>
            <p className="text-sm text-slate-500 text-center font-medium">등록하신 휴대전화로<br/>상담 대화 내역을 보내드릴까요?</p>
            
            <div className="flex flex-col items-center justify-center rounded-xl mt-4 mb-5 w-full py-3 bg-slate-50 border-2 border-slate-100 shadow-inner">
              <span className="text-xs text-slate-400 font-bold mb-1">{actualCitizenName}님의 연락처</span>
              <div className="text-xl font-black text-slate-800 tracking-wider">{formatPhone(actualCitizenPhone)}</div>
            </div>
            
            <div className="flex flex-col w-full gap-2">
              <button onClick={handleSendKakaoSummary} disabled={sendStatus === 'sending' || actualCitizenPhone.length < 10} className={`w-full py-3 rounded-xl text-sm md:text-base font-black flex items-center justify-center gap-2 transition-all ${sendStatus === 'sending' || actualCitizenPhone.length < 10 ? 'bg-slate-100 text-slate-300' : 'bg-[#FEE500] text-[#3C1E1E] hover:brightness-105 active:scale-[0.98] shadow-md shadow-yellow-100'}`}>
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
                <p className="text-xs text-slate-500 text-center leading-relaxed font-medium">메시지를 성공적으로 보냈습니다.<br/>상담실을 나가셔도 좋습니다.</p>
                <button onClick={handleClosePopup} className="mt-6 w-full py-3 rounded-xl text-sm font-black bg-blue-600 text-white hover:bg-blue-700 shadow-md shadow-blue-100 transition-all active:scale-95">확인 (메인으로)</button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
