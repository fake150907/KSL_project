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

type TextSizeMode = 'base' | 'large' | 'xlarge'
type WelfareThemeMode = 'light' | 'dark'

const CONTENT_TEXT_SIZE_CLASSES: Record<TextSizeMode, { caption: string; chat: string }> = {
  base: { caption: 'text-2xl', chat: 'text-sm' },
  large: { caption: 'text-3xl', chat: 'text-base' },
  xlarge: { caption: 'text-4xl', chat: 'text-lg' },
}

export default function CitizenKiosk({
  messages,
  onNewMessage,
  onSessionReset,
  roomLabel = '3번 창구',
  sessionEnded = false,
  citizenName = '민원인',
  citizenPhone = '01000000000',
}: CitizenKioskProps) {
  const navigate = useNavigate()
  const location = useLocation()
  const navState = location.state as { citizenData?: { name: string, phone: string } } | null
  const storedCitizenData = (() => {
    try {
      return JSON.parse(sessionStorage.getItem('current_citizen_data') || 'null') as { name?: string; phone?: string } | null
    } catch {
      return null
    }
  })()
  const activeCitizenData = navState?.citizenData || storedCitizenData

  const actualCitizenName = activeCitizenData?.name || citizenName
  const actualCitizenPhone = activeCitizenData?.phone || citizenPhone

  const chatEndRef = useRef<HTMLDivElement>(null)
  const staffChatScrollRef = useRef<HTMLDivElement>(null)
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null)
  
  const [isWaitingForAgent, setIsWaitingForAgent] = useState(sessionEnded)
  const [sendStatus, setSendStatus] = useState<'idle' | 'sending' | 'sent' | 'error'>('idle')
  const [sendError, setSendError] = useState('')
  const [showPopup, setShowPopup] = useState(false)
  const [cachedSummary, setCachedSummary] = useState('')
  const [showDemoList, setShowDemoList] = useState(false)
  const [predictionLog, setPredictionLog] = useState<Array<{ label: string; confidence: number; timestamp: number }>>([])
  const [agentVoiceActive, setAgentVoiceActive] = useState(false)
  const [agentVoiceLevels, setAgentVoiceLevels] = useState<number[]>(Array(42).fill(0.12))

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
      if (msg.sender === 'agent') {
        setAgentVoiceActive(true)
        window.setTimeout(() => setAgentVoiceActive(false), 2800)
      }
    }
    socket.on('chat_message', handleIncomingMessage)
    return () => {
      socket.off('chat_message', handleIncomingMessage)
    }
  }, [onNewMessage])

  useEffect(() => {
    const handleAgentVoiceStatus = (payload: { active?: boolean; levels?: number[] }) => {
      setAgentVoiceActive(!!payload?.active)
      if (Array.isArray(payload?.levels) && payload.levels.length > 0) {
        setAgentVoiceLevels(payload.levels)
      }
    }

    socket.on('agent_voice_status', handleAgentVoiceStatus)
    return () => {
      socket.off('agent_voice_status', handleAgentVoiceStatus)
    }
  }, [])

  useEffect(() => {
    const scrollEl = staffChatScrollRef.current
    if (scrollEl) scrollEl.scrollTo({ top: scrollEl.scrollHeight, behavior: 'smooth' })
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
  const [textSizeMode, setTextSizeMode] = useState<TextSizeMode>('base')
  const [welfareThemeMode, setWelfareThemeMode] = useState<WelfareThemeMode>('light')

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

  const handleEmergencyCall = () => {
    if (window.confirm('119 긴급 전화로 연결하시겠습니까?')) {
      window.location.href = 'tel:119'
    }
  }

  const citizenStep = isRunning ? (isRecognized ? 3 : currentPrediction?.window_filled ? 2 : 1) : 0
  const agentStep = agentVoiceActive ? 1 : 0
  const contentTextSize = CONTENT_TEXT_SIZE_CLASSES[textSizeMode]
  const isDarkMode = welfareThemeMode === 'dark'
  const latestGlossCaption = [...messages]
    .reverse()
    .find((message) =>
      message.sender === 'citizen'
      && ['수어 문장 변환', '데모 문장 변환', '수어 글로스', '데모 글로스'].includes(message.label || '')
    )?.text || ''
  const latestCitizenCaption = latestGlossCaption || [...messages].reverse().find((message) => message.sender === 'citizen')?.text || ''

  const renderStepLane = (tone: 'blue' | 'green', label: string, activeStep: number) => {
    const activeClass = tone === 'blue'
      ? 'border-[#2563eb] bg-[#2563eb] text-white shadow-[0_0_16px_rgba(37,99,235,0.22)]'
      : 'border-[#0f766e] bg-[#0f766e] text-white shadow-[0_0_16px_rgba(15,118,110,0.24)]'
    const labelClass = tone === 'blue' ? (isDarkMode ? 'text-[#93c5fd]' : 'text-[#2563eb]') : (isDarkMode ? 'text-[#5eead4]' : 'text-[#0f766e]')
    const steps = ['입력', '출력']
    const displayStep = Math.min(activeStep, steps.length)
    const arrowClass = tone === 'blue'
      ? tone === 'blue' ? 'text-[#2563eb]' : 'text-[#0f766e]'
      : isDarkMode ? 'text-[#111827]' : 'text-white'

    return (
      <div className={`rounded-lg border px-3 py-3 ${isDarkMode ? 'border-[#324155] bg-[#111827]' : 'border-[#d8e0ea] bg-white'}`}>
        <p className={`mb-3 text-center text-xs font-black ${labelClass}`}>{label}</p>
        <div className="grid grid-cols-[minmax(0,1fr)_2.25rem_minmax(0,1fr)] items-center gap-2 sm:grid-cols-[minmax(0,1fr)_2.75rem_minmax(0,1fr)]">
          {steps.map((step, index) => (
            <div key={step} className="contents">
              {index === 1 && (
                <div className="flex h-9 items-center justify-center">
                  <svg className={`h-6 w-9 -mt-4 transition-colors duration-300 sm:w-11 ${arrowClass}`} viewBox="0 0 44 24" fill="none" stroke="currentColor" strokeWidth="3.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    <path d="M5 12h28" />
                    <path d="m26 5 8 7-8 7" />
                  </svg>
                </div>
              )}
              <div className="flex min-w-0 flex-col items-center gap-1.5">
                <span className={`flex h-9 w-9 items-center justify-center rounded-full border-2 text-sm font-black transition-all duration-300 ${displayStep === index + 1 ? activeClass : isDarkMode ? 'border-[#40516a] bg-[#172235] text-slate-200' : 'border-[#cbd5e1] bg-white text-[#334155]'}`}>
                  {index + 1}
                </span>
                <span className={`text-[11px] font-bold ${isDarkMode ? 'text-slate-300' : 'text-slate-600'}`}>{step}</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  const VoiceBars = ({ active, levels }: { active: boolean; levels: number[] }) => (
    <div className="flex h-10 w-full max-w-[300px] items-center justify-center gap-1" aria-label="직원 음성 출력 상태">
      {Array.from({ length: 42 }).map((_, index) => {
        const level = levels[index % levels.length] ?? 0.12
        const height = active ? Math.max(8, Math.round(level * 40)) : 8

        return (
          <span
            key={index}
            className={`w-1 rounded-full transition-all duration-100 ${active ? 'bg-[#14b8a6]' : 'bg-slate-300'}`}
            style={{ height: `${height}px` }}
          />
        )
      })}
    </div>
  )

  return (
    <div data-theme={isDarkMode ? 'dark' : 'light'} className={`fixed inset-0 flex flex-col overflow-hidden transition-colors duration-300 ${isDarkMode ? 'bg-[#0f172a]' : 'bg-[#f5f7fb]'}`}>
      
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

      <header className={`z-10 flex flex-shrink-0 items-center justify-between border-b px-6 py-3 shadow-sm transition-colors duration-300 ${isDarkMode ? 'border-[#263244] bg-[#121b2b]' : 'border-[#d8e0ea] bg-white'}`}>
        <div className="flex flex-col gap-0.5">
          <span className={`text-xl font-black tracking-tight ${isDarkMode ? 'text-slate-50' : 'text-[#172033]'}`}>수어 통역 키오스크</span>
          <span className={`inline-block w-fit rounded-md px-2 py-0.5 text-sm font-bold ${isDarkMode ? 'bg-[#1e3a5f] text-[#bfdbfe]' : 'bg-[#eaf2ff] text-[#2563eb]'}`}>{roomLabel} - {actualCitizenName}님</span>
        </div>
        <div className="flex items-center gap-4">
          <div className={`flex items-center overflow-hidden rounded-lg border shadow-sm ${isDarkMode ? 'border-[#324155] bg-[#172235]' : 'border-[#d8e0ea] bg-[#f8fafc]'}`}>
            <span className={`border-r px-3 py-2 text-sm font-black ${isDarkMode ? 'border-[#324155] text-slate-200' : 'border-[#d8e0ea] text-slate-600'}`}>글자 크기</span>
            {([
              ['base', '가', 'text-sm'],
              ['large', '가', 'text-base'],
              ['xlarge', '가', 'text-lg'],
            ] as const).map(([mode, label, sizeClass]) => (
              <button
                key={mode}
                type="button"
                onClick={() => setTextSizeMode(mode)}
                aria-label={`글자 크기 ${mode === 'base' ? '기본' : mode === 'large' ? '크게' : '더 크게'}`}
                title={`글자 크기 ${mode === 'base' ? '기본' : mode === 'large' ? '크게' : '더 크게'}`}
                className={`flex h-10 w-11 items-center justify-center border-r font-black transition-all last:border-r-0 active:scale-95 ${sizeClass} ${isDarkMode ? 'border-[#324155]' : 'border-[#d8e0ea]'} ${textSizeMode === mode ? 'bg-[#2563eb] text-white shadow-inner' : isDarkMode ? 'bg-[#111827] text-slate-100 hover:bg-[#1f2a3d]' : 'bg-white text-[#172033] hover:bg-[#eef6ff]'}`}
              >
                {label}
              </button>
            ))}
          </div>
          <div className={`flex items-center overflow-hidden rounded-lg border shadow-sm ${isDarkMode ? 'border-[#324155] bg-[#172235]' : 'border-[#d8e0ea] bg-[#f8fafc]'}`}>
            <span className={`border-r px-3 py-2 text-sm font-black ${isDarkMode ? 'border-[#324155] text-slate-200' : 'border-[#d8e0ea] text-slate-600'}`}>화면 모드</span>
            {([
              ['light', '라이트'],
              ['dark', '다크'],
            ] as const).map(([mode, label]) => (
              <button
                key={mode}
                type="button"
                onClick={() => setWelfareThemeMode(mode)}
                aria-label={`${label} 모드`}
                title={`${label} 모드`}
                className={`flex h-10 min-w-14 items-center justify-center border-r px-3 text-sm font-black transition-all last:border-r-0 active:scale-95 ${isDarkMode ? 'border-[#324155]' : 'border-[#d8e0ea]'} ${welfareThemeMode === mode ? (mode === 'dark' ? 'bg-[#2563eb] text-white shadow-inner' : 'bg-[#2563eb] text-white shadow-inner') : isDarkMode ? 'bg-[#111827] text-slate-100 hover:bg-[#1f2a3d]' : 'bg-white text-slate-700 hover:bg-[#f1f5f9]'}`}
              >
                {label}
              </button>
            ))}
          </div>
          <div className="flex flex-col items-end gap-0.5">
            <span className={`text-2xl font-black tabular-nums ${isDarkMode ? 'text-slate-100' : 'text-[#172033]'}`}>{clock}</span>
            <span className={`flex items-center gap-1.5 text-xs font-bold ${isRunning ? (isDarkMode ? 'text-[#5eead4]' : 'text-[#0f766e]') : 'text-slate-400'}`}>
              <span className={`h-2 w-2 shrink-0 rounded-full ${isRunning ? 'bg-[#14b8a6] shadow-[0_0_8px_rgba(20,184,166,0.36)]' : 'bg-slate-300'}`} />
              {isRunning ? '카메라 활성' : '카메라 대기'}
            </span>
          </div>
        </div>
      </header>

      <div className={`min-h-0 flex-1 overflow-y-auto p-3 transition-colors duration-300 md:p-5 ${isDarkMode ? 'bg-[#0f172a]' : 'bg-[#f5f7fb]'}`}>
        <div className="mx-auto grid min-h-full w-full max-w-[1280px] grid-cols-1 gap-3 lg:h-full lg:min-h-0 lg:grid-cols-[minmax(300px,1fr)_minmax(260px,0.78fr)_minmax(300px,1fr)]">
          <section className={`order-1 flex min-h-[560px] flex-col overflow-hidden rounded-lg border-[3px] shadow-sm transition-colors duration-300 lg:h-full lg:min-h-0 ${isDarkMode ? 'border-[#2b3a50] bg-[#121b2b]' : 'border-[#c9d7ee] bg-white'}`}>
            <div className="bg-[#2563eb] px-4 py-3 text-center text-base font-black text-white">민원인 (수어 입력)</div>
            <div className={`min-h-0 flex-1 p-3 ${isDarkMode ? 'bg-[#0f172a]' : 'bg-[#f3f6fb]'}`}>
              <div className={`relative h-full min-h-[270px] overflow-hidden rounded-lg border bg-black ${isDarkMode ? 'border-[#334155]' : 'border-[#d8e0ea]'}`}>
                <VideoFeed videoRef={videoRef} canvasRef={canvasRef} landmarkCanvasRef={landmarkCanvasRef} isRunning={isRunning} currentPrediction={currentPrediction} predictionStatus={predictionStatus} onVideoEnded={handleDemoVideoEnded} camFps={camFps} sendFps={sendFps} />
              </div>
            </div>
            <div className={`flex shrink-0 items-center gap-3 border-t px-4 py-3 transition-colors duration-300 ${isDarkMode ? 'border-[#263244]' : 'border-[#d8e0ea]'} ${isRunning ? isRecognized ? (isDarkMode ? 'bg-[#123334]' : 'bg-[#ecfdf5]') : (isDarkMode ? 'bg-[#132746]' : 'bg-[#eff6ff]') : (isDarkMode ? 'bg-[#121b2b]' : 'bg-white')}`}>
              {isRunning ? (
                <>
                  <div className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${isRecognized ? 'bg-emerald-100' : 'bg-blue-100'}`}>
                    <span className={`h-2.5 w-2.5 rounded-full ${isRecognized ? 'bg-emerald-500' : 'bg-blue-500'}`} />
                  </div>
                  <div className="flex min-w-0 flex-col">
                    <span className={`text-xs font-bold ${isRecognized ? 'text-emerald-600' : 'text-blue-600'}`}>{isRecognized ? '수어 인식 성공' : '실시간 인식 중'}</span>
                    <span className={`truncate text-sm font-black leading-tight ${isDarkMode ? 'text-slate-100' : 'text-slate-800'}`}>{bannerLabel || '손을 카메라에 맞춰 보여주세요.'}</span>
                  </div>
                </>
              ) : (
                <span className="mx-auto text-center text-xs font-bold italic text-slate-400">카메라 시작 버튼을 누르면 수어 번역을 시작합니다.</span>
              )}
            </div>
            <div className={`border-t p-4 ${isDarkMode ? 'border-[#263244] bg-[#121b2b]' : 'border-[#d8e0ea] bg-white'}`}>
              <span className={`mb-2 block text-xs font-black ${isDarkMode ? 'text-[#93c5fd]' : 'text-[#2563eb]'}`}>실시간 자막</span>
              <div className={`h-[116px] overflow-y-scroll overscroll-contain rounded-lg border px-4 py-3 font-black leading-snug ${contentTextSize.caption} ${isDarkMode ? 'border-[#334155] bg-[#0f172a]' : 'border-[#d8e0ea] bg-[#f8fafc]'} ${latestCitizenCaption ? (isDarkMode ? 'text-slate-50' : 'text-[#172033]') : 'text-slate-400'}`}>
                {latestCitizenCaption || '수어를 인식하면 변환 문장이 표시됩니다.'}
              </div>
            </div>
            <div className={`flex shrink-0 flex-col gap-2 border-t px-4 py-3 ${isDarkMode ? 'border-[#263244] bg-[#121b2b]' : 'border-[#edf1f7] bg-white'}`}>
              <div className="flex gap-2">
                <button onClick={isRunning ? stopCamera : startCamera} disabled={sessionEnded} className={`flex-1 rounded-lg border py-2.5 text-sm font-black shadow-sm transition-all active:scale-95 ${sessionEnded ? 'cursor-not-allowed border-slate-200 bg-slate-50 text-slate-300' : isRunning ? 'border-[#fecaca] bg-[#fff1f2] text-[#dc2626] hover:bg-[#ffe4e6]' : 'border-[#1d4ed8] bg-[#2563eb] text-white shadow-blue-500/15 hover:bg-[#1d4ed8]'}`}>
                  {sessionEnded ? '상담 종료됨' : isRunning ? '카메라 중지' : '카메라 시작'}
                </button>
                <div className="relative flex flex-1 items-stretch gap-2">
                  <button onClick={() => validationDemoScenarios[0] && handleDemoSelect(validationDemoScenarios[0])} disabled={sessionEnded} className={`flex flex-1 items-center justify-center rounded-lg border py-2.5 text-sm font-black shadow-sm transition-all active:scale-95 ${sessionEnded ? 'cursor-not-allowed border-slate-200 bg-slate-50 text-slate-300' : isDemoMode ? 'border-[#99f6e4] bg-[#ecfdf5] text-[#0f766e]' : isDarkMode ? 'border-[#334155] bg-[#111827] text-white hover:bg-[#1f2937]' : 'border-[#cbd5e1] bg-[#172033] text-white hover:bg-[#0f172a]'}`}>
                    {isDemoMode ? `데모: ${activeDemoLabel}` : '데모 시연'}
                  </button>
                  <button type="button" onClick={() => setShowDemoList((prev) => !prev)} disabled={sessionEnded} className={`flex w-10 items-center justify-center rounded-lg border shadow-sm active:scale-95 disabled:text-slate-300 ${isDarkMode ? 'border-[#334155] bg-[#111827] text-slate-100 hover:bg-[#1f2937]' : 'border-[#d8e0ea] bg-white text-slate-700 hover:bg-[#f8fafc]'}`}>
                    <svg className="h-4 w-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"><path d="m6 9 6 6 6-6" /></svg>
                  </button>
                  {showDemoList && (
                    <div className={`absolute bottom-full left-0 right-0 z-50 mb-1 overflow-hidden rounded-lg border shadow-2xl ${isDarkMode ? 'border-[#334155] bg-[#111827]' : 'border-[#d8e0ea] bg-white'}`}>
                      {validationDemoScenarios.map((scenario, index) => (
                        <button key={scenario.displayText} onClick={() => handleDemoSelect(scenario)} className={`flex w-full items-center gap-2 border-b px-3 py-2.5 text-left text-sm font-black last:border-b-0 ${isDarkMode ? 'border-[#263244] text-slate-100 hover:bg-[#1f2937]' : 'border-slate-100 text-[#172033] hover:bg-[#eff6ff]'}`}>
                          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md bg-blue-600 text-[10px] text-white">{index + 1}</span>
                          <span className="truncate">{scenario.displayText}</span>
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
              {videoDevices.length > 1 && (
                <select value={selectedDeviceId} onChange={(e) => setSelectedDeviceId(e.target.value)} className={`w-full rounded-lg border px-3 py-2 text-sm font-bold outline-none focus:border-[#2563eb] ${isDarkMode ? 'border-[#334155] bg-[#111827] text-slate-100' : 'border-[#d8e0ea] bg-[#f8fafc] text-slate-700'}`}>
                  {videoDevices.map((d, i) => <option key={d.deviceId} value={d.deviceId}>{d.label || `카메라 ${i + 1}`}</option>)}
                </select>
              )}
            </div>
          </section>

          <section className={`order-2 flex min-h-[430px] flex-col overflow-hidden rounded-lg border-[3px] shadow-sm transition-colors duration-300 lg:h-full lg:min-h-0 ${isDarkMode ? 'border-[#263244] bg-[#121b2b]' : 'border-[#d8e0ea] bg-white'}`}>
            <div className={`border-b px-4 py-3 text-center text-base font-black ${isDarkMode ? 'border-[#263244] text-slate-50' : 'border-[#d8e0ea] text-[#172033]'}`}>통역 상태</div>
            <div className={`flex shrink-0 flex-col gap-3 border-b px-4 py-4 ${isDarkMode ? 'border-[#263244]' : 'border-[#d8e0ea]'}`}>
              {renderStepLane('blue', '민원인 수어 입력 흐름', citizenStep)}
              <div className={`mt-1 flex items-center justify-center gap-2 text-sm font-bold ${isDarkMode ? 'text-slate-200' : 'text-[#334155]'}`}><span className="h-2.5 w-2.5 rounded-full bg-[#10b981] shadow-[0_0_8px_rgba(16,185,129,0.28)]" />양방향 실시간 통역 중</div>
            </div>
            {welfarePanel.length > 0 ? (
              <div className={`flex min-h-0 flex-1 border-b p-2.5 ${isDarkMode ? 'border-[#263244] bg-[#101827]' : 'border-[#d8e0ea] bg-[#eef5ff]'}`}>
                <WelfarePanel items={welfarePanel} onClose={dismissWelfarePanel} compact theme={welfareThemeMode} />
              </div>
            ) : (
              <div className="flex-1" />
            )}
            <div className={`shrink-0 border-t px-4 py-3 ${isDarkMode ? 'border-[#263244]' : 'border-[#d8e0ea]'}`}>
              <button onClick={handleEmergencyCall} className="flex w-full items-center justify-center gap-2 rounded-lg border border-[#dc2626] bg-[#ef4444] py-3 text-base font-black text-white shadow-sm shadow-red-500/10 transition-all hover:bg-[#dc2626] active:scale-[0.98]">
                <svg className="h-5 w-5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.8 19.8 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6A19.8 19.8 0 0 1 2.12 4.18 2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.13.96.35 1.9.65 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.22a2 2 0 0 1 2.11-.45c.91.3 1.85.52 2.81.65A2 2 0 0 1 22 16.92z" />
                </svg>
                긴급 전화
              </button>
            </div>
          </section>

          <section className={`order-3 flex min-h-[430px] flex-col overflow-hidden rounded-lg border-[3px] shadow-sm transition-colors duration-300 lg:h-full lg:min-h-0 ${isDarkMode ? 'border-[#24433f] bg-[#121b2b]' : 'border-[#b9e4d4] bg-white'}`}>
            <div className="bg-[#0f766e] px-4 py-3 text-center text-base font-black text-white">직원 (음성 출력)</div>
            <div className={`flex shrink-0 flex-col items-center justify-center gap-3 border-b px-5 py-4 text-center ${isDarkMode ? 'border-[#263244] bg-[#101827]' : 'border-[#d8e0ea] bg-[#f8fafc]'}`}>
              <p className={`text-base font-black ${agentVoiceActive ? (isDarkMode ? 'text-[#5eead4]' : 'text-[#0f766e]') : 'text-slate-500'}`}>{agentVoiceActive ? '직원이 음성 출력 중입니다' : '직원 음성 출력 대기 중입니다'}</p>
              <div className="flex w-full items-center justify-center gap-4">
                <div className={`flex h-14 w-14 shrink-0 items-center justify-center rounded-full text-white shadow-lg transition-all ${agentVoiceActive ? 'scale-105 bg-[#0f766e] shadow-teal-500/20' : 'bg-slate-400'}`}>
                  <svg className="h-7 w-7" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5"><path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><path d="M12 19v3"/></svg>
                </div>
                <VoiceBars active={agentVoiceActive} levels={agentVoiceLevels} />
              </div>
            </div>
            <div ref={staffChatScrollRef} className={`min-h-0 flex-1 overflow-y-scroll overscroll-contain px-4 py-3 ${isDarkMode ? 'bg-[#121b2b]' : 'bg-white'}`}>
              {messages.length === 0 ? (
                <div className="flex h-full min-h-[180px] flex-col items-center justify-center gap-3 text-center opacity-40">
                  <div className={`flex h-12 w-12 items-center justify-center rounded-full ${isDarkMode ? 'bg-slate-800' : 'bg-slate-100'}`}><svg className="h-6 w-6" viewBox="0 0 24 24" fill="none" stroke="#475569" strokeWidth="1.5"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg></div>
                  <div><p className={`text-sm font-black ${isDarkMode ? 'text-slate-300' : 'text-slate-600'}`}>대화 내역이 없습니다</p><p className={`mt-1 text-xs font-medium ${isDarkMode ? 'text-slate-400' : 'text-slate-500'}`}>상담이 시작되면 대화 내용이 표시됩니다</p></div>
                </div>
              ) : (
                <div className="flex flex-col gap-3">
                  {messages.map((msg) => <ChatMessage key={msg.id} message={msg} textClassName={contentTextSize.chat} dark={isDarkMode} />)}
                  <div ref={chatEndRef} />
                </div>
              )}
            </div>
          </section>
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
