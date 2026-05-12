import { useEffect, useRef, useState, useCallback } from 'react'
import { useSignLanguage } from '../hooks/useSignLanguage'
import { useSpeechRecognition } from '../hooks/useSpeechRecognition'
import type { ChatMessage } from '../types'
import '../styles/SignLanguageStream.css'

export default function SignLanguageStream() {
  const chatEndRef = useRef<HTMLDivElement>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [showSettings, setShowSettings] = useState(false)
  
  // 💡 메인 훅 연결: 메시지가 들어오면 채팅창에 추가
  const handleNewMessage = useCallback((msg: ChatMessage) => {
    setMessages((prev) => [...prev, msg])
  }, [])

  // 💡 1. 커스텀 훅 가져오기 (코드 중복 완전 해결)
  const {
    videoRef, canvasRef, landmarkCanvasRef,
    isRunning, isDemoMode, currentPrediction,
    videoDevices, selectedDeviceId, setSelectedDeviceId,
    startCamera, stopCamera, getPredictionStatus
  } = useSignLanguage(handleNewMessage)

  const {
    isActive: isVoiceActive, 
    voiceLevels, 
    start: startVoiceRecognition, 
    stop: stopVoiceRecognition 
  } = useSpeechRecognition(handleNewMessage)

  // 채팅 자동 스크롤
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const clearMessages = () => setMessages([])

  // UI 상태 변수들
  const cameraStateLabel = isDemoMode ? '샘플 영상 재생 중' : isRunning ? '카메라 실행 중' : '카메라 대기'
  const voiceStateLabel = isVoiceActive ? '음성 인식 중' : '음성 대기'
  const detectStateLabel = currentPrediction?.has_hand ? '손 검출' : '검출 대기'
  
  const voiceBarColors = [
    'linear-gradient(180deg, #f0abfc 0%, #ec4899 100%)',
    'linear-gradient(180deg, #93c5fd 0%, #2563eb 100%)',
    'linear-gradient(180deg, #86efac 0%, #16a34a 100%)',
    'linear-gradient(180deg, #fde68a 0%, #f59e0b 100%)',
    'linear-gradient(180deg, #c4b5fd 0%, #7c3aed 100%)',
    'linear-gradient(180deg, #67e8f9 0%, #0891b2 100%)'
  ]

  return (
    <div className="sign-stream-container">
      {/* Header 영역 생략 (기존 UI 동일 유지) */}
      
      <main className="demo-grid">
        {/* 카메라 피드 영역 */}
        <section className="video-card primary-feed">
          <div className="video-wrapper">
             <video ref={videoRef} autoPlay playsInline className="camera-video" />
             <canvas ref={canvasRef} width={640} height={480} className="hidden-canvas" />
             <canvas ref={landmarkCanvasRef} width={640} height={480} className="landmark-canvas" />
             
             {currentPrediction && (
              <div className="prediction-overlay">
                <span className="prediction-caption">현재 예측</span>
                <strong>{getPredictionStatus(currentPrediction)}</strong>
              </div>
            )}
          </div>
        </section>

        {/* 사이드 패널 (음성/채팅) 영역 */}
        <section className="side-panel">
          <div className="voice-panel">
            <div className="voice-meter">
              {voiceLevels.map((level, index) => (
                <span
                  key={index}
                  className={isVoiceActive && level > 0.12 ? 'lit' : ''}
                  style={{
                    height: `${Math.round(level * 100)}%`,
                    background: isVoiceActive && level > 0.12 ? voiceBarColors[index % 6] : '#ccc'
                  }}
                />
              ))}
            </div>
          </div>

          <div className="chat-section">
             {messages.map((msg) => (
                <div key={msg.id} className={`chat-message ${msg.sender}`}>
                  <div className="message-text">{msg.text}</div>
                </div>
              ))}
              <div ref={chatEndRef} />
          </div>
        </section>
      </main>

      <section className="controls-section">
        <div className="button-group">
          {!isRunning ? (
            <button onClick={startCamera} className="btn-primary">카메라 시작</button>
          ) : (
            <button onClick={stopCamera} className="btn-danger">카메라 중지</button>
          )}

          {!isVoiceActive ? (
            <button onClick={startVoiceRecognition} className="btn-voice">음성 인식 시작</button>
          ) : (
            <button onClick={stopVoiceRecognition} className="btn-voice-active">음성 인식 중지</button>
          )}
        </div>
      </section>
    </div>
  )
}