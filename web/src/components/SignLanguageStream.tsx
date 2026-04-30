import { useEffect, useRef, useState } from 'react'
import '../styles/SignLanguageStream.css'

interface Prediction {
  label: string | null
  confidence: number
  timestamp: number
  has_hand?: boolean
  window_filled?: boolean
  window_progress?: number
  window_size?: number
  missing_frames?: number
  max_missing_frames?: number
  top_predictions?: Array<{ label: string; confidence: number }>
}

interface ChatMessage {
  id: string
  sender: 'deaf' | 'speaker'
  text: string
  timestamp: Date
  avatar: string
}

const demoVideos = [
  {
    label: '우유',
    name: 'WORD0815 REAL18 L',
    src: '/demo-videos/NIA_SL_WORD0815_REAL18_L.mp4'
  },
  {
    label: '자다',
    name: 'WORD1377 REAL18 R',
    src: '/demo-videos/NIA_SL_WORD1377_REAL18_R.mp4'
  },
  {
    label: '가다',
    name: 'WORD0946 REAL17 R',
    src: '/demo-videos/NIA_SL_WORD0946_REAL17_R.mp4'
  }
]

export default function SignLanguageStream() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const landmarkCanvasRef = useRef<HTMLCanvasElement>(null)
  const chatEndRef = useRef<HTMLDivElement>(null)
  const recognitionRef = useRef<any>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioStreamRef = useRef<MediaStream | null>(null)
  const audioAnimationRef = useRef<number | null>(null)
  const isPredictingRef = useRef(false)
  const nextFrameIdRef = useRef(0)
  const latestFrameIdRef = useRef(0)
  const clientIdRef = useRef<string>(Math.random().toString(36).substring(7))

  const [, setPredictions] = useState<Prediction[]>([])
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isRunning, setIsRunning] = useState(false)
  const [isVoiceActive, setIsVoiceActive] = useState(false)
  const [showMjpegFeed, setShowMjpegFeed] = useState(false)
  const [currentPrediction, setCurrentPrediction] = useState<Prediction | null>(null)
  const [modelType, setModelType] = useState<'cnn_gru' | 'lstm'>('cnn_gru')
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.35)
  const [windowSize] = useState(32)
  const [stableMinCount, setStableMinCount] = useState(2)
  const [captureIntervalMs, setCaptureIntervalMs] = useState(100)
  const [maxMissingFrames, setMaxMissingFrames] = useState(3)
  const [lastAddedLabel, setLastAddedLabel] = useState('')
  const [lastAddedTime, setLastAddedTime] = useState(0)
  const [cooldownSeconds, setCooldownSeconds] = useState(2)
  const [showSettings, setShowSettings] = useState(false)
  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([])
  const [selectedDeviceId, setSelectedDeviceId] = useState('')
  const [voiceLevels, setVoiceLevels] = useState<number[]>(Array(48).fill(0.12))
  const [selectedDemoVideo, setSelectedDemoVideo] = useState(demoVideos[0].src)
  const [isDemoMode, setIsDemoMode] = useState(false)

  const refreshVideoDevices = async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return
    const devices = await navigator.mediaDevices.enumerateDevices()
    const cameras = devices.filter((device) => device.kind === 'videoinput')
    setVideoDevices(cameras)
    if (!selectedDeviceId && cameras[0]?.deviceId) {
      setSelectedDeviceId(cameras[0].deviceId)
    }
  }

  const stopCamera = () => {
    if (videoRef.current?.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks()
      tracks.forEach((track) => track.stop())
      videoRef.current.srcObject = null
    }
    if (videoRef.current && isDemoMode) {
      videoRef.current.pause()
      videoRef.current.removeAttribute('src')
      videoRef.current.load()
    }
    isPredictingRef.current = false
    setIsDemoMode(false)
    setIsRunning(false)
  }

  const startCamera = async () => {
    try {
      stopCamera()
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          ...(selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : {})
        },
        audio: false
      })
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        videoRef.current.onloadedmetadata = () => {
          if (!videoRef.current || !landmarkCanvasRef.current) return
          landmarkCanvasRef.current.width = videoRef.current.videoWidth || 640
          landmarkCanvasRef.current.height = videoRef.current.videoHeight || 480
        }
        setIsRunning(true)
      }
      await refreshVideoDevices()
    } catch (err) {
      console.error('Failed to access camera:', err)
    }
  }

  const startDemoVideo = async () => {
    stopCamera()
    if (!videoRef.current || !landmarkCanvasRef.current) return

    const video = videoRef.current
    video.srcObject = null
    video.src = selectedDemoVideo
    video.loop = true
    video.muted = true
    video.controls = false
    video.playsInline = true
    video.onloadedmetadata = () => {
      if (!landmarkCanvasRef.current) return
      landmarkCanvasRef.current.width = video.videoWidth || 640
      landmarkCanvasRef.current.height = video.videoHeight || 480
    }
    await video.play()
    setIsDemoMode(true)
    setIsRunning(true)
  }

  const stopVoiceVisualizer = () => {
    if (audioAnimationRef.current !== null) {
      cancelAnimationFrame(audioAnimationRef.current)
      audioAnimationRef.current = null
    }
    audioStreamRef.current?.getTracks().forEach((track) => track.stop())
    audioStreamRef.current = null
    audioContextRef.current?.close()
    audioContextRef.current = null
    setVoiceLevels(Array(48).fill(0.12))
  }

  const startVoiceVisualizer = async () => {
    stopVoiceVisualizer()
    if (!navigator.mediaDevices?.getUserMedia) return

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
    const AudioContextCtor = window.AudioContext || (window as any).webkitAudioContext
    const audioContext = new AudioContextCtor()
    const analyser = audioContext.createAnalyser()
    const source = audioContext.createMediaStreamSource(stream)

    analyser.fftSize = 256
    analyser.smoothingTimeConstant = 0.72
    source.connect(analyser)

    const data = new Uint8Array(analyser.frequencyBinCount)
    audioStreamRef.current = stream
    audioContextRef.current = audioContext

    const tick = () => {
      analyser.getByteFrequencyData(data)
      const bucketCount = 48
      const bucketSize = Math.floor(data.length / bucketCount)
      const nextLevels = Array.from({ length: bucketCount }, (_, index) => {
        const start = index * bucketSize
        const bucket = data.slice(start, start + bucketSize)
        const average = bucket.reduce((sum, value) => sum + value, 0) / Math.max(bucket.length, 1)
        return Math.max(0.12, Math.min(1, average / 160))
      })
      setVoiceLevels(nextLevels)
      audioAnimationRef.current = requestAnimationFrame(tick)
    }

    tick()
  }

  const startVoiceRecognition = async () => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SpeechRecognition) {
      alert('브라우저가 음성 인식을 지원하지 않습니다. Chrome 또는 Edge를 사용해 주세요.')
      return
    }

    try {
      await startVoiceVisualizer()
    } catch (err) {
      console.warn('Failed to start voice visualizer:', err)
    }

    const recognition = new SpeechRecognition()
    recognition.lang = 'ko-KR'
    recognition.continuous = true
    recognition.interimResults = false
    recognition.onstart = () => setIsVoiceActive(true)
    recognition.onresult = (event: any) => {
      const transcript = event.results[event.results.length - 1][0].transcript.trim()
      if (!transcript) return
      setMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}-${Math.random()}`,
          sender: 'speaker',
          text: transcript,
          timestamp: new Date(),
          avatar: '음성'
        }
      ])
    }
    recognition.onerror = (event: any) => console.error('Speech recognition error:', event.error)
    recognition.onend = () => {
      setIsVoiceActive(false)
      if (recognitionRef.current) recognition.start()
    }
    recognition.start()
    recognitionRef.current = recognition
  }

  const stopVoiceRecognition = () => {
    if (recognitionRef.current) {
      recognitionRef.current.stop()
      recognitionRef.current = null
    }
    stopVoiceVisualizer()
  }

  useEffect(() => {
    refreshVideoDevices()
    navigator.mediaDevices?.addEventListener?.('devicechange', refreshVideoDevices)
    return () => {
      stopCamera()
      recognitionRef.current?.stop()
      stopVoiceVisualizer()
      navigator.mediaDevices?.removeEventListener?.('devicechange', refreshVideoDevices)
    }
  }, [])

  const drawLandmarks = (ctx: CanvasRenderingContext2D, landmarks: any) => {
    if (!landmarks) return

    const overlay = landmarkCanvasRef.current
    const video = videoRef.current
    const width = overlay?.width || 640
    const height = overlay?.height || 480
    const videoWidth = video?.videoWidth || width
    const videoHeight = video?.videoHeight || height
    const scale = Math.min(width / videoWidth, height / videoHeight)
    const drawWidth = videoWidth * scale
    const drawHeight = videoHeight * scale
    const offsetX = (width - drawWidth) / 2
    const offsetY = (height - drawHeight) / 2
    const projectX = (x: number) => offsetX + x * drawWidth
    const projectY = (y: number) => offsetY + y * drawHeight

    ctx.clearRect(0, 0, width, height)

    const drawConnections = (points: any[], connections: number[][], color: string, radius: number) => {
      if (!points?.length) return
      ctx.strokeStyle = color
      ctx.fillStyle = color
      ctx.lineWidth = 2

      connections.forEach(([start, end]) => {
        if (points[start] && points[end]) {
          ctx.beginPath()
          ctx.moveTo(projectX(points[start][0]), projectY(points[start][1]))
          ctx.lineTo(projectX(points[end][0]), projectY(points[end][1]))
          ctx.stroke()
        }
      })

      points.forEach((point) => {
        ctx.beginPath()
        ctx.arc(projectX(point[0]), projectY(point[1]), radius, 0, 2 * Math.PI)
        ctx.fill()
      })
    }

    const handConnections = [
      [0, 1], [1, 2], [2, 3], [3, 4],
      [0, 5], [5, 6], [6, 7], [7, 8],
      [0, 9], [9, 10], [10, 11], [11, 12],
      [0, 13], [13, 14], [14, 15], [15, 16],
      [0, 17], [17, 18], [18, 19], [19, 20]
    ]
    const poseConnections = [
      [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
      [11, 23], [12, 24], [23, 24]
    ]

    drawConnections(landmarks.pose, poseConnections, '#38bdf8', 2)
    drawConnections(landmarks.left_hand, handConnections, '#22c55e', 3)
    drawConnections(landmarks.right_hand, handConnections, '#f97316', 3)
  }

  const captureAndSend = async () => {
    if (!videoRef.current || !canvasRef.current) return
    if (isPredictingRef.current) return
    isPredictingRef.current = true

    const ctx = canvasRef.current.getContext('2d')
    if (!ctx) {
      isPredictingRef.current = false
      return
    }

    const sourceWidth = videoRef.current.videoWidth || 640
    const sourceHeight = videoRef.current.videoHeight || 480
    const targetScale = Math.min(640 / sourceWidth, 480 / sourceHeight, 1)
    const targetWidth = Math.round(sourceWidth * targetScale)
    const targetHeight = Math.round(sourceHeight * targetScale)
    if (canvasRef.current.width !== targetWidth || canvasRef.current.height !== targetHeight) {
      canvasRef.current.width = targetWidth
      canvasRef.current.height = targetHeight
    }
    ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height)
    const frameId = nextFrameIdRef.current + 1
    nextFrameIdRef.current = frameId
    latestFrameIdRef.current = frameId

    canvasRef.current.toBlob(async (blob) => {
      if (!blob) {
        isPredictingRef.current = false
        return
      }

      const formData = new FormData()
      formData.append('frame', blob)
      formData.append('frame_id', frameId.toString())
      formData.append('model_type', modelType === 'cnn_gru' ? 'sequence' : 'lstm')
      formData.append('landmark_layout', 'mediapipe_xyz')
      formData.append('client_id', clientIdRef.current)
      formData.append('confidence_threshold', confidenceThreshold.toString())
      formData.append('window_size', windowSize.toString())
      formData.append('stable_min_count', stableMinCount.toString())
      formData.append('max_missing_frames', maxMissingFrames.toString())

      try {
        const response = await fetch('/api/predict', { method: 'POST', body: formData })
        const data = await response.json()
        const responseFrameId = Number(data.frame_id ?? data.prediction?.frame_id ?? frameId)
        if (responseFrameId < latestFrameIdRef.current) return
        if (!data.prediction) return

        if (data.prediction.landmarks) {
          const overlayCtx = landmarkCanvasRef.current?.getContext('2d')
          if (overlayCtx) drawLandmarks(overlayCtx, data.prediction.landmarks)
        }

        const pred: Prediction = {
          label: data.prediction.label,
          confidence: data.prediction.confidence,
          timestamp: Date.now(),
          has_hand: data.prediction.has_hand,
          window_filled: data.prediction.window_filled,
          window_progress: data.prediction.window_progress,
          window_size: data.prediction.window_size,
          missing_frames: data.prediction.missing_frames,
          max_missing_frames: data.prediction.max_missing_frames,
          top_predictions: data.prediction.top_predictions
        }

        setCurrentPrediction(pred)
        setPredictions((prev) => [...prev.slice(-9), pred])

        const now = Date.now()
        if (
          data.prediction.window_filled &&
          data.prediction.label &&
          data.prediction.confidence >= confidenceThreshold &&
          (lastAddedLabel !== data.prediction.label || now - lastAddedTime >= cooldownSeconds * 1000)
        ) {
          setMessages((prev) => [
            ...prev,
            {
              id: `${now}-${Math.random()}`,
              sender: 'deaf',
              text: data.prediction.label,
              timestamp: new Date(),
              avatar: '수어'
            }
          ])
          setLastAddedLabel(data.prediction.label)
          setLastAddedTime(now)
        }
      } catch (err) {
        console.error('Failed to send frame:', err)
      } finally {
        isPredictingRef.current = false
      }
    }, 'image/jpeg', 0.75)
  }

  useEffect(() => {
    if (!isRunning) return
    const interval = setInterval(captureAndSend, captureIntervalMs)
    return () => clearInterval(interval)
  }, [
    isRunning,
    modelType,
    confidenceThreshold,
    windowSize,
    stableMinCount,
    maxMissingFrames,
    captureIntervalMs,
    lastAddedLabel,
    lastAddedTime,
    cooldownSeconds
  ])

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const clearMessages = () => setMessages([])

  const getPredictionStatus = (prediction: Prediction) => {
    if (!prediction.has_hand) {
      const misses = prediction.missing_frames ?? 0
      const maxMisses = prediction.max_missing_frames ?? maxMissingFrames
      return misses <= maxMisses && (prediction.window_progress ?? 0) > 0
        ? `추적 보정 중 ${misses}/${maxMisses}`
        : '손을 카메라 안에 보여주세요'
    }
    if (!prediction.window_filled) {
      const progress = prediction.window_progress ?? 0
      const size = prediction.window_size ?? windowSize
      return `동작 수집 중 ${progress}/${size}`
    }
    if (!prediction.label) return '인식 불확실'
    return `${prediction.label} ${(prediction.confidence * 100).toFixed(0)}%`
  }

  const cameraStateLabel = showMjpegFeed ? 'MJPEG 확인 중' : isDemoMode ? '샘플 영상 재생 중' : isRunning ? '카메라 실행 중' : '카메라 대기'
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
      <header className="demo-header">
        <div>
          <p className="eyebrow">TEAM LEAD WEB DEMO</p>
          <h1>실시간 수어 인식 데모</h1>
        </div>
        <div className="status-bar" aria-label="demo status">
          <span className={`status-pill ${isRunning || showMjpegFeed ? 'on' : ''}`}>{cameraStateLabel}</span>
          <span className={`status-pill ${isVoiceActive ? 'on' : ''}`}>{voiceStateLabel}</span>
          <span className={`status-pill ${currentPrediction?.has_hand ? 'on' : ''}`}>{detectStateLabel}</span>
        </div>
      </header>

      <main className="demo-grid">
        <section className="video-card primary-feed">
          <div className="panel-heading">
            <div>
              <p className="panel-kicker">수어 입력</p>
              <h2>{showMjpegFeed ? 'Backend MJPEG smoke test' : isDemoMode ? 'Sample video input' : 'MediaPipe webcam input'}</h2>
            </div>
            <span className="spec-chip">mediapipe_xyz 225</span>
          </div>
          <div className="video-wrapper">
            {showMjpegFeed ? (
              <img src="/video_feed" className="camera-video" alt="Backend MJPEG stream" />
            ) : (
              <>
                <video ref={videoRef} autoPlay playsInline className="camera-video" />
                <canvas ref={canvasRef} width={640} height={480} className="hidden-canvas" />
                <canvas ref={landmarkCanvasRef} width={640} height={480} className="landmark-canvas" />
              </>
            )}
            {!showMjpegFeed && currentPrediction && (
              <div className="prediction-overlay">
                <span className="prediction-caption">현재 예측</span>
                <strong>{getPredictionStatus(currentPrediction)}</strong>
                {currentPrediction.top_predictions && currentPrediction.top_predictions.length > 0 && (
                  <div className="prediction-top-list">
                    {currentPrediction.top_predictions.map((item, index) => (
                      <span key={item.label}>
                        Top {index + 1} {item.label} {(item.confidence * 100).toFixed(0)}%
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </section>

        <section className="side-panel">
          <div className="voice-panel">
            <div className="panel-heading compact">
              <div>
                <p className="panel-kicker">음성 입력</p>
                <h2>발화 인식</h2>
              </div>
              <span className={`live-dot ${isVoiceActive ? 'active' : ''}`} />
            </div>
            <div className="voice-meter">
              {voiceLevels.map((level, index) => (
                <span
                  key={index}
                  className={isVoiceActive && level > 0.2 ? 'lit' : ''}
                  style={{
                    height: `${Math.round(level * 100)}%`,
                    opacity: isVoiceActive && level > 0.2 ? 0.58 + level * 0.42 : 0.24,
                    background: isVoiceActive && level > 0.2
                      ? voiceBarColors[index % voiceBarColors.length]
                      : 'linear-gradient(180deg, rgba(203, 213, 225, 0.28), rgba(100, 116, 139, 0.18))'
                  }}
                />
              ))}
            </div>
            <p>{isVoiceActive ? '말하는 사람의 음성을 대화창에 기록하고 있습니다.' : '음성 인식을 시작하면 발화가 오른쪽 메시지로 표시됩니다.'}</p>
          </div>

          <div className="chat-section">
            <div className="chat-header">
              <div>
                <p className="panel-kicker">Conversation</p>
                <h2>대화 기록</h2>
              </div>
              <button onClick={clearMessages} className="btn-clear-chat">초기화</button>
            </div>

            <div className="chat-messages">
              {messages.length === 0 && (
                <div className="chat-empty">
                  <strong>아직 대화가 없습니다.</strong>
                  <span>수어 예측 또는 음성 인식 결과가 여기에 쌓입니다.</span>
                </div>
              )}
              {messages.map((msg) => (
                <div key={msg.id} className={`chat-message ${msg.sender}`}>
                  <div className="message-avatar">{msg.avatar}</div>
                  <div className="message-bubble">
                    <div className="message-text">{msg.text}</div>
                    <div className="message-time">
                      {msg.timestamp.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' })}
                    </div>
                  </div>
                </div>
              ))}
              <div ref={chatEndRef} />
            </div>
          </div>
        </section>
      </main>

      <section className="controls-section">
        <div className="button-group">
          {!isRunning ? (
            <button onClick={startCamera} className="btn-primary" disabled={showMjpegFeed}>카메라 시작</button>
          ) : (
            <button onClick={stopCamera} className="btn-danger">카메라 중지</button>
          )}

          <button onClick={startDemoVideo} className="btn-secondary" disabled={showMjpegFeed}>
            샘플 영상 데모
          </button>

          {!isVoiceActive ? (
            <button onClick={startVoiceRecognition} className="btn-voice">음성 인식 시작</button>
          ) : (
            <button onClick={stopVoiceRecognition} className="btn-voice-active">음성 인식 중지</button>
          )}

          <button
            onClick={() => {
              stopCamera()
              setShowMjpegFeed((value) => !value)
            }}
            className="btn-secondary"
          >
            {showMjpegFeed ? 'MJPEG 끄기' : 'MJPEG 확인'}
          </button>

          <button onClick={() => setShowSettings(!showSettings)} className="btn-secondary">
            {showSettings ? '설정 닫기' : '설정 열기'}
          </button>
        </div>

        {showSettings && (
          <div className="settings-panel">
            <div className="setting-group">
              <label>카메라</label>
              <select value={selectedDeviceId} onChange={(e) => setSelectedDeviceId(e.target.value)} disabled={isRunning}>
                {videoDevices.length === 0 && <option value="">카메라 없음</option>}
                {videoDevices.map((device, index) => (
                  <option key={device.deviceId || index} value={device.deviceId}>
                    {device.label || `카메라 ${index + 1}`}
                  </option>
                ))}
              </select>
            </div>

            <div className="setting-group">
              <label>샘플 영상</label>
              <select value={selectedDemoVideo} onChange={(e) => setSelectedDemoVideo(e.target.value)} disabled={isDemoMode}>
                {demoVideos.map((video) => (
                  <option key={video.src} value={video.src}>
                    {video.label} - {video.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="setting-group">
              <label>모델</label>
              <select value={modelType} onChange={(e) => setModelType(e.target.value as 'cnn_gru' | 'lstm')}>
                <option value="cnn_gru">CNN-GRU</option>
                <option value="lstm">LSTM</option>
              </select>
            </div>

            <div className="setting-group">
              <label>신뢰도 임계값 {confidenceThreshold.toFixed(2)} 고정</label>
              <input type="range" min="0.3" max="0.95" step="0.05" value={confidenceThreshold} onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))} disabled />
            </div>

            <div className="setting-group">
              <label>윈도우 크기</label>
              <select value={windowSize} disabled>
                <option value={32}>32 frames 고정</option>
              </select>
            </div>

            <div className="setting-group">
              <label>전송 간격 {captureIntervalMs}ms 고정</label>
              <input type="range" min="100" max="600" step="50" value={captureIntervalMs} onChange={(e) => setCaptureIntervalMs(parseInt(e.target.value))} disabled />
            </div>

            <div className="setting-group">
              <label>손 미검출 허용 {maxMissingFrames}프레임 고정</label>
              <input type="range" min="0" max="8" step="1" value={maxMissingFrames} onChange={(e) => setMaxMissingFrames(parseInt(e.target.value))} disabled />
            </div>

            <div className="setting-group">
              <label>안정화 최소 개수 {stableMinCount} 고정</label>
              <input type="range" min="1" max="5" step="1" value={stableMinCount} onChange={(e) => setStableMinCount(parseInt(e.target.value))} disabled />
            </div>

            <div className="setting-group">
              <label>중복 방지 쿨다운 {cooldownSeconds}초</label>
              <input type="range" min="0.5" max="5" step="0.5" value={cooldownSeconds} onChange={(e) => setCooldownSeconds(parseFloat(e.target.value))} />
            </div>
          </div>
        )}
      </section>
    </div>
  )
}
