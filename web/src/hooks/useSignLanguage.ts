import { useEffect, useRef, useState, useCallback } from 'react'
import type { Prediction, ChatMessage } from '../types'

const VIDEO_WIDTH = 640
const VIDEO_HEIGHT = 480
const PREDICT_FRAME_WIDTH = 424
const PREDICT_FRAME_HEIGHT = 320
const JPEG_QUALITY = 0.72
const MODEL_INFERENCE_EVERY_N_FRAMES = 5
const DEMO_PLAYBACK_RATE = 0.65
const LIVE_GLOSS_IDLE_MS = 6500

export interface DemoClip {
  id: string
  src: string
}

export interface DemoScenario {
  displayText: string
  clips: DemoClip[]
  segmentBoundariesSec?: number[]
}

export const validationDemoScenarios: DemoScenario[] = [
  {
    displayText: '상처 + 붕대 + 원하다',
    segmentBoundariesSec: [4.4, 8.9],
    clips: [
      { id: 'merge1', src: 'data/raw/validation_mp4/merge1.mp4' },
    ],
  },
  {
    displayText: '다리 + 골절 + 아프다',
    segmentBoundariesSec: [3.47, 7.97],
    clips: [
      { id: 'merge2', src: 'data/raw/validation_mp4/merge2.mp4' },
    ],
  },
  {
    displayText: '소화불량 + 어떻게 + 치료',
    segmentBoundariesSec: [4.5, 7.233],
    clips: [
      { id: 'merge3', src: 'data/raw/validation_mp4/merge3.mp4' },
    ],
  },
  {
    displayText: '오른쪽 + 위 + 통증 + 못견디다',
    segmentBoundariesSec: [2.367, 5.5, 8.133],
    clips: [
      { id: 'mpeg4', src: 'data/raw/validation_mp4/mpeg4.mp4' },
    ],
  },
  {
    displayText: '골절 + 회복 + 얼마',
    segmentBoundariesSec: [3.833, 6.933],
    clips: [
      { id: 'mpeg5', src: 'data/raw/validation_mp4/mpeg5.mp4' },
    ],
  },
]

interface SignLanguageConfig {
  modelType?: 'cnn_gru' | 'lstm'
  confidenceThreshold?: number
  windowSize?: number
  stableMinCount?: number
  captureIntervalMs?: number
  maxMissingFrames?: number
}

export function useSignLanguage(
  onMessage: (msg: ChatMessage) => void,
  config: SignLanguageConfig = {}
) {
  const {
    modelType = 'cnn_gru',
    confidenceThreshold = 0.30,
    windowSize = 32,
    stableMinCount = 1,
    captureIntervalMs = 60,
    maxMissingFrames = 3,
  } = config

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const landmarkCanvasRef = useRef<HTMLCanvasElement>(null)
  
  const canvasCtxRef = useRef<CanvasRenderingContext2D | null>(null)
  const abortControllerRef = useRef<AbortController | null>(null)
  const onMessageRef = useRef(onMessage)
  const isMounted = useRef(true)

  const isPredictingRef = useRef(false)
  const nextFrameIdRef = useRef(0)
  const latestFrameIdRef = useRef(0)
  const clientIdRef = useRef<string>(Math.random().toString(36).substring(7))
  
  const peakGestureRef = useRef<{ label: string, confidence: number } | null>(null)
  const isHandUpRef = useRef(false)

  const isDemoModeRef = useRef(false)
  const activeStreamRef = useRef<MediaStream | null>(null)
  const demoScenarioRef = useRef<DemoScenario | null>(null)
  const demoClipIndexRef = useRef(0)
  const demoGlossBufferRef = useRef<string[]>([])
  const liveGlossBufferRef = useRef<string[]>([])
  const liveGlossTimerRef = useRef<number | null>(null)
  const demoMotionPrevRef = useRef<Uint8Array | null>(null)
  const demoPauseFramesRef = useRef(0)
  const demoSegmentStartTimeRef = useRef(0)
  const demoHandFramesSinceBoundaryRef = useRef(0)
  const demoNextBoundaryIndexRef = useRef(0)
  const demoSuppressUntilMotionRef = useRef(false)
  const demoBoundaryFinalizingRef = useRef(false)

  const [isRunning, setIsRunning] = useState(false)
  const [isDemoMode, setIsDemoMode] = useState(false)
  const [activeDemoLabel, setActiveDemoLabel] = useState('')
  const [camFps, setCamFps] = useState(0)
  const [sendFps, setSendFps] = useState(0)
  const camFrameCountRef = useRef(0)
  const camFpsLastTimeRef = useRef(0)
  const sendFrameCountRef = useRef(0)
  const sendFpsLastTimeRef = useRef(0)
  const camRafRef = useRef<number | null>(null)
  const [activeDemoClipLabel, setActiveDemoClipLabel] = useState('')
  const [currentPrediction, setCurrentPrediction] = useState<Prediction | null>(null)
  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([])
  const [selectedDeviceId, setSelectedDeviceId] = useState('')

  const selectedDeviceIdRef = useRef(selectedDeviceId)
  useEffect(() => { selectedDeviceIdRef.current = selectedDeviceId }, [selectedDeviceId])
  useEffect(() => { onMessageRef.current = onMessage }, [onMessage])

  useEffect(() => {
    isMounted.current = true
    return () => {
      isMounted.current = false
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
    }
  }, [])

  const refreshVideoDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return
    const devices = await navigator.mediaDevices.enumerateDevices()
    const cameras = devices.filter((d) => d.kind === 'videoinput')
    if (!isMounted.current) return
    
    setVideoDevices(cameras)
    if (!selectedDeviceIdRef.current && cameras[0]?.deviceId) {
      setSelectedDeviceId(cameras[0].deviceId)
    }
  }, [])

  const clearLandmarkOverlay = useCallback(() => {
    const canvas = landmarkCanvasRef.current
    const ctx = canvas?.getContext('2d')
    if (canvas && ctx) ctx.clearRect(0, 0, canvas.width, canvas.height)
  }, [])

  const resolveDemoSrc = (src: string) => {
    if (src.startsWith('/')) return src
    if (src.startsWith('data/raw/validation_mp4/')) {
      return `/demo-videos/${src.split('/').pop()}`
    }
    return src
  }

  const resetDemoInternalMotionState = useCallback(() => {
    demoMotionPrevRef.current = null
    demoPauseFramesRef.current = 0
    demoSegmentStartTimeRef.current = videoRef.current?.currentTime || 0
    demoHandFramesSinceBoundaryRef.current = 0
    demoNextBoundaryIndexRef.current = 0
    demoSuppressUntilMotionRef.current = false
    demoBoundaryFinalizingRef.current = false
  }, [])

  const convertLiveGlossToText = useCallback(async () => {
    const words = liveGlossBufferRef.current
    liveGlossBufferRef.current = []
    const gloss = words.join(' + ')
    if (!gloss) return

    try {
      const response = await fetch('/api/gloss_to_text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gloss, client_id: `live-gloss-${clientIdRef.current}` }),
      })
      const data = await response.json()
      onMessageRef.current({
        id: `${Date.now()}-live-gloss-${Math.random()}`,
        sender: 'patient',
        text: data.text || gloss,
        timestamp: new Date(),
        label: '수어 문장 변환',
      })
    } catch (err) {
      console.error('Live gloss-to-text failed:', err)
      onMessageRef.current({
        id: `${Date.now()}-live-gloss-fallback-${Math.random()}`,
        sender: 'patient',
        text: gloss,
        timestamp: new Date(),
        label: '수어 글로스',
      })
    }
  }, [])

  const commitRecognizedWord = useCallback((word: string) => {
    if (isDemoModeRef.current) {
      const prev = demoGlossBufferRef.current
      if (prev[prev.length - 1] !== word) {
        demoGlossBufferRef.current = [...prev, word]
      }
    } else {
      const prev = liveGlossBufferRef.current
      if (prev[prev.length - 1] !== word) {
        liveGlossBufferRef.current = [...prev, word]
      }
      if (liveGlossTimerRef.current) {
        window.clearTimeout(liveGlossTimerRef.current)
      }
      liveGlossTimerRef.current = window.setTimeout(() => {
        void convertLiveGlossToText()
      }, LIVE_GLOSS_IDLE_MS)
    }
  }, [convertLiveGlossToText])

  const convertDemoGlossToText = useCallback(async (_scenario: DemoScenario) => {
    const words = demoGlossBufferRef.current
    const gloss = words.join(' + ')
    if (!gloss) {
      onMessageRef.current({
        id: `${Date.now()}-demo-gloss-empty-${Math.random()}`,
        sender: 'doctor',
        text: '영상에서 예측된 gloss가 없어 자연어 변환을 실행하지 않았습니다.',
        timestamp: new Date(),
        label: '데모 인식 안내',
      })
      return
    }

    try {
      // 💡 동적 IP 자동 감지 적용
      const response = await fetch('/api/gloss_to_text', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gloss, client_id: `demo-gloss-${Date.now()}` }),
      })
      const data = await response.json()
      onMessageRef.current({
        id: `${Date.now()}-demo-gloss-${Math.random()}`,
        sender: 'patient',
        text: data.text || gloss,
        timestamp: new Date(),
        label: '데모 문장 변환',
      })
    } catch (err) {
      console.error('Demo gloss-to-text failed:', err)
      onMessageRef.current({
        id: `${Date.now()}-demo-gloss-fallback-${Math.random()}`,
        sender: 'patient',
        text: gloss,
        timestamp: new Date(),
        label: '데모 글로스',
      })
    }
  }, [])

  const finalizeDemoSegment = useCallback(async (forceFinalize = true) => {
    if (!isDemoModeRef.current || !forceFinalize) return

    const startedAt = Date.now()
    while (isPredictingRef.current && Date.now() - startedAt < 700) {
      await new Promise((resolve) => window.setTimeout(resolve, 40))
    }

    isPredictingRef.current = true
    try {
      const frameId = nextFrameIdRef.current + 1
      nextFrameIdRef.current = frameId
      latestFrameIdRef.current = frameId

      const formData = new FormData()
      formData.append('frame_id', frameId.toString())
      formData.append('model_type', modelType === 'cnn_gru' ? 'sequence' : 'lstm')
      formData.append('landmark_layout', 'mediapipe_xyz')
      formData.append('client_id', clientIdRef.current)
      formData.append('confidence_threshold', confidenceThreshold.toString())
      formData.append('window_size', windowSize.toString())
      formData.append('stable_min_count', stableMinCount.toString())
      formData.append('max_missing_frames', maxMissingFrames.toString())
      formData.append('min_segment_frames', '8')
      formData.append('run_model', 'true')
      formData.append('tta_enabled', 'false')
      formData.append('force_finalize', 'true')

      const res = await fetch('/api/predict', { method: 'POST', body: formData, credentials: 'include' })
      const data = await res.json().catch(() => ({}))
      const prediction = data.prediction
      if (!prediction) return

      const pred: Prediction = {
        label: prediction.label,
        confidence: prediction.confidence,
        timestamp: Date.now(),
        has_hand: prediction.has_hand,
        window_filled: prediction.window_filled,
        window_progress: prediction.window_progress,
        window_size: prediction.window_size,
        missing_frames: prediction.missing_frames,
        max_missing_frames: prediction.max_missing_frames,
        top_predictions: prediction.top_predictions,
      }
      setCurrentPrediction(pred)

      if (prediction.segment_finalized && pred.label && pred.confidence >= confidenceThreshold) {
        commitRecognizedWord(pred.label)
      }
    } catch (err) {
      console.error('Failed to finalize demo segment:', err)
    } finally {
      isPredictingRef.current = false
    }
  }, [
    commitRecognizedWord,
    confidenceThreshold,
    maxMissingFrames,
    modelType,
    stableMinCount,
    windowSize,
  ])

  const finalizeDemoInternalBoundary = useCallback(async () => {
    if (demoBoundaryFinalizingRef.current) return
    demoBoundaryFinalizingRef.current = true
    try {
      await finalizeDemoSegment(true)
      peakGestureRef.current = null
      isHandUpRef.current = false
    } finally {
      demoMotionPrevRef.current = null
      demoPauseFramesRef.current = 0
      demoSegmentStartTimeRef.current = videoRef.current?.currentTime || 0
      demoHandFramesSinceBoundaryRef.current = 0
      demoSuppressUntilMotionRef.current = !demoScenarioRef.current?.segmentBoundariesSec?.length
      demoBoundaryFinalizingRef.current = false
    }
  }, [finalizeDemoSegment])

  const stopCamera = useCallback(() => {
    const wasDemoMode = isDemoModeRef.current
    if (activeStreamRef.current) {
      activeStreamRef.current.getTracks().forEach((t) => t.stop())
      activeStreamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    if (isDemoModeRef.current && videoRef.current) {
      videoRef.current.pause()
      videoRef.current.removeAttribute('src')
      videoRef.current.load()
    }
    isPredictingRef.current = false
    isDemoModeRef.current = false
    demoScenarioRef.current = null
    demoClipIndexRef.current = 0
    demoGlossBufferRef.current = []
    resetDemoInternalMotionState()
    if (liveGlossTimerRef.current) {
      window.clearTimeout(liveGlossTimerRef.current)
      liveGlossTimerRef.current = null
    }
    if (!wasDemoMode && liveGlossBufferRef.current.length > 0) {
      void convertLiveGlossToText()
    }
    if (isMounted.current) {
      setIsDemoMode(false)
      setIsRunning(false)
      setActiveDemoLabel('')
      setActiveDemoClipLabel('')
      setCurrentPrediction(null)
    }
    clearLandmarkOverlay()
  }, [clearLandmarkOverlay, convertLiveGlossToText, resetDemoInternalMotionState])

  const startCamera = useCallback(async () => {
    stopCamera()
    try {
      const deviceId = selectedDeviceIdRef.current
      const videoConstraints: MediaTrackConstraints = {
        width: { ideal: VIDEO_WIDTH },
        height: { ideal: VIDEO_HEIGHT },
        frameRate: { ideal: 24, max: 30 },
        ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false,
      })

      activeStreamRef.current = stream

      const video = videoRef.current
      if (!video) {
        stream.getTracks().forEach((t) => t.stop())
        activeStreamRef.current = null
        return
      }

      video.srcObject = stream
      video.muted = true
      video.playsInline = true

      video.onloadedmetadata = () => {
        if (!landmarkCanvasRef.current) return
        landmarkCanvasRef.current.width = video.videoWidth || VIDEO_WIDTH
        landmarkCanvasRef.current.height = video.videoHeight || VIDEO_HEIGHT
      }

      await video.play()
      if (isMounted.current) setIsRunning(true)
      await refreshVideoDevices()
    } catch (err) {
      console.error('[startCamera] Failed:', err)
      if (activeStreamRef.current) {
        activeStreamRef.current.getTracks().forEach((t) => t.stop())
        activeStreamRef.current = null
      }
    }
  }, [stopCamera, refreshVideoDevices])

  const playDemoClip = useCallback(async (scenario: DemoScenario, clipIndex: number) => {
    const video = videoRef.current
    if (!video) return
    const clip = scenario.clips[clipIndex]
    if (!clip) return

    if (activeStreamRef.current) {
      activeStreamRef.current.getTracks().forEach((track) => track.stop())
      activeStreamRef.current = null
    }

    clearLandmarkOverlay()
    setCurrentPrediction(null)
    isPredictingRef.current = false
    resetDemoInternalMotionState()
    clientIdRef.current = `demo-${clip.id}-${Date.now()}`
    demoScenarioRef.current = scenario
    demoClipIndexRef.current = clipIndex
    isDemoModeRef.current = true

    video.srcObject = null
    video.src = resolveDemoSrc(clip.src)
    video.loop = false
    video.muted = true
    video.playsInline = true
    video.playbackRate = DEMO_PLAYBACK_RATE

    if (landmarkCanvasRef.current) {
      landmarkCanvasRef.current.width = video.videoWidth || VIDEO_WIDTH
      landmarkCanvasRef.current.height = video.videoHeight || VIDEO_HEIGHT
    }

    setIsDemoMode(true)
    setActiveDemoLabel(scenario.displayText)
    setActiveDemoClipLabel('')
    setIsRunning(true)

    await new Promise<void>((resolve) => {
      const done = () => {
        video.removeEventListener('loadeddata', done)
        video.removeEventListener('canplay', done)
        resolve()
      }
      video.addEventListener('loadeddata', done, { once: true })
      video.addEventListener('canplay', done, { once: true })
      video.load()
      window.setTimeout(done, 1000)
    })

    await video.play()
  }, [clearLandmarkOverlay, resetDemoInternalMotionState])

  const startDemoScenario = useCallback(async (scenario: DemoScenario) => {
    stopCamera()
    peakGestureRef.current = null
    isHandUpRef.current = false
    demoGlossBufferRef.current = []
    resetDemoInternalMotionState()
    await playDemoClip(scenario, 0)
  }, [playDemoClip, resetDemoInternalMotionState, stopCamera])

  const handleDemoVideoEnded = useCallback(async () => {
    if (!isDemoModeRef.current || !demoScenarioRef.current) return
    const scenario = demoScenarioRef.current
    const nextClipIndex = demoClipIndexRef.current + 1

    await finalizeDemoSegment(!!scenario.segmentBoundariesSec?.length)

    if (nextClipIndex < scenario.clips.length) {
      void playDemoClip(scenario, nextClipIndex)
      return
    }

    if (peakGestureRef.current) {
      commitRecognizedWord(peakGestureRef.current.label)
      peakGestureRef.current = null
    }

    void convertDemoGlossToText(scenario)
    isDemoModeRef.current = false
    demoScenarioRef.current = null
    demoClipIndexRef.current = 0
    resetDemoInternalMotionState()
    setIsRunning(false)
    setIsDemoMode(false)
    setActiveDemoLabel('')
    setActiveDemoClipLabel('')
    clearLandmarkOverlay()
  }, [clearLandmarkOverlay, commitRecognizedWord, convertDemoGlossToText, finalizeDemoSegment, playDemoClip, resetDemoInternalMotionState])

  const drawLandmarks = useCallback((ctx: CanvasRenderingContext2D, landmarks: any) => {
    if (!landmarks) return
    const overlay = landmarkCanvasRef.current
    const video = videoRef.current
    if (!overlay || !video) return

    const dpr = window.devicePixelRatio || 1
    const cssWidth = overlay.clientWidth || VIDEO_WIDTH
    const cssHeight = overlay.clientHeight || VIDEO_HEIGHT
    const canvasWidth = Math.round(cssWidth * dpr)
    const canvasHeight = Math.round(cssHeight * dpr)

    if (overlay.width !== canvasWidth || overlay.height !== canvasHeight) {
      overlay.width = canvasWidth
      overlay.height = canvasHeight
    }

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, cssWidth, cssHeight)
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    const videoWidth = video.videoWidth || VIDEO_WIDTH
    const videoHeight = video.videoHeight || VIDEO_HEIGHT
    const scale = Math.min(cssWidth / videoWidth, cssHeight / videoHeight)
    const drawWidth = videoWidth * scale
    const drawHeight = videoHeight * scale
    const offsetX = (cssWidth - drawWidth) / 2
    const offsetY = (cssHeight - drawHeight) / 2
    const px = (x: number) => offsetX + x * drawWidth
    const py = (y: number) => offsetY + y * drawHeight

    const drawConnections = (points: any[], connections: number[][], color: string, radius: number, lineWidth = 2) => {
      if (!points?.length) return
      ctx.strokeStyle = color
      ctx.fillStyle = color
      ctx.lineWidth = lineWidth
      connections.forEach(([s, e]) => {
        if (points[s] && points[e]) {
          ctx.beginPath()
          ctx.moveTo(px(points[s][0]), py(points[s][1]))
          ctx.lineTo(px(points[e][0]), py(points[e][1]))
          ctx.stroke()
        }
      })
      points.forEach((p) => {
        ctx.beginPath()
        ctx.arc(px(p[0]), py(p[1]), radius, 0, 2 * Math.PI)
        ctx.fill()
      })
    }

    const handConnections = [
      [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
      [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
      [0,17],[17,18],[18,19],[19,20],
    ]
    const poseConnections = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24],[23,24]]

    drawConnections(landmarks.pose, poseConnections, 'rgba(56,189,248,0.75)', 2, 2)
    drawConnections(landmarks.left_hand, handConnections, '#22c55e', 3, 2.5)
    drawConnections(landmarks.right_hand, handConnections, '#f97316', 3, 2.5)
  }, [])

  const detectDemoInternalBoundary = useCallback((
    _ctx: CanvasRenderingContext2D,
    _width: number,
    _height: number,
  ): 'send' | 'suppress' | 'finalize' => {
    const scenario = demoScenarioRef.current
    const video = videoRef.current
    if (!isDemoModeRef.current || !scenario || scenario.clips.length !== 1 || !video) {
      return 'send'
    }

    const nextBoundary = scenario.segmentBoundariesSec?.[demoNextBoundaryIndexRef.current]
    if (nextBoundary !== undefined && !demoBoundaryFinalizingRef.current && video.currentTime >= nextBoundary) {
      demoNextBoundaryIndexRef.current += 1
      demoMotionPrevRef.current = null
      demoPauseFramesRef.current = 0
      return 'finalize'
    }

    return 'send'
  }, [])

  const captureAndSend = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return
    if (videoRef.current.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return
    if (isPredictingRef.current) return
    isPredictingRef.current = true

    sendFrameCountRef.current++
    const sendNow = performance.now()
    const sendElapsed = sendNow - sendFpsLastTimeRef.current
    if (sendElapsed >= 500) {
      setSendFps(Math.round(sendFrameCountRef.current / (sendElapsed / 1000)))
      sendFrameCountRef.current = 0
      sendFpsLastTimeRef.current = sendNow
    }

    if (!canvasCtxRef.current) {
      canvasCtxRef.current = canvasRef.current.getContext('2d')
    }
    const ctx = canvasCtxRef.current
    if (!ctx) { isPredictingRef.current = false; return }

    const sourceWidth = videoRef.current.videoWidth || VIDEO_WIDTH
    const sourceHeight = videoRef.current.videoHeight || VIDEO_HEIGHT
    const targetScale = Math.min(PREDICT_FRAME_WIDTH / sourceWidth, PREDICT_FRAME_HEIGHT / sourceHeight, 1)
    const targetWidth = Math.round(sourceWidth * targetScale)
    const targetHeight = Math.round(sourceHeight * targetScale)
    
    if (canvasRef.current.width !== targetWidth || canvasRef.current.height !== targetHeight) {
      canvasRef.current.width = targetWidth
      canvasRef.current.height = targetHeight
    }
    ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height)

    const demoBoundaryAction = detectDemoInternalBoundary(ctx, canvasRef.current.width, canvasRef.current.height)
    if (demoBoundaryAction === 'suppress') {
      isPredictingRef.current = false
      return
    }
    if (demoBoundaryAction === 'finalize') {
      isPredictingRef.current = false
      void finalizeDemoInternalBoundary()
      return
    }

    const useInternalDemoSegmentation = isDemoModeRef.current && demoScenarioRef.current?.clips.length === 1
    const frameId = nextFrameIdRef.current + 1
    nextFrameIdRef.current = frameId
    latestFrameIdRef.current = frameId

    canvasRef.current.toBlob(async (blob) => {
      if (!blob) { isPredictingRef.current = false; return }
      const formData = new FormData()
      formData.append('frame', blob)
      formData.append('frame_id', frameId.toString())
      formData.append('model_type', modelType === 'cnn_gru' ? 'sequence' : 'lstm')
      formData.append('landmark_layout', 'mediapipe_xyz')
      formData.append('client_id', clientIdRef.current)
      formData.append('confidence_threshold', confidenceThreshold.toString())
      formData.append('window_size', windowSize.toString())
      formData.append('stable_min_count', stableMinCount.toString())
      formData.append('max_missing_frames', useInternalDemoSegmentation ? '999' : maxMissingFrames.toString())
      formData.append('min_segment_frames', '8')
      formData.append('tta_enabled', 'false')
      formData.append('run_model', (frameId % MODEL_INFERENCE_EVERY_N_FRAMES === 0).toString())

      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
      }
      abortControllerRef.current = new AbortController()

      try {
        // 💡 동적 IP 자동 감지 적용
        const res = await fetch('/api/predict', { 
          method: 'POST', 
          body: formData,
          credentials: 'include',
          signal: abortControllerRef.current.signal 
        })
        const data = await res.json()
        
        if (!isMounted.current) return

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
          top_predictions: data.prediction.top_predictions,
        }
        
        setCurrentPrediction(pred)
        if (data.prediction.segment_finalized) {
          isHandUpRef.current = false
          peakGestureRef.current = null
          if (pred.label && pred.confidence >= confidenceThreshold) {
            commitRecognizedWord(pred.label)
          }
          return
        }

        if (useInternalDemoSegmentation) return

        if (pred.has_hand) {
          isHandUpRef.current = true;
          
          if (pred.window_filled && pred.label && pred.confidence >= confidenceThreshold) {
            if (peakGestureRef.current && peakGestureRef.current.label !== pred.label) {
              commitRecognizedWord(peakGestureRef.current.label)
              peakGestureRef.current = { label: pred.label, confidence: pred.confidence };
            } else {
              if (!peakGestureRef.current || pred.confidence > peakGestureRef.current.confidence) {
                peakGestureRef.current = { label: pred.label, confidence: pred.confidence };
              }
            }
          } else {
            if (peakGestureRef.current) {
              commitRecognizedWord(peakGestureRef.current.label)
              peakGestureRef.current = null;
            }
          }
        } else {
          if (isHandUpRef.current) {
            isHandUpRef.current = false;
            if (peakGestureRef.current) {
              commitRecognizedWord(peakGestureRef.current.label)
              peakGestureRef.current = null;
            }
          }
        }
        
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          console.error('Failed to send frame:', err)
        }
      } finally {
        isPredictingRef.current = false
      }
    }, 'image/jpeg', JPEG_QUALITY)
  }, [modelType, confidenceThreshold, windowSize, stableMinCount, maxMissingFrames, commitRecognizedWord, detectDemoInternalBoundary, drawLandmarks, finalizeDemoInternalBoundary])

  useEffect(() => {
    refreshVideoDevices()
    navigator.mediaDevices?.addEventListener?.('devicechange', refreshVideoDevices)
    return () => {
      stopCamera()
      navigator.mediaDevices?.removeEventListener?.('devicechange', refreshVideoDevices)
    }
  }, [refreshVideoDevices, stopCamera])

  useEffect(() => {
    if (!isRunning) return
    const interval = setInterval(captureAndSend, captureIntervalMs)
    return () => clearInterval(interval)
  }, [isRunning, captureAndSend, captureIntervalMs])

  useEffect(() => {
    if (!isRunning) {
      if (camRafRef.current) cancelAnimationFrame(camRafRef.current)
      setCamFps(0)
      setSendFps(0)
      sendFrameCountRef.current = 0
      return
    }
    camFrameCountRef.current = 0
    camFpsLastTimeRef.current = performance.now()
    sendFrameCountRef.current = 0
    sendFpsLastTimeRef.current = performance.now()

    const tick = (ts: number) => {
      camFrameCountRef.current++
      const elapsed = ts - camFpsLastTimeRef.current
      if (elapsed >= 500) {
        setCamFps(Math.round(camFrameCountRef.current / (elapsed / 1000)))
        camFrameCountRef.current = 0
        camFpsLastTimeRef.current = ts
      }
      camRafRef.current = requestAnimationFrame(tick)
    }
    camRafRef.current = requestAnimationFrame(tick)
    return () => { if (camRafRef.current) cancelAnimationFrame(camRafRef.current) }
  }, [isRunning])

  const getPredictionStatus = (prediction: Prediction): string => {
    if (prediction.window_filled) {
      if (!prediction.label) return '인식 불확실'
      return `인식 중... ${prediction.label}`
    }
    if (!prediction.has_hand) {
      const misses = prediction.missing_frames ?? 0
      const maxMisses = prediction.max_missing_frames ?? maxMissingFrames
      return misses <= maxMisses && (prediction.window_progress ?? 0) > 0
        ? `추적 보정 중...`
        : '손을 카메라 안에 보여주세요'
    }
    const progress = prediction.window_progress ?? 0
    const size = prediction.window_size ?? windowSize
    return `동작 수집 중...`
  }

  return {
    videoRef, canvasRef, landmarkCanvasRef,
    isRunning, isDemoMode, activeDemoLabel, activeDemoClipLabel, currentPrediction,
    videoDevices, selectedDeviceId, setSelectedDeviceId,
    startCamera, stopCamera, startDemoScenario, handleDemoVideoEnded, getPredictionStatus,
    camFps, sendFps,
  }
}
