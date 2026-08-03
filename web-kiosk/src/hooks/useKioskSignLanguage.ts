import { useEffect, useRef, useState, useCallback } from 'react'
import type { Prediction, ChatMessage } from '../types'

const VIDEO_WIDTH = 1280
const VIDEO_HEIGHT = 720
const PREDICT_FRAME_WIDTH = 640
const PREDICT_FRAME_HEIGHT = 360
const JPEG_QUALITY = 0.72
const MODEL_INFERENCE_EVERY_N_FRAMES = 5
const LIVE_GLOSS_IDLE_MS = 6500

// 랜드마크 연결선 (MediaPipe Hands 21포인트)
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
]

function drawLandmarks(
  canvas: HTMLCanvasElement,
  videoWidth: number,
  videoHeight: number,
  landmarks: Record<string, unknown>,
) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  const scaleX = canvas.width / videoWidth
  const scaleY = canvas.height / videoHeight

  // 손 랜드마크 (left_hand, right_hand 각 21포인트 배열)
  for (const key of ['left_hand', 'right_hand']) {
    const pts = landmarks[key] as [number, number][] | undefined
    if (!pts || !Array.isArray(pts) || pts.length < 21) continue

    const color = key === 'right_hand' ? '#14b8a6' : '#818cf8'

    // 연결선
    ctx.strokeStyle = color
    ctx.lineWidth = 2
    ctx.globalAlpha = 0.7
    for (const [a, b] of HAND_CONNECTIONS) {
      const [ax, ay] = pts[a]
      const [bx, by] = pts[b]
      ctx.beginPath()
      ctx.moveTo(ax * scaleX, ay * scaleY)
      ctx.lineTo(bx * scaleX, by * scaleY)
      ctx.stroke()
    }

    // 관절 점
    ctx.globalAlpha = 1
    for (let i = 0; i < pts.length; i++) {
      const [x, y] = pts[i]
      ctx.beginPath()
      ctx.arc(x * scaleX, y * scaleY, i === 0 ? 5 : 3, 0, Math.PI * 2)
      ctx.fillStyle = i === 0 ? '#ffffff' : color
      ctx.fill()
    }
  }
  ctx.globalAlpha = 1
}

export function useKioskSignLanguage(onMessage: (msg: ChatMessage) => void) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const landmarkCanvasRef = useRef<HTMLCanvasElement>(null)
  const canvasCtxRef = useRef<CanvasRenderingContext2D | null>(null)

  const isMounted = useRef(true)
  const clientIdRef = useRef(Math.random().toString(36).substring(7))
  const isPredictingRef = useRef(false)
  const nextFrameIdRef = useRef(0)
  const latestFrameIdRef = useRef(0)
  const activeStreamRef = useRef<MediaStream | null>(null)
  const camRafRef = useRef<number | null>(null)

  const liveGlossBufferRef = useRef<string[]>([])
  const liveGlossTimerRef = useRef<number | null>(null)
  const onMessageRef = useRef(onMessage)

  const [isRunning, setIsRunning] = useState(false)
  const isRunningRef = useRef(false)
  const [currentPrediction, setCurrentPrediction] = useState<Prediction | null>(null)
  const [videoDevices, setVideoDevices] = useState<MediaDeviceInfo[]>([])
  const [selectedDeviceId, setSelectedDeviceId] = useState('')
  const selectedDeviceIdRef = useRef(selectedDeviceId)

  useEffect(() => { onMessageRef.current = onMessage }, [onMessage])
  useEffect(() => { isRunningRef.current = isRunning }, [isRunning])
  useEffect(() => { selectedDeviceIdRef.current = selectedDeviceId }, [selectedDeviceId])
  useEffect(() => {
    isMounted.current = true
    return () => { isMounted.current = false }
  }, [])

  // ── 카메라 목록 ───────────────────────────────────────
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

  // ── 글로스 → 자연어 변환 ──────────────────────────────
  const convertLiveGlossToText = useCallback(async () => {
    const words = liveGlossBufferRef.current
    liveGlossBufferRef.current = []
    if (!words.length) return

    try {
      const res = await fetch('/api/gloss_to_text', {
        method: 'POST',
        credentials: 'include',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ glosses: words }),
      })
      const data = await res.json()
      onMessageRef.current({
        id: `${Date.now()}-gloss-${Math.random()}`,
        sender: 'citizen',
        text: data.text || words.join(' + '),
        timestamp: new Date(),
        label: '수어 문장 변환',
      })
    } catch {
      onMessageRef.current({
        id: `${Date.now()}-gloss-fallback-${Math.random()}`,
        sender: 'citizen',
        text: words.join(' + '),
        timestamp: new Date(),
        label: '수어 글로스',
      })
    }
  }, [])

  const commitWord = useCallback((word: string) => {
    const prev = liveGlossBufferRef.current
    if (prev[prev.length - 1] !== word) {
      liveGlossBufferRef.current = [...prev, word]
    }
    if (liveGlossTimerRef.current) window.clearTimeout(liveGlossTimerRef.current)
    liveGlossTimerRef.current = window.setTimeout(() => void convertLiveGlossToText(), LIVE_GLOSS_IDLE_MS)
  }, [convertLiveGlossToText])

  // ── 예측 결과 처리 + 랜드마크 드로잉 ─────────────────
  const handlePrediction = useCallback((prediction: Prediction) => {
    if (!isMounted.current) return
    setCurrentPrediction(prediction)

    // 랜드마크 그리기
    const lc = landmarkCanvasRef.current
    const video = videoRef.current
    if (lc && video) {
      lc.width  = video.clientWidth  || PREDICT_FRAME_WIDTH
      lc.height = video.clientHeight || PREDICT_FRAME_HEIGHT
      const lms = (prediction as any)?.landmarks
      if (lms) {
        drawLandmarks(lc, PREDICT_FRAME_WIDTH, PREDICT_FRAME_HEIGHT, lms)
      } else {
        lc.getContext('2d')?.clearRect(0, 0, lc.width, lc.height)
      }
    }

    const label = prediction.display_label ?? prediction.label
    if (!label || (prediction.confidence ?? 0) < 0.30) return

    const scenarioText = String(prediction.scenario_text || '').trim()
    if (scenarioText) {
      liveGlossBufferRef.current = []
      if (liveGlossTimerRef.current) {
        window.clearTimeout(liveGlossTimerRef.current)
        liveGlossTimerRef.current = null
      }
      onMessageRef.current({
        id: `${Date.now()}-scenario-${Math.random()}`,
        sender: 'citizen',
        text: scenarioText,
        timestamp: new Date(),
        label: '수어 문장 인식',
      })
      return
    }

    commitWord(label)
  }, [commitWord])

  // ── 프레임 전송 ───────────────────────────────────────
  const sendFrame = useCallback(async (frameId: number) => {
    if (!isRunningRef.current || !isMounted.current) return
    if (isPredictingRef.current && frameId < latestFrameIdRef.current) return

    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.readyState < 2) return

    const ctx = canvasCtxRef.current ?? canvas.getContext('2d')
    if (!ctx) return
    canvasCtxRef.current = ctx

    canvas.width  = PREDICT_FRAME_WIDTH
    canvas.height = PREDICT_FRAME_HEIGHT
    // 전면 카메라 좌우 반전 보정 (미러링 해제 → MediaPipe 정확도 향상)
    ctx.save()
    ctx.scale(-1, 1)
    ctx.drawImage(video, -PREDICT_FRAME_WIDTH, 0, PREDICT_FRAME_WIDTH, PREDICT_FRAME_HEIGHT)
    ctx.restore()

    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, 'image/jpeg', JPEG_QUALITY),
    )
    if (!blob || !isRunningRef.current || !isMounted.current) return

    latestFrameIdRef.current = frameId
    isPredictingRef.current = true

    try {
      const formData = new FormData()
      formData.append('frame', blob, 'frame.jpg')
      formData.append('frame_id', String(frameId))
      formData.append('client_id', clientIdRef.current)
      formData.append('model_type', 'cnn_gru')

      const res = await fetch('/api/predict', {
        method: 'POST',
        credentials: 'include',
        body: formData,
      })
      if (!res.ok || !isMounted.current) return
      const json = await res.json()
      // /api/predict 응답은 { prediction: {...}, frame_id: ... } 구조
      const prediction: Prediction = json.prediction ?? json
      if (!prediction) return
      handlePrediction(prediction)
    } catch {
      // 무시
    } finally {
      isPredictingRef.current = false
    }
  }, [handlePrediction])

  // ── RAF 루프 ──────────────────────────────────────────
  const startCameraLoop = useCallback(() => {
    let frameCount = 0
    const loop = () => {
      if (!isRunningRef.current) return
      frameCount++
      if (frameCount % MODEL_INFERENCE_EVERY_N_FRAMES === 0) {
        void sendFrame(nextFrameIdRef.current++)
      }
      camRafRef.current = requestAnimationFrame(loop)
    }
    camRafRef.current = requestAnimationFrame(loop)
  }, [sendFrame])

  // ── 카메라 시작 (스트림을 외부에도 제공) ─────────────
  const startCamera = useCallback(async () => {
    await refreshVideoDevices()

    const constraints: MediaStreamConstraints = {
      video: {
        deviceId: selectedDeviceIdRef.current
          ? { exact: selectedDeviceIdRef.current }
          : undefined,
        width:  { ideal: VIDEO_WIDTH },
        height: { ideal: VIDEO_HEIGHT },
        facingMode: 'user',
      },

    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      activeStreamRef.current = stream

      const video = videoRef.current
      if (!video || !isMounted.current) {
        stream.getTracks().forEach((t) => t.stop())
        return
      }
      video.srcObject = stream
      await video.play()
      setIsRunning(true)
      isRunningRef.current = true
      startCameraLoop()
    } catch (err) {
      console.error('[useKioskSignLanguage] 카메라 시작 실패:', err)
    }
  }, [refreshVideoDevices, startCameraLoop])

  // ── 카메라 중지 ───────────────────────────────────────
  const stopCamera = useCallback(() => {
    isRunningRef.current = false
    setIsRunning(false)
    if (camRafRef.current) cancelAnimationFrame(camRafRef.current)
    activeStreamRef.current?.getTracks().forEach((t) => t.stop())
    activeStreamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    liveGlossBufferRef.current = []
    if (liveGlossTimerRef.current) window.clearTimeout(liveGlossTimerRef.current)
  }, [])

  const getPredictionStatus = useCallback((p: Prediction | null) => {
    if (!p) return '대기 중'
    if (!p.has_hand) return '손 없음'
    const label = p.display_label ?? p.label
    if (!label) return '인식 중...'
    return `${label} (${Math.round((p.confidence ?? 0) * 100)}%)`
  }, [])

  // 현재 활성 스트림 반환 (WebRTC에서 사용)
  const getActiveStream = useCallback(() => activeStreamRef.current, [])

  useEffect(() => () => { stopCamera() }, [stopCamera])

  return {
    videoRef,
    canvasRef,
    landmarkCanvasRef,
    isRunning,
    currentPrediction,
    videoDevices,
    selectedDeviceId,
    setSelectedDeviceId,
    startCamera,
    stopCamera,
    getPredictionStatus,
    getActiveStream,   // WebRTC 스트림 주입용
  }
}