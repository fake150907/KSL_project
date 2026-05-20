import { useEffect, useRef, useState, useCallback } from 'react'
import type { Prediction, ChatMessage } from '../types'
import { getClientMediaPipeRuntime } from '../mediapipeClient'

const VIDEO_WIDTH = 1280
const VIDEO_HEIGHT = 720
const PREDICT_FRAME_WIDTH = 640
const PREDICT_FRAME_HEIGHT = 360
const JPEG_QUALITY = 0.72
const MODEL_INFERENCE_EVERY_N_FRAMES = 5
const DEMO_PLAYBACK_RATE = 0.65
const LIVE_GLOSS_IDLE_MS = 6500
const DEMO_INTERNAL_PAUSE_DIFF_THRESHOLD = 0.2
const DEMO_INTERNAL_PAUSE_RESUME_THRESHOLD = 0.35
const DEMO_INTERNAL_PAUSE_MIN_FRAMES = 2
const DEMO_INTERNAL_RESUME_MIN_FRAMES = 3
const DEMO_INTERNAL_SUPPRESS_MIN_SECONDS = 0.25
const DEMO_INTERNAL_SEGMENT_MIN_SECONDS = 2.0
const DEMO_INTERNAL_SEGMENT_MIN_HAND_FRAMES = 45
const DEMO_MIRROR_PAUSE_DIFF_THRESHOLD = 0.55
const DEMO_MIRROR_SEGMENT_MIN_HAND_FRAMES = 35
const DEMO_MOTION_SAMPLE_STEP = 8

const LIVE_PAUSE_VELOCITY_THRESHOLD = 0.015
const LIVE_RESUME_VELOCITY_THRESHOLD = 0.06
const LIVE_PAUSE_MIN_FRAMES = 3
const LIVE_RESUME_MIN_FRAMES = 3
const LIVE_SEGMENT_MIN_FRAMES = 12
const LIVE_SEGMENT_MAX_FRAMES = 96
const LIVE_SHOULDER_WIDTH_FALLBACK = 0.2
const LIVE_SHOULDER_WIDTH_MIN = 0.02
const LIVE_HANDS_DOWN_REST_SECONDS = 0.25
const LIVE_HANDS_DOWN_REST_VELOCITY_THRESHOLD = 0.020
export const MEDIAPIPE_MODE_STORAGE_KEY = 'sign-lang-mediapipe-mode'
export const SCENARIO_MODE_STORAGE_KEY = 'sign-lang-scenario-mode'

export type LiveSegmentState =
  | 'idle'
  | 'waiting_for_motion'
  | 'collecting'
  | 'about_to_finalize'

export interface LiveSegmentStatus {
  state: LiveSegmentState
  segmentFrameCount: number
  segmentMinFrames: number
  segmentMaxFrames: number
  pauseFrames: number
  pauseMinFrames: number
  resumeFrames: number
  resumeMinFrames: number
  velocity: number | null
  pauseThreshold: number
  resumeThreshold: number
  handsDownElapsedSec: number
  handsDownRequiredSec: number
  lastFinalize: {
    reason: string
    label: string | null
    confidence: number
    segmentFrames: number | null
    at: number
  } | null
  segmentCount: number
}

const INITIAL_LIVE_SEGMENT_STATUS: LiveSegmentStatus = {
  state: 'waiting_for_motion',
  segmentFrameCount: 0,
  segmentMinFrames: LIVE_SEGMENT_MIN_FRAMES,
  segmentMaxFrames: LIVE_SEGMENT_MAX_FRAMES,
  pauseFrames: 0,
  pauseMinFrames: LIVE_PAUSE_MIN_FRAMES,
  resumeFrames: 0,
  resumeMinFrames: LIVE_RESUME_MIN_FRAMES,
  velocity: null,
  pauseThreshold: LIVE_PAUSE_VELOCITY_THRESHOLD,
  resumeThreshold: LIVE_RESUME_VELOCITY_THRESHOLD,
  handsDownElapsedSec: 0,
  handsDownRequiredSec: LIVE_HANDS_DOWN_REST_SECONDS,
  lastFinalize: null,
  segmentCount: 0,
}

const LIVE_NO_HAND_GRACE_FRAMES = 3

export interface DemoClip {
  id: string
  src: string
}

type DemoSegmentation =
  | 'auto'
  | {
      mode: 'manual'
      boundariesSec: number[]
    }

export interface DemoScenario {
  displayText: string
  clips: DemoClip[]
  segmentation?: DemoSegmentation
  mirrorForPrediction?: boolean
  relaxedSegmentation?: boolean
  forceScenarioMode?: boolean
}

export const validationDemoScenarios: DemoScenario[] = [
  {
    displayText: 'REALZ01 복지카드 분실/재발급 시나리오',
    segmentation: {
      mode: 'manual',
      boundariesSec: [3.767, 8.267, 11.767, 16.267, 23.767, 30.533, 34.533, 38.800, 44.300, 47.567],
    },
    relaxedSegmentation: true,
    clips: [
      { id: 'resident_realz01_scenario', src: 'data/raw/validation_mp4/resident_realz01_scenario.mp4' },
    ],
  },
  {
    displayText: 'REALZ03 01 hello',
    forceScenarioMode: true,
    segmentation: 'auto',
    relaxedSegmentation: true,
    clips: [
      { id: 'resident_realz03_01_hello', src: 'data/raw/validation_mp4/resident_realz03_01_hello.mp4' },
    ],
  },
  {
    displayText: 'REALZ03 02 welfare card + lost',
    forceScenarioMode: true,
    segmentation: {
      mode: 'manual',
      boundariesSec: [7.068],
    },
    relaxedSegmentation: true,
    clips: [
      { id: 'resident_realz03_02_welfare_card_lost', src: 'data/raw/validation_mp4/resident_realz03_02_welfare_card_lost.mp4' },
    ],
  },
  {
    displayText: 'REALZ03 03 possible + exemption',
    forceScenarioMode: true,
    segmentation: {
      mode: 'manual',
      boundariesSec: [5.732],
    },
    relaxedSegmentation: true,
    clips: [
      { id: 'resident_realz03_03_reissue_possible', src: 'data/raw/validation_mp4/resident_realz03_03_reissue_possible.mp4' },
    ],
  },
  {
    displayText: 'REALZ03 04 ID + here',
    forceScenarioMode: true,
    segmentation: {
      mode: 'manual',
      boundariesSec: [5.067],
    },
    relaxedSegmentation: true,
    clips: [
      { id: 'resident_realz03_04_id_here', src: 'data/raw/validation_mp4/resident_realz03_04_id_here.mp4' },
    ],
  },
  {
    displayText: 'REALZ03 05 subway lost',
    forceScenarioMode: true,
    segmentation: 'auto',
    relaxedSegmentation: true,
    clips: [
      { id: 'resident_realz03_05_subway_lost', src: 'data/raw/validation_mp4/resident_realz03_05_subway_lost.mp4' },
    ],
  },
  {
    displayText: '상처 + 붕대 + 원하다',
    segmentation: 'auto',
    clips: [
      { id: 'merge1', src: 'data/raw/validation_mp4/merge1.mp4' },
    ],
  },
  {
    displayText: '다리 + 골절 + 아프다',
    segmentation: 'auto',
    clips: [
      { id: 'merge2', src: 'data/raw/validation_mp4/merge2.mp4' },
    ],
  },
  {
    displayText: '소화불량 + 어떻게 + 치료',
    segmentation: 'auto',
    clips: [
      { id: 'merge3', src: 'data/raw/validation_mp4/merge3.mp4' },
    ],
  },
  {
    displayText: '오른쪽 + 위 + 통증 + 못견디다',
    segmentation: 'auto',
    clips: [
      { id: 'mpeg4', src: 'data/raw/validation_mp4/mpeg4.mp4' },
    ],
  },
  {
    displayText: '골절 + 회복 + 얼마',
    segmentation: 'auto',
    clips: [
      { id: 'mpeg5', src: 'data/raw/validation_mp4/mpeg5.mp4' },
    ],
  },
  {
    displayText: 'GY1 보정: 상처 + 붕대 + 원하다',
    segmentation: 'auto',
    relaxedSegmentation: true,
    clips: [
      { id: 'gy1_2_corrected', src: 'data/raw/validation_mp4/gy1_2_corrected.mp4' },
    ],
  },
  {
    displayText: 'GY1-3 보정: 상처 + 붕대 + 원하다',
    segmentation: {
      mode: 'manual',
      boundariesSec: [1.9, 6.25],
    },
    relaxedSegmentation: true,
    clips: [
      { id: 'gy1_3_corrected', src: 'data/raw/validation_mp4/gy1_3_corrected.mp4' },
    ],
  },
  {
    displayText: 'GY2 보정: 다리 + 골절 + 아프다',
    segmentation: 'auto',
    relaxedSegmentation: true,
    clips: [
      { id: 'gy2_2_corrected', src: 'data/raw/validation_mp4/gy2_2_corrected.mp4' },
    ],
  },
  {
    displayText: 'GY3 보정: 소화불량 + 어떻게 + 치료',
    segmentation: 'auto',
    relaxedSegmentation: true,
    clips: [
      { id: 'gy3_2_corrected', src: 'data/raw/validation_mp4/gy3_2_corrected.mp4' },
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
  mediaPipeProcessingMode?: 'server' | 'client'
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
    mediaPipeProcessingMode,
  } = config

  const selectedMediaPipeMode: 'server' | 'client' = (() => {
    const queryMode = new URLSearchParams(window.location.search).get('mp')
    const envMode = import.meta.env.VITE_MEDIAPIPE_AB_MODE
    const storedMode = localStorage.getItem(MEDIAPIPE_MODE_STORAGE_KEY)
    const rawMode = mediaPipeProcessingMode || queryMode || storedMode || envMode || 'server'
    const normalizedMode = rawMode === 'client' ? 'client' : 'server'
    if (queryMode === 'client' || queryMode === 'server') {
      localStorage.setItem(MEDIAPIPE_MODE_STORAGE_KEY, normalizedMode)
    }
    return normalizedMode
  })()

  const selectedScenarioMode = (() => {
    const queryMode = new URLSearchParams(window.location.search).get('scenario')
    const storedMode = localStorage.getItem(SCENARIO_MODE_STORAGE_KEY)
    const rawMode = queryMode || storedMode || 'off'
    const enabled = ['1', 'true', 'yes', 'resident'].includes(rawMode.toLowerCase())
    if (queryMode) {
      localStorage.setItem(SCENARIO_MODE_STORAGE_KEY, enabled ? 'resident' : 'off')
    }
    return enabled
  })()

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
  const demoMotionPrevRef = useRef<Uint8Array | null>(null)
  const demoPauseFramesRef = useRef(0)
  const demoSegmentStartTimeRef = useRef(0)
  const demoHandFramesSinceBoundaryRef = useRef(0)
  const demoNextBoundaryIndexRef = useRef(0)
  const demoSuppressUntilMotionRef = useRef(false)
  const demoSuppressStartedTimeRef = useRef(0)
  const demoResumeMotionFramesRef = useRef(0)
  const demoBoundaryFinalizingRef = useRef(false)
  const demoFinalizeReasonRef = useRef('')
  const demoSingleClipFinalizedRef = useRef(false)
  const demoRunTokenRef = useRef(0)
  const liveGlossBufferRef = useRef<string[]>([])
  const liveGlossTimerRef = useRef<number | null>(null)

  const liveMotionPrevLeftRef = useRef<[number, number] | null>(null)
  const liveMotionPrevRightRef = useRef<[number, number] | null>(null)
  const liveMotionPauseFramesRef = useRef(0)
  const liveMotionResumeFramesRef = useRef(0)
  const liveMotionSuppressedRef = useRef(true)
  const liveSegmentFrameCountRef = useRef(0)
  const liveBoundaryFinalizingRef = useRef(false)
  const liveFinalizeReasonRef = useRef('')
  const liveNoHandFramesRef = useRef(0)
  const liveHandsDownStartRef = useRef<number | null>(null)

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
  const [liveSegmentStatus, setLiveSegmentStatus] = useState<LiveSegmentStatus>(INITIAL_LIVE_SEGMENT_STATUS)

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

  const getManualDemoBoundaries = (scenario: DemoScenario | null) => {
    if (!scenario || !scenario.segmentation || typeof scenario.segmentation === 'string') return []
    return scenario.segmentation.mode === 'manual' ? scenario.segmentation.boundariesSec : []
  }

  const getEffectiveScenarioMode = useCallback(() => (
    selectedScenarioMode || Boolean(demoScenarioRef.current?.forceScenarioMode)
  ), [selectedScenarioMode])

  const convertLiveGlossToText = useCallback(async () => {
    const words = liveGlossBufferRef.current
    liveGlossBufferRef.current = []
    const gloss = words.join(' + ')
    if (!gloss) return
    if (/\b(?:WORD|SEN)\d{3,4}\b/.test(gloss)) {
      onMessageRef.current({
        id: `${Date.now()}-live-id-fallback-${Math.random()}`,
        sender: 'patient',
        text: gloss,
        timestamp: new Date(),
        label: '수어 인식 후보',
      })
      return
    }

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

  const commitScenarioText = useCallback((text: string | null | undefined) => {
    const sentence = String(text || '').trim()
    if (!sentence) return false
    demoGlossBufferRef.current = []
    liveGlossBufferRef.current = []
    if (liveGlossTimerRef.current) {
      window.clearTimeout(liveGlossTimerRef.current)
      liveGlossTimerRef.current = null
    }
    onMessageRef.current({
      id: `${Date.now()}-scenario-text-${Math.random()}`,
      sender: 'patient',
      text: sentence,
      timestamp: new Date(),
      label: '수어 문장 인식',
    })
    return true
  }, [])

  const shouldCommitScenarioText = (prediction: any): boolean => {
    const text = String(prediction?.scenario_text || '').trim()
    if (!text) return false
    if (isDemoModeRef.current && demoScenarioRef.current?.forceScenarioMode) {
      const hasManualBoundaries = getManualDemoBoundaries(demoScenarioRef.current).length > 0
      const source = String(prediction?.scenario?.lookup_source || '')
      if (!hasManualBoundaries && source === 'single_word') return false
    }
    return true
  }

  const getScenarioLookupKey = (prediction: any): string => {
    if (isDemoModeRef.current && demoScenarioRef.current?.forceScenarioMode) {
      const pairCandidate = prediction?.scenario?.fusion_candidates?.find(
        (candidate: any) => {
          if (candidate?.source !== 'word_sentence_pair' || !candidate?.key) return false
          const score = Number(candidate?.score || 0)
          const wordConf = Number(candidate?.word?.confidence || 0)
          const sentenceConf = Number(candidate?.sentence?.confidence || 0)
          const lookupKey = String(prediction?.scenario?.lookup_key || '')
          return lookupKey === candidate.key || (score >= 0.35 && wordConf >= 0.20 && sentenceConf >= 0.12)
        },
      )
      if (pairCandidate?.key) return String(pairCandidate.key).trim()
    }
    return String(prediction?.scenario?.lookup_key || '').trim()
  }

  const shouldBufferDemoScenarioGloss = (prediction: any): boolean => {
    if (!isDemoModeRef.current || !demoScenarioRef.current?.forceScenarioMode) return false
    if (getManualDemoBoundaries(demoScenarioRef.current).length <= 0) return false
    const key = getScenarioLookupKey(prediction)
    return Boolean(key)
  }

  const commitDemoScenarioGloss = useCallback((prediction: any): boolean => {
    if (!shouldBufferDemoScenarioGloss(prediction)) return false
    const key = getScenarioLookupKey(prediction)
    const parts = key.split('+').map((item) => item.trim()).filter(Boolean)
    if (parts.length === 0) return false
    parts.forEach((part) => commitRecognizedWord(part))
    return true
  }, [commitRecognizedWord])

  const convertDemoGlossToText = useCallback(async (scenario: DemoScenario) => {
    const runToken = demoRunTokenRef.current
    const words = demoGlossBufferRef.current
    demoGlossBufferRef.current = []
    const gloss = words.join(' + ')
    if (!gloss) {
      if (scenario.forceScenarioMode) return
      onMessageRef.current({
        id: `${Date.now()}-demo-gloss-empty-${Math.random()}`,
        sender: 'doctor',
        text: '영상에서 예측된 gloss가 없어 자연어 변환을 실행하지 않았습니다.',
        timestamp: new Date(),
        label: '데모 변환 중단',
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
      if (runToken !== demoRunTokenRef.current) return
      onMessageRef.current({
        id: `${Date.now()}-demo-gloss-${Math.random()}`,
        sender: 'patient',
        text: data.text || gloss,
        timestamp: new Date(),
        label: '데모 문장 변환',
      })
    } catch (err) {
      if (runToken !== demoRunTokenRef.current) return
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

  const createBlankFrameBlob = useCallback(async () => {
    const canvas = canvasRef.current || document.createElement('canvas')
    canvas.width = PREDICT_FRAME_WIDTH
    canvas.height = PREDICT_FRAME_HEIGHT
    const ctx = canvas.getContext('2d')
    if (!ctx) return null
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, canvas.width, canvas.height)
    return await new Promise<Blob | null>((resolve) => {
      canvas.toBlob((blob) => resolve(blob), 'image/jpeg', JPEG_QUALITY)
    })
  }, [])

  const buildPredictionPayload = useCallback((frameId: number, extra: Record<string, unknown> = {}) => {
    const useInternalDemoSegmentation = isDemoModeRef.current && demoScenarioRef.current?.clips.length === 1
    return {
      frame_id: frameId.toString(),
      model_type: modelType === 'cnn_gru' ? 'sequence' : 'lstm',
      landmark_layout: 'mediapipe_xyz',
      client_id: clientIdRef.current,
      confidence_threshold: confidenceThreshold,
      window_size: windowSize,
      stable_min_count: stableMinCount,
      max_missing_frames: useInternalDemoSegmentation ? 999 : maxMissingFrames,
      min_segment_frames: 8,
      tta_enabled: false,
      run_model: frameId % MODEL_INFERENCE_EVERY_N_FRAMES === 0,
      scenario_mode: getEffectiveScenarioMode(),
      ...extra,
    }
  }, [confidenceThreshold, getEffectiveScenarioMode, maxMissingFrames, modelType, stableMinCount, windowSize])

  const postClientLandmarks = useCallback(async (
    frameId: number,
    extra: Record<string, unknown> = {},
  ) => {
    const payload = buildPredictionPayload(frameId, extra)
    const firstPassBody = JSON.stringify(payload)
    const payloadWithStats = {
      ...payload,
      client_payload_bytes: new Blob([firstPassBody]).size,
    }
    const body = JSON.stringify(payloadWithStats)
    const res = await fetch('/api/predict_landmarks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      credentials: 'include',
      body,
    })
    return await res.json().catch(() => ({}))
  }, [buildPredictionPayload])

  const finalizeDemoSegment = useCallback(async (forceFinalize = true) => {
    if (!isDemoModeRef.current) return
    const scenario = demoScenarioRef.current
    const isSingleClipWithoutBoundaries = Boolean(
      scenario &&
      scenario.clips.length === 1 &&
      getManualDemoBoundaries(scenario).length === 0,
    )
    if (isSingleClipWithoutBoundaries && demoSingleClipFinalizedRef.current) return

    const startedAt = Date.now()
    while (isPredictingRef.current && Date.now() - startedAt < 700) {
      await new Promise((resolve) => window.setTimeout(resolve, 40))
    }

    const blob = forceFinalize ? null : await createBlankFrameBlob()
    if (!forceFinalize && !blob) return

    isPredictingRef.current = true
    try {
      for (let i = 0; i < (forceFinalize ? 1 : maxMissingFrames + 2); i += 1) {
        const frameId = nextFrameIdRef.current + 1
        nextFrameIdRef.current = frameId
        latestFrameIdRef.current = frameId

        const formData = new FormData()
        if (blob) {
          formData.append('frame', blob)
          formData.append('upload_bytes', blob.size.toString())
        } else {
          formData.append('upload_bytes', '0')
        }
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
        formData.append('scenario_mode', getEffectiveScenarioMode().toString())
        if (forceFinalize) {
          const video = videoRef.current
          formData.append('tta_enabled', 'false')
          formData.append('force_finalize', 'true')
          formData.append('demo_video_time_sec', (video?.currentTime || 0).toFixed(3))
          formData.append('demo_segment_start_sec', demoSegmentStartTimeRef.current.toFixed(3))
          formData.append('demo_finalize_reason', demoFinalizeReasonRef.current || 'force_finalize')
        }

        const data = selectedMediaPipeMode === 'client' && forceFinalize
          ? await postClientLandmarks(frameId, {
              force_finalize: true,
              run_model: true,
              tta_enabled: false,
              max_missing_frames: maxMissingFrames,
              demo_video_time_sec: (videoRef.current?.currentTime || 0).toFixed(3),
              demo_segment_start_sec: demoSegmentStartTimeRef.current.toFixed(3),
              demo_finalize_reason: demoFinalizeReasonRef.current || 'force_finalize',
            })
          : await fetch('/api/predict', { method: 'POST', body: formData }).then((res) => res.json().catch(() => ({})))
        const prediction = data.prediction
        if (!prediction) continue

        const pred: Prediction = {
          label: prediction.label,
          display_label: prediction.display_label,
          confidence: prediction.confidence,
          timestamp: Date.now(),
          has_hand: prediction.has_hand,
          window_filled: prediction.window_filled,
          window_progress: prediction.window_progress,
          window_size: prediction.window_size,
          missing_frames: prediction.missing_frames,
          max_missing_frames: prediction.max_missing_frames,
          top_predictions: prediction.top_predictions,
          processing_mode: prediction.processing_mode,
          process_ms: prediction.process_ms,
          client_mediapipe_ms: prediction.client_mediapipe_ms,
          upload_bytes: prediction.upload_bytes,
          scenario_text: prediction.scenario_text,
          scenario: prediction.scenario,
        }
        setCurrentPrediction(pred)

        if (prediction.segment_finalized) {
          if (commitDemoScenarioGloss(prediction)) {
            if (isSingleClipWithoutBoundaries) demoSingleClipFinalizedRef.current = true
            break
          }
          if (shouldCommitScenarioText(prediction) && commitScenarioText(prediction.scenario_text)) {
            if (isSingleClipWithoutBoundaries) demoSingleClipFinalizedRef.current = true
            break
          }
          if (pred.label && pred.confidence >= confidenceThreshold) {
            commitRecognizedWord(pred.display_label || pred.label)
            if (isSingleClipWithoutBoundaries) demoSingleClipFinalizedRef.current = true
          }
          break
        }
      }
    } catch (err) {
      console.error('Failed to finalize demo segment:', err)
    } finally {
      isPredictingRef.current = false
    }
  }, [
    commitRecognizedWord,
    commitDemoScenarioGloss,
    commitScenarioText,
    confidenceThreshold,
    createBlankFrameBlob,
    getEffectiveScenarioMode,
    maxMissingFrames,
    modelType,
    postClientLandmarks,
    selectedMediaPipeMode,
    stableMinCount,
    windowSize,
  ])

  const resetDemoInternalMotionState = useCallback(() => {
    demoMotionPrevRef.current = null
    demoPauseFramesRef.current = 0
    demoSegmentStartTimeRef.current = videoRef.current?.currentTime || 0
    demoHandFramesSinceBoundaryRef.current = 0
    demoNextBoundaryIndexRef.current = 0
    demoSuppressUntilMotionRef.current = false
    demoSuppressStartedTimeRef.current = 0
    demoResumeMotionFramesRef.current = 0
    demoBoundaryFinalizingRef.current = false
    demoFinalizeReasonRef.current = ''
  }, [])

  const finalizeDemoInternalBoundary = useCallback(async () => {
    if (demoBoundaryFinalizingRef.current) return
    demoBoundaryFinalizingRef.current = true
    try {
      await finalizeDemoSegment()
      peakGestureRef.current = null
      isHandUpRef.current = false
    } finally {
      const hasManualBoundaries = getManualDemoBoundaries(demoScenarioRef.current).length > 0
      demoMotionPrevRef.current = null
      demoPauseFramesRef.current = 0
      demoSegmentStartTimeRef.current = videoRef.current?.currentTime || 0
      demoHandFramesSinceBoundaryRef.current = 0
      demoSuppressUntilMotionRef.current = !hasManualBoundaries
      demoSuppressStartedTimeRef.current = videoRef.current?.currentTime || 0
      demoResumeMotionFramesRef.current = 0
      demoBoundaryFinalizingRef.current = false
    }
  }, [finalizeDemoSegment])

  const detectDemoInternalBoundary = useCallback((
    ctx: CanvasRenderingContext2D,
    width: number,
    height: number,
  ): 'send' | 'suppress' | 'finalize' => {
    const scenario = demoScenarioRef.current
    const video = videoRef.current
    if (!isDemoModeRef.current || !scenario || scenario.clips.length !== 1 || !video) {
      return 'send'
    }

    const manualBoundaries = getManualDemoBoundaries(scenario)
    const nextBoundary = manualBoundaries[demoNextBoundaryIndexRef.current]
    if (nextBoundary !== undefined && !demoBoundaryFinalizingRef.current && video.currentTime >= nextBoundary) {
      demoNextBoundaryIndexRef.current += 1
      demoMotionPrevRef.current = null
      demoPauseFramesRef.current = 0
      demoFinalizeReasonRef.current = `manual_boundary:${nextBoundary}`
      return 'finalize'
    }
    if (manualBoundaries.length > 0) {
      return demoSuppressUntilMotionRef.current ? 'suppress' : 'send'
    }

    if (scenario.forceScenarioMode) {
      return 'send'
    }

    const image = ctx.getImageData(0, 0, width, height).data
    const sampledWidth = Math.ceil(width / DEMO_MOTION_SAMPLE_STEP)
    const sampledHeight = Math.ceil(height / DEMO_MOTION_SAMPLE_STEP)
    const gray = new Uint8Array(sampledWidth * sampledHeight)

    let p = 0
    for (let y = 0; y < height; y += DEMO_MOTION_SAMPLE_STEP) {
      for (let x = 0; x < width; x += DEMO_MOTION_SAMPLE_STEP) {
        const idx = (y * width + x) * 4
        gray[p] = Math.round((image[idx] + image[idx + 1] + image[idx + 2]) / 3)
        p += 1
      }
    }

    const prev = demoMotionPrevRef.current
    if (!prev || prev.length !== gray.length) {
      demoMotionPrevRef.current = gray
      return demoSuppressUntilMotionRef.current ? 'suppress' : 'send'
    }

    let totalDiff = 0
    for (let i = 0; i < gray.length; i += 1) {
      totalDiff += Math.abs(gray[i] - prev[i])
    }
    const diff = totalDiff / gray.length
    demoMotionPrevRef.current = gray

    if (demoSuppressUntilMotionRef.current) {
      const suppressedElapsed = video.currentTime - demoSuppressStartedTimeRef.current
      if (suppressedElapsed >= DEMO_INTERNAL_SUPPRESS_MIN_SECONDS && diff > DEMO_INTERNAL_PAUSE_RESUME_THRESHOLD) {
        demoResumeMotionFramesRef.current += 1
      } else {
        demoResumeMotionFramesRef.current = 0
      }

      if (demoResumeMotionFramesRef.current >= DEMO_INTERNAL_RESUME_MIN_FRAMES) {
        demoSuppressUntilMotionRef.current = false
        demoSuppressStartedTimeRef.current = 0
        demoResumeMotionFramesRef.current = 0
        demoPauseFramesRef.current = 0
        demoSegmentStartTimeRef.current = video.currentTime
        demoHandFramesSinceBoundaryRef.current = 0
        return 'send'
      }
      return 'suppress'
    }

    if (demoBoundaryFinalizingRef.current) return 'suppress'

    const segmentElapsed = video.currentTime - demoSegmentStartTimeRef.current
    const useRelaxedSegmentation = Boolean(scenario.mirrorForPrediction || scenario.relaxedSegmentation)
    const pauseDiffThreshold = useRelaxedSegmentation
      ? DEMO_MIRROR_PAUSE_DIFF_THRESHOLD
      : DEMO_INTERNAL_PAUSE_DIFF_THRESHOLD
    const minHandFrames = useRelaxedSegmentation
      ? DEMO_MIRROR_SEGMENT_MIN_HAND_FRAMES
      : DEMO_INTERNAL_SEGMENT_MIN_HAND_FRAMES
    if (
      segmentElapsed < DEMO_INTERNAL_SEGMENT_MIN_SECONDS ||
      demoHandFramesSinceBoundaryRef.current < minHandFrames
    ) {
      demoPauseFramesRef.current = 0
      return 'send'
    }

    if (diff <= pauseDiffThreshold) {
      demoPauseFramesRef.current += 1
    } else {
      demoPauseFramesRef.current = 0
    }

    if (demoPauseFramesRef.current >= DEMO_INTERNAL_PAUSE_MIN_FRAMES) {
      demoPauseFramesRef.current = 0
      demoSuppressUntilMotionRef.current = true
      demoSuppressStartedTimeRef.current = video.currentTime
      demoResumeMotionFramesRef.current = 0
      demoFinalizeReasonRef.current = `auto_pause:diff=${diff.toFixed(3)}`
      return 'finalize'
    }

    return 'send'
  }, [])

  const resetLiveMotionState = useCallback(() => {
    liveMotionPrevLeftRef.current = null
    liveMotionPrevRightRef.current = null
    liveMotionPauseFramesRef.current = 0
    liveMotionResumeFramesRef.current = 0
    liveMotionSuppressedRef.current = true
    liveSegmentFrameCountRef.current = 0
    liveBoundaryFinalizingRef.current = false
    liveFinalizeReasonRef.current = ''
    liveNoHandFramesRef.current = 0
    liveHandsDownStartRef.current = null
    setLiveSegmentStatus(INITIAL_LIVE_SEGMENT_STATUS)
  }, [])

  const getWristPosition = (
    landmarks: any,
    side: 'left' | 'right',
  ): [number, number] | null => {
    const handKey = side === 'left' ? 'left_hand' : 'right_hand'
    const hand = landmarks?.[handKey]
    if (hand && hand.length > 0 && hand[0]) {
      const p = hand[0]
      if (typeof p[0] === 'number' && typeof p[1] === 'number') {
        return [p[0], p[1]]
      }
    }
    return null
  }

  const computeShoulderWidth = (landmarks: any): number => {
    const pose = landmarks?.pose
    if (!pose) return LIVE_SHOULDER_WIDTH_FALLBACK
    const pairs: [number, number][] = [
      [11, 12],
      [23, 24],
    ]
    for (const [a, b] of pairs) {
      const pa = pose[a]
      const pb = pose[b]
      if (pa && pb) {
        const dx = pa[0] - pb[0]
        const dy = pa[1] - pb[1]
        const w = Math.sqrt(dx * dx + dy * dy)
        if (w > LIVE_SHOULDER_WIDTH_MIN) return w
      }
    }
    return LIVE_SHOULDER_WIDTH_FALLBACK
  }

  const computeHipY = (landmarks: any): number | null => {
    const pose = landmarks?.pose
    if (!pose) return null
    const lh = pose[23]
    const rh = pose[24]
    if (!lh || !rh) return null
    if (typeof lh[1] !== 'number' || typeof rh[1] !== 'number') return null
    return (lh[1] + rh[1]) / 2
  }

  const publishLiveSegmentStatus = useCallback((
    state: LiveSegmentState,
    velocity: number | null,
  ) => {
    const handsDownStart = liveHandsDownStartRef.current
    const handsDownElapsed = handsDownStart === null
      ? 0
      : (performance.now() - handsDownStart) / 1000
    setLiveSegmentStatus((prev) => ({
      ...prev,
      state,
      velocity,
      segmentFrameCount: liveSegmentFrameCountRef.current,
      pauseFrames: liveMotionPauseFramesRef.current,
      resumeFrames: liveMotionResumeFramesRef.current,
      handsDownElapsedSec: handsDownElapsed,
    }))
  }, [])

  const detectLiveVelocityBoundary = useCallback((
    landmarks: any,
  ): 'send' | 'suppress' | 'finalize' => {
    if (isDemoModeRef.current) return 'send'
    if (!landmarks) return 'send'

    const curLeft = getWristPosition(landmarks, 'left')
    const curRight = getWristPosition(landmarks, 'right')
    const prevLeft = liveMotionPrevLeftRef.current
    const prevRight = liveMotionPrevRightRef.current
    if (curLeft) liveMotionPrevLeftRef.current = curLeft
    if (curRight) liveMotionPrevRightRef.current = curRight

    if (!curLeft && !curRight) {
      liveNoHandFramesRef.current += 1
      liveHandsDownStartRef.current = null
      if (
        !liveMotionSuppressedRef.current &&
        liveNoHandFramesRef.current >= LIVE_NO_HAND_GRACE_FRAMES
      ) {
        liveMotionSuppressedRef.current = true
        liveMotionPauseFramesRef.current = 0
        liveMotionResumeFramesRef.current = 0
        liveSegmentFrameCountRef.current = 0
      }
      publishLiveSegmentStatus(
        liveMotionSuppressedRef.current ? 'waiting_for_motion' : 'idle',
        null,
      )
      return 'send'
    }

    liveNoHandFramesRef.current = 0

    const shoulderWidth = computeShoulderWidth(landmarks)
    const norm = (cur: [number, number] | null, prev: [number, number] | null): number | null => {
      if (!cur || !prev) return null
      const dx = cur[0] - prev[0]
      const dy = cur[1] - prev[1]
      return Math.sqrt(dx * dx + dy * dy) / shoulderWidth
    }

    const vLeft = norm(curLeft, prevLeft)
    const vRight = norm(curRight, prevRight)
    const velocities: number[] = []
    if (vLeft !== null) velocities.push(vLeft)
    if (vRight !== null) velocities.push(vRight)
    if (velocities.length === 0) {
      publishLiveSegmentStatus(
        liveMotionSuppressedRef.current ? 'waiting_for_motion' : 'idle',
        null,
      )
      return 'send'
    }

    const velocity = Math.max(...velocities)

    if (liveMotionSuppressedRef.current) {
      if (velocity > LIVE_RESUME_VELOCITY_THRESHOLD) {
        liveMotionResumeFramesRef.current += 1
      } else {
        liveMotionResumeFramesRef.current = 0
      }
      if (liveMotionResumeFramesRef.current >= LIVE_RESUME_MIN_FRAMES) {
        liveMotionSuppressedRef.current = false
        liveMotionResumeFramesRef.current = 0
        liveMotionPauseFramesRef.current = 0
        liveSegmentFrameCountRef.current = 0
        publishLiveSegmentStatus('collecting', velocity)
        return 'send'
      }
      publishLiveSegmentStatus('waiting_for_motion', velocity)
      return 'suppress'
    }

    if (liveBoundaryFinalizingRef.current) {
      publishLiveSegmentStatus('about_to_finalize', velocity)
      return 'suppress'
    }

    liveSegmentFrameCountRef.current += 1

    const hipY = computeHipY(landmarks)
    const bothHandsBelowHip =
      hipY !== null &&
      curLeft !== null &&
      curRight !== null &&
      curLeft[1] > hipY &&
      curRight[1] > hipY
    const isResting = velocity < LIVE_HANDS_DOWN_REST_VELOCITY_THRESHOLD
    if (bothHandsBelowHip && isResting) {
      if (liveHandsDownStartRef.current === null) {
        liveHandsDownStartRef.current = performance.now()
      }
      const handsDownElapsedSec =
        (performance.now() - liveHandsDownStartRef.current) / 1000
      if (handsDownElapsedSec >= LIVE_HANDS_DOWN_REST_SECONDS) {
        liveFinalizeReasonRef.current = `hands_down_rest:${handsDownElapsedSec.toFixed(2)}s`
        liveMotionPauseFramesRef.current = 0
        liveMotionResumeFramesRef.current = 0
        liveMotionSuppressedRef.current = true
        liveHandsDownStartRef.current = null
        publishLiveSegmentStatus('about_to_finalize', velocity)
        return 'finalize'
      }
    } else {
      liveHandsDownStartRef.current = null
    }

    if (liveSegmentFrameCountRef.current >= LIVE_SEGMENT_MAX_FRAMES) {
      liveFinalizeReasonRef.current = `max_segment:v=${velocity.toFixed(4)}`
      liveMotionPauseFramesRef.current = 0
      liveMotionSuppressedRef.current = true
      liveMotionResumeFramesRef.current = 0
      liveHandsDownStartRef.current = null
      publishLiveSegmentStatus('about_to_finalize', velocity)
      return 'finalize'
    }

    if (liveSegmentFrameCountRef.current < LIVE_SEGMENT_MIN_FRAMES) {
      liveMotionPauseFramesRef.current = 0
      publishLiveSegmentStatus('collecting', velocity)
      return 'send'
    }

    liveMotionPauseFramesRef.current = 0
    publishLiveSegmentStatus(
      liveHandsDownStartRef.current !== null ? 'about_to_finalize' : 'collecting',
      velocity,
    )
    return 'send'
  }, [publishLiveSegmentStatus])

  const finalizeLiveSegment = useCallback(async () => {
    if (isDemoModeRef.current) return
    if (liveBoundaryFinalizingRef.current) return
    liveBoundaryFinalizingRef.current = true

    const startedAt = Date.now()
    while (isPredictingRef.current && Date.now() - startedAt < 500) {
      await new Promise((resolve) => window.setTimeout(resolve, 40))
    }

    isPredictingRef.current = true
    try {
      const frameId = nextFrameIdRef.current + 1
      nextFrameIdRef.current = frameId
      latestFrameIdRef.current = frameId

      const formData = new FormData()
      formData.append('frame_id', frameId.toString())
      formData.append('upload_bytes', '0')
      formData.append('model_type', modelType === 'cnn_gru' ? 'sequence' : 'lstm')
      formData.append('landmark_layout', 'mediapipe_xyz')
      formData.append('client_id', clientIdRef.current)
      formData.append('confidence_threshold', confidenceThreshold.toString())
      formData.append('window_size', windowSize.toString())
      formData.append('stable_min_count', stableMinCount.toString())
      formData.append('max_missing_frames', maxMissingFrames.toString())
      formData.append('min_segment_frames', '8')
      formData.append('run_model', 'true')
      formData.append('scenario_mode', getEffectiveScenarioMode().toString())
      formData.append('tta_enabled', 'false')
      formData.append('force_finalize', 'true')
      formData.append('live_finalize_reason', liveFinalizeReasonRef.current || 'auto_pause')

      const data = selectedMediaPipeMode === 'client'
        ? await postClientLandmarks(frameId, {
            force_finalize: true,
            run_model: true,
            tta_enabled: false,
            max_missing_frames: maxMissingFrames,
            live_finalize_reason: liveFinalizeReasonRef.current || 'auto_pause',
          })
        : await fetch('/api/predict', { method: 'POST', body: formData }).then((res) => res.json().catch(() => ({})))
      const prediction = data.prediction
      if (prediction) {
        const pred: Prediction = {
          label: prediction.label,
          display_label: prediction.display_label,
          confidence: prediction.confidence,
          timestamp: Date.now(),
          has_hand: prediction.has_hand,
          window_filled: prediction.window_filled,
          window_progress: prediction.window_progress,
          window_size: prediction.window_size,
          missing_frames: prediction.missing_frames,
          max_missing_frames: prediction.max_missing_frames,
          top_predictions: prediction.top_predictions,
          processing_mode: prediction.processing_mode,
          process_ms: prediction.process_ms,
          client_mediapipe_ms: prediction.client_mediapipe_ms,
          upload_bytes: prediction.upload_bytes,
          scenario_text: prediction.scenario_text,
          scenario: prediction.scenario,
        }
        setCurrentPrediction(pred)
        if (
          prediction.segment_finalized &&
          !commitDemoScenarioGloss(prediction) &&
          !(shouldCommitScenarioText(prediction) && commitScenarioText(prediction.scenario_text)) &&
          pred.label &&
          pred.confidence >= confidenceThreshold
        ) {
          commitRecognizedWord(pred.display_label || pred.label)
        }
        const finalizeReason = liveFinalizeReasonRef.current || 'auto_pause'
        const finalizedSegmentFrames =
          typeof prediction.segment_frames === 'number' ? prediction.segment_frames : null
        setLiveSegmentStatus((prev) => ({
          ...prev,
          state: 'waiting_for_motion',
          segmentFrameCount: 0,
          pauseFrames: 0,
          resumeFrames: 0,
          segmentCount: prev.segmentCount + 1,
          lastFinalize: {
            reason: finalizeReason,
            label: pred.label ?? null,
            confidence: pred.confidence ?? 0,
            segmentFrames: finalizedSegmentFrames,
            at: Date.now(),
          },
        }))
        console.log(
          `[live segmentation] finalize reason=${finalizeReason}, ` +
            `label=${pred.label}, conf=${pred.confidence?.toFixed(3)}, ` +
            `segment_frames=${prediction.segment_frames}`,
        )
      }
    } catch (err) {
      console.error('Failed to finalize live segment:', err)
    } finally {
      isPredictingRef.current = false
      liveBoundaryFinalizingRef.current = false
      peakGestureRef.current = null
      isHandUpRef.current = false
    }
  }, [
    commitRecognizedWord,
    commitDemoScenarioGloss,
    commitScenarioText,
    confidenceThreshold,
    getEffectiveScenarioMode,
    maxMissingFrames,
    modelType,
    postClientLandmarks,
    selectedMediaPipeMode,
    stableMinCount,
    windowSize,
  ])

  const stopCamera = useCallback(() => {
    const wasDemoMode = isDemoModeRef.current
    if (activeStreamRef.current) {
      activeStreamRef.current.getTracks().forEach((t) => t.stop())
      activeStreamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.pause()
      videoRef.current.srcObject = null
    }
    if (isDemoModeRef.current && videoRef.current) {
      videoRef.current.pause()
      videoRef.current.removeAttribute('src')
      videoRef.current.load()
    }
    isPredictingRef.current = false
    isDemoModeRef.current = false
    demoRunTokenRef.current += 1
    demoScenarioRef.current = null
    demoClipIndexRef.current = 0
    demoGlossBufferRef.current = []
    resetDemoInternalMotionState()
    resetLiveMotionState()
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
  }, [clearLandmarkOverlay, convertLiveGlossToText, resetDemoInternalMotionState, resetLiveMotionState])

  const startCamera = useCallback(async () => {
    stopCamera()
    resetLiveMotionState()
    clientIdRef.current = `live-${Date.now()}-${Math.random().toString(36).substring(7)}`
    try {
      const deviceId = selectedDeviceIdRef.current
      const videoConstraints: MediaTrackConstraints = {
        width: { ideal: VIDEO_WIDTH },
        height: { ideal: VIDEO_HEIGHT },
        aspectRatio: { ideal: 16 / 9 },
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
  }, [stopCamera, refreshVideoDevices, resetLiveMotionState])

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
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    resetDemoInternalMotionState()
    demoSingleClipFinalizedRef.current = false
    clientIdRef.current = `demo-${clip.id}-${Date.now()}`
    demoScenarioRef.current = scenario
    demoClipIndexRef.current = clipIndex
    isDemoModeRef.current = true

    video.srcObject = null
    video.pause()
    video.removeAttribute('src')
    video.load()
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
    setActiveDemoClipLabel(scenario.displayText)
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
    demoRunTokenRef.current += 1
    peakGestureRef.current = null
    isHandUpRef.current = false
    demoGlossBufferRef.current = []
    demoSingleClipFinalizedRef.current = false
    resetDemoInternalMotionState()
    await playDemoClip(scenario, 0)
  }, [playDemoClip, resetDemoInternalMotionState, stopCamera])

  const handleDemoVideoEnded = useCallback(async () => {
    if (!isDemoModeRef.current || !demoScenarioRef.current) return
    const scenario = demoScenarioRef.current
    const nextClipIndex = demoClipIndexRef.current + 1

    if (!demoSingleClipFinalizedRef.current) {
      demoFinalizeReasonRef.current = 'video_ended'
      await finalizeDemoSegment(scenario.clips.length === 1)
    }

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
    setIsRunning(false)
    setIsDemoMode(false)
    setActiveDemoLabel('')
    setActiveDemoClipLabel('')
    clearLandmarkOverlay()
  }, [clearLandmarkOverlay, commitRecognizedWord, convertDemoGlossToText, finalizeDemoSegment, playDemoClip])

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
    const mirrorForPrediction = Boolean(demoScenarioRef.current?.mirrorForPrediction)
    const px = (x: number) => offsetX + (mirrorForPrediction ? 1 - x : x) * drawWidth
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

  const handlePredictionData = useCallback((data: any, frameId: number) => {
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
      display_label: data.prediction.display_label,
      confidence: data.prediction.confidence,
      timestamp: Date.now(),
      has_hand: data.prediction.has_hand,
      window_filled: data.prediction.window_filled,
      window_progress: data.prediction.window_progress,
      window_size: data.prediction.window_size,
      missing_frames: data.prediction.missing_frames,
      max_missing_frames: data.prediction.max_missing_frames,
      top_predictions: data.prediction.top_predictions,
      processing_mode: data.prediction.processing_mode,
      process_ms: data.prediction.process_ms,
      client_mediapipe_ms: data.prediction.client_mediapipe_ms,
      upload_bytes: data.prediction.upload_bytes,
      scenario_text: data.prediction.scenario_text,
      scenario: data.prediction.scenario,
    }

    setCurrentPrediction(pred)
    if (isDemoModeRef.current && pred.has_hand) {
      demoHandFramesSinceBoundaryRef.current += 1
    }

    if (!isDemoModeRef.current) {
      const liveAction = detectLiveVelocityBoundary(data.prediction.landmarks)
      if (liveAction === 'suppress') return
      if (liveAction === 'finalize') {
        void finalizeLiveSegment()
        return
      }
    }

    if (data.prediction.segment_finalized) {
      isHandUpRef.current = false
      peakGestureRef.current = null
      liveMotionSuppressedRef.current = true
      liveSegmentFrameCountRef.current = 0
      liveMotionPauseFramesRef.current = 0
      liveMotionResumeFramesRef.current = 0
      liveBoundaryFinalizingRef.current = false
      liveNoHandFramesRef.current = 0
      liveHandsDownStartRef.current = null
      const finalizedSegmentFrames =
        typeof data.prediction.segment_frames === 'number' ? data.prediction.segment_frames : null
      setLiveSegmentStatus((prev) => ({
        ...prev,
        state: 'waiting_for_motion',
        segmentFrameCount: 0,
        pauseFrames: 0,
        resumeFrames: 0,
        velocity: null,
        segmentCount: prev.segmentCount + 1,
        lastFinalize: {
          reason: liveFinalizeReasonRef.current || 'backend_max_missing',
          label: pred.label ?? null,
          confidence: pred.confidence ?? 0,
          segmentFrames: finalizedSegmentFrames,
          at: Date.now(),
        },
      }))
      liveFinalizeReasonRef.current = ''
      if (commitDemoScenarioGloss(data.prediction)) {
        return
      }
      if (shouldCommitScenarioText(data.prediction) && commitScenarioText(data.prediction.scenario_text)) {
        return
      }
      if (pred.label && pred.confidence >= confidenceThreshold) {
        commitRecognizedWord(pred.display_label || pred.label)
      }
      return
    }

    if (pred.has_hand) {
      isHandUpRef.current = true
      if (pred.window_filled && pred.label && pred.confidence >= confidenceThreshold) {
        if (peakGestureRef.current && peakGestureRef.current.label !== pred.label) {
          commitRecognizedWord(peakGestureRef.current.label)
          peakGestureRef.current = { label: pred.display_label || pred.label, confidence: pred.confidence }
        } else if (!peakGestureRef.current || pred.confidence > peakGestureRef.current.confidence) {
          peakGestureRef.current = { label: pred.display_label || pred.label, confidence: pred.confidence }
        }
      } else if (peakGestureRef.current) {
        commitRecognizedWord(peakGestureRef.current.label)
        peakGestureRef.current = null
      }
    } else if (isHandUpRef.current) {
      isHandUpRef.current = false
      if (peakGestureRef.current) {
        commitRecognizedWord(peakGestureRef.current.label)
        peakGestureRef.current = null
      }
    }
  }, [
    commitRecognizedWord,
    commitDemoScenarioGloss,
    commitScenarioText,
    confidenceThreshold,
    detectLiveVelocityBoundary,
    drawLandmarks,
    finalizeLiveSegment,
  ])

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
    const mirrorForPrediction = Boolean(demoScenarioRef.current?.mirrorForPrediction)
    ctx.save()
    if (mirrorForPrediction) {
      ctx.translate(canvasRef.current.width, 0)
      ctx.scale(-1, 1)
    }
    ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height)
    ctx.restore()

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

    const frameId = nextFrameIdRef.current + 1
    nextFrameIdRef.current = frameId
    latestFrameIdRef.current = frameId

    if (selectedMediaPipeMode === 'client') {
      try {
        const runtime = await getClientMediaPipeRuntime()
        const clientResult = runtime.detect(canvasRef.current, performance.now())
        const data = await postClientLandmarks(frameId, {
          landmarks: clientResult.landmarks,
          has_hand: clientResult.hasHand,
          has_pose: clientResult.hasPose,
          client_mediapipe_ms: Number(clientResult.processMs.toFixed(1)),
          run_model: frameId % MODEL_INFERENCE_EVERY_N_FRAMES === 0,
        })
        handlePredictionData(data, frameId)
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          console.error('Failed to run client MediaPipe:', err)
        }
      } finally {
        isPredictingRef.current = false
      }
      return
    }

    canvasRef.current.toBlob(async (blob) => {
      if (!blob) { isPredictingRef.current = false; return }
      const formData = new FormData()
      formData.append('frame', blob)
      formData.append('upload_bytes', blob.size.toString())
      formData.append('frame_id', frameId.toString())
      formData.append('model_type', modelType === 'cnn_gru' ? 'sequence' : 'lstm')
      formData.append('landmark_layout', 'mediapipe_xyz')
      formData.append('client_id', clientIdRef.current)
      formData.append('confidence_threshold', confidenceThreshold.toString())
      formData.append('window_size', windowSize.toString())
      formData.append('stable_min_count', stableMinCount.toString())
      const useInternalDemoSegmentation = isDemoModeRef.current && demoScenarioRef.current?.clips.length === 1
      formData.append('max_missing_frames', useInternalDemoSegmentation ? '999' : maxMissingFrames.toString())
      formData.append('min_segment_frames', '8')
      formData.append('tta_enabled', 'false')
      formData.append('scenario_mode', getEffectiveScenarioMode().toString())
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
          display_label: data.prediction.display_label,
          confidence: data.prediction.confidence,
          timestamp: Date.now(),
          has_hand: data.prediction.has_hand,
          window_filled: data.prediction.window_filled,
          window_progress: data.prediction.window_progress,
          window_size: data.prediction.window_size,
          missing_frames: data.prediction.missing_frames,
          max_missing_frames: data.prediction.max_missing_frames,
          top_predictions: data.prediction.top_predictions,
          processing_mode: data.prediction.processing_mode,
          process_ms: data.prediction.process_ms,
          client_mediapipe_ms: data.prediction.client_mediapipe_ms,
          upload_bytes: data.prediction.upload_bytes,
          scenario_text: data.prediction.scenario_text,
          scenario: data.prediction.scenario,
        }
        
        setCurrentPrediction(pred)
        if (isDemoModeRef.current && pred.has_hand) {
          demoHandFramesSinceBoundaryRef.current += 1
        }

        if (!isDemoModeRef.current) {
          const liveAction = detectLiveVelocityBoundary(data.prediction.landmarks)
          if (liveAction === 'suppress') {
            return
          }
          if (liveAction === 'finalize') {
            void finalizeLiveSegment()
            return
          }
        }

        if (data.prediction.segment_finalized) {
          isHandUpRef.current = false
          peakGestureRef.current = null
          liveMotionSuppressedRef.current = true
          liveSegmentFrameCountRef.current = 0
          liveMotionPauseFramesRef.current = 0
          liveMotionResumeFramesRef.current = 0
          liveBoundaryFinalizingRef.current = false
          liveNoHandFramesRef.current = 0
          liveHandsDownStartRef.current = null
          const finalizedSegmentFrames =
            typeof data.prediction.segment_frames === 'number' ? data.prediction.segment_frames : null
          setLiveSegmentStatus((prev) => ({
            ...prev,
            state: 'waiting_for_motion',
            segmentFrameCount: 0,
            pauseFrames: 0,
            resumeFrames: 0,
            velocity: null,
            segmentCount: prev.segmentCount + 1,
            lastFinalize: {
              reason: liveFinalizeReasonRef.current || 'backend_max_missing',
              label: pred.label ?? null,
              confidence: pred.confidence ?? 0,
              segmentFrames: finalizedSegmentFrames,
              at: Date.now(),
            },
          }))
          liveFinalizeReasonRef.current = ''
          if (commitDemoScenarioGloss(data.prediction)) {
            return
          }
          if (shouldCommitScenarioText(data.prediction) && commitScenarioText(data.prediction.scenario_text)) {
            return
          }
          if (pred.label && pred.confidence >= confidenceThreshold) {
            commitRecognizedWord(pred.display_label || pred.label)
          }
          return
        }

        if (pred.has_hand) {
          isHandUpRef.current = true;
          
          if (pred.window_filled && pred.label && pred.confidence >= confidenceThreshold) {
            if (peakGestureRef.current && peakGestureRef.current.label !== pred.label) {
              commitRecognizedWord(peakGestureRef.current.label)
              peakGestureRef.current = { label: pred.display_label || pred.label, confidence: pred.confidence };
            } else {
              if (!peakGestureRef.current || pred.confidence > peakGestureRef.current.confidence) {
                peakGestureRef.current = { label: pred.display_label || pred.label, confidence: pred.confidence };
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
  }, [
    modelType,
    confidenceThreshold,
    windowSize,
    stableMinCount,
    maxMissingFrames,
    commitRecognizedWord,
    commitDemoScenarioGloss,
    commitScenarioText,
    detectDemoInternalBoundary,
    detectLiveVelocityBoundary,
    drawLandmarks,
    finalizeDemoInternalBoundary,
    finalizeLiveSegment,
    getEffectiveScenarioMode,
    handlePredictionData,
    postClientLandmarks,
    selectedMediaPipeMode,
  ])

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
      const sentence = prediction.scenario?.sentence
      const word = prediction.scenario?.word
      if (
        sentence?.label &&
        (sentence.confidence || 0) >= (word?.confidence || 0)
      ) {
        return `SENTENCE ${sentence.display_label || sentence.label} ${(sentence.confidence * 100).toFixed(0)}%`
      }
      if (!prediction.label) return '인식 불확실'
      return `인식 중... ${prediction.display_label || prediction.label}`
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
    liveSegmentStatus,
  }
}
