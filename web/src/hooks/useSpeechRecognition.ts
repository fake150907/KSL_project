import { useCallback, useEffect, useRef, useState } from 'react'
import type { ChatMessage } from '../types'

const FFT_SIZE = 256
const SMOOTHING_TIME_CONSTANT = 0.72
const BUCKET_COUNT = 48
const VOICE_THRESHOLD_MIN = 0.12
const MAX_VOLUME = 160

interface SpeechRecognitionEvent extends Event {
  results: { [index: number]: { [index: number]: { transcript: string } }; length: number }
}

interface SpeechRecognitionObj extends EventTarget {
  continuous: boolean
  interimResults: boolean
  lang: string
  start(): void
  stop(): void
  onstart: ((this: SpeechRecognitionObj, ev: Event) => void) | null
  onresult: ((this: SpeechRecognitionObj, ev: SpeechRecognitionEvent) => void) | null
  onerror: ((this: SpeechRecognitionObj, ev: Event) => void) | null
  onend: ((this: SpeechRecognitionObj, ev: Event) => void) | null
}

export function useSpeechRecognition(onMessage: (msg: ChatMessage) => void) {
  const recognitionRef = useRef<SpeechRecognitionObj | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioStreamRef = useRef<MediaStream | null>(null)
  const audioAnimationRef = useRef<number | null>(null)
  const isMounted = useRef(true)
  const isIntentionalStop = useRef(false)

  const [isActive, setIsActive] = useState(false)
  const [voiceLevels, setVoiceLevels] = useState<number[]>(Array(BUCKET_COUNT).fill(VOICE_THRESHOLD_MIN))

  const stopVisualizer = useCallback(() => {
    if (audioAnimationRef.current !== null) {
      cancelAnimationFrame(audioAnimationRef.current)
      audioAnimationRef.current = null
    }
    audioStreamRef.current?.getTracks().forEach((track) => track.stop())
    audioStreamRef.current = null
    audioContextRef.current?.close()
    audioContextRef.current = null
    if (isMounted.current) setVoiceLevels(Array(BUCKET_COUNT).fill(VOICE_THRESHOLD_MIN))
  }, [])

  const stop = useCallback(() => {
    isIntentionalStop.current = true
    recognitionRef.current?.stop()
    recognitionRef.current = null
    stopVisualizer()
    if (isMounted.current) setIsActive(false)
  }, [stopVisualizer])

  useEffect(() => {
    isMounted.current = true
    return () => {
      isMounted.current = false
      stop()
    }
  }, [stop])

  const startVisualizer = useCallback(async () => {
    stopVisualizer()
    if (!navigator.mediaDevices?.getUserMedia) return

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
    const AudioCtx = window.AudioContext || (window as any).webkitAudioContext
    const audioContext = new AudioCtx()
    const analyser = audioContext.createAnalyser()

    analyser.fftSize = FFT_SIZE
    analyser.smoothingTimeConstant = SMOOTHING_TIME_CONSTANT
    audioContext.createMediaStreamSource(stream).connect(analyser)

    const data = new Uint8Array(analyser.frequencyBinCount)
    audioStreamRef.current = stream
    audioContextRef.current = audioContext

    const tick = () => {
      analyser.getByteFrequencyData(data)
      const bucketSize = Math.floor(data.length / BUCKET_COUNT)
      const nextLevels = Array.from({ length: BUCKET_COUNT }, (_, i) => {
        const startIndex = i * bucketSize
        const avg = data.slice(startIndex, startIndex + bucketSize).reduce((sum, value) => sum + value, 0) / Math.max(bucketSize, 1)
        return Math.max(VOICE_THRESHOLD_MIN, Math.min(1, avg / MAX_VOLUME))
      })
      if (isMounted.current) setVoiceLevels(nextLevels)
      audioAnimationRef.current = requestAnimationFrame(tick)
    }
    tick()
  }, [stopVisualizer])

  const start = useCallback(async () => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SR) {
      alert('브라우저가 음성 인식을 지원하지 않습니다.')
      return
    }

    try {
      await startVisualizer()
    } catch {}

    isIntentionalStop.current = false

    const recognition = new SR() as SpeechRecognitionObj
    recognition.lang = 'ko-KR'
    recognition.continuous = true
    recognition.interimResults = false

    recognition.onstart = () => {
      if (isMounted.current) setIsActive(true)
    }

    recognition.onresult = (event) => {
      const transcript = event.results[event.results.length - 1][0].transcript.trim()
      if (!transcript || !isMounted.current) return
      onMessage({
        id: `${Date.now()}-${Math.random()}`,
        sender: 'agent',
        text: transcript,
        timestamp: new Date(),
        label: '상담원',
      })
    }

    recognition.onerror = (event: any) => console.error('Speech error:', event.error)

    recognition.onend = () => {
      if (isMounted.current) setIsActive(false)
      if (!isIntentionalStop.current && recognitionRef.current && isMounted.current) {
        recognition.start()
      }
    }

    recognition.start()
    recognitionRef.current = recognition
  }, [onMessage, startVisualizer])

  return { isActive, voiceLevels, start, stop }
}
