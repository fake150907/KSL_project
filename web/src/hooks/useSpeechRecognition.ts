import { useEffect, useRef, useState, useCallback } from 'react'
import type { ChatMessage } from '../types'

// 💡 매직 넘버 상수화
const FFT_SIZE = 256
const SMOOTHING_TIME_CONSTANT = 0.72
const BUCKET_COUNT = 48
const VOICE_THRESHOLD_MIN = 0.12
const MAX_VOLUME = 160

// 💡 타입스크립트 안전장치 (Interface) 추가: any 타입 철거
interface SpeechRecognitionEvent extends Event {
  results: { [index: number]: { [index: number]: { transcript: string } }, length: number };
}
interface SpeechRecognitionObj extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  start(): void;
  stop(): void;
  onstart: ((this: SpeechRecognitionObj, ev: Event) => void) | null;
  onresult: ((this: SpeechRecognitionObj, ev: SpeechRecognitionEvent) => void) | null;
  onerror: ((this: SpeechRecognitionObj, ev: Event) => void) | null;
  onend: ((this: SpeechRecognitionObj, ev: Event) => void) | null;
}

export function useSpeechRecognition(onMessage: (msg: ChatMessage) => void) {
  const recognitionRef = useRef<SpeechRecognitionObj | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const audioStreamRef = useRef<MediaStream | null>(null)
  const audioAnimationRef = useRef<number | null>(null)
  
  const isMounted = useRef(true) // 메모리 누수 방지

  const [isActive, setIsActive] = useState(false)
  const [voiceLevels, setVoiceLevels] = useState<number[]>(Array(BUCKET_COUNT).fill(VOICE_THRESHOLD_MIN))

  useEffect(() => {
    isMounted.current = true
    return () => {
      isMounted.current = false
      stop()
    }
  }, [])

  const stopVisualizer = useCallback(() => {
    if (audioAnimationRef.current !== null) {
      cancelAnimationFrame(audioAnimationRef.current)
      audioAnimationRef.current = null
    }
    audioStreamRef.current?.getTracks().forEach((t) => t.stop())
    audioStreamRef.current = null
    audioContextRef.current?.close()
    audioContextRef.current = null
    if (isMounted.current) setVoiceLevels(Array(BUCKET_COUNT).fill(VOICE_THRESHOLD_MIN))
  }, [])

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
        const start = i * bucketSize
        const avg = data.slice(start, start + bucketSize).reduce((s, v) => s + v, 0) / Math.max(bucketSize, 1)
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
      alert('브라우저가 음성 인식을 지원하지 않습니다. Chrome 또는 Edge를 사용해 주세요.')
      return
    }
    try { await startVisualizer() } catch {}
    
    const recognition = new SR() as SpeechRecognitionObj
    recognition.lang = 'ko-KR'
    recognition.continuous = true
    recognition.interimResults = false
    recognition.onstart = () => { if(isMounted.current) setIsActive(true) }
    recognition.onresult = (event) => {
      const transcript = event.results[event.results.length - 1][0].transcript.trim()
      if (!transcript) return
      
      if(isMounted.current) {
        onMessage({
          id: `${Date.now()}-${Math.random()}`,
          sender: 'doctor',
          text: transcript,
          timestamp: new Date(),
          label: '의사',
        })
      }
    }
    recognition.onerror = (e: any) => console.error('Speech error:', e.error)
    recognition.onend = () => {
      if(isMounted.current) setIsActive(false)
      if (recognitionRef.current && isMounted.current) recognition.start()
    }
    recognition.start()
    recognitionRef.current = recognition
  }, [startVisualizer, onMessage])

  const stop = useCallback(() => {
    if (recognitionRef.current) {
      recognitionRef.current.onend = null // 무한 재시작 루프 끊기
      recognitionRef.current.stop()
    }
    recognitionRef.current = null
    stopVisualizer()
    if (isMounted.current) setIsActive(false)
  }, [stopVisualizer])

  return { isActive, voiceLevels, start, stop }
}