import { RefObject } from 'react'
import type { Prediction } from '../types'

interface VideoFeedProps {
  videoRef: RefObject<HTMLVideoElement | null>
  canvasRef: RefObject<HTMLCanvasElement | null>
  landmarkCanvasRef: RefObject<HTMLCanvasElement | null>
  isRunning: boolean
  currentPrediction: Prediction | null
  predictionStatus: string
  onVideoEnded?: () => void
  compact?: boolean
}

export default function VideoFeed({
  videoRef,
  canvasRef,
  landmarkCanvasRef,
  isRunning,
  currentPrediction,
  predictionStatus,
  onVideoEnded,
  compact = false,
}: VideoFeedProps) {
  const confidence = currentPrediction?.confidence ? `${Math.round(currentPrediction.confidence * 100)}%` : ''

  return (
    <div className={`relative h-full w-full overflow-hidden bg-[#020617] ${compact ? 'rounded-lg' : ''}`}>
      {!isRunning && (
        <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
          <div className="flex h-20 w-20 items-center justify-center rounded-full bg-[#1f2937]">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="1.5">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
              <circle cx="12" cy="7" r="4" />
            </svg>
          </div>
          <span className="text-xs font-semibold text-[#4b5563]">카메라 연결 대기 중</span>
        </div>
      )}

      <video
        ref={videoRef as RefObject<HTMLVideoElement>}
        autoPlay
        playsInline
        onEnded={onVideoEnded}
        className="h-full w-full object-contain"
        style={{ transform: 'scaleX(-1)', display: isRunning ? 'block' : 'none' }}
      />
      <canvas ref={canvasRef as RefObject<HTMLCanvasElement>} width={640} height={480} className="hidden" />
      <canvas
        ref={landmarkCanvasRef as RefObject<HTMLCanvasElement>}
        width={640}
        height={480}
        className="pointer-events-none absolute inset-0 h-full w-full"
        style={{ transform: 'scaleX(-1)' }}
      />

      {isRunning && (
        <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between rounded-lg border border-white/10 bg-slate-950/80 px-4 py-3 backdrop-blur">
          <span className="text-sm font-bold text-slate-200">{predictionStatus || '인식 준비 중'}</span>
          {confidence && <span className="text-xs font-black text-cyan-200">{confidence}</span>}
        </div>
      )}
    </div>
  )
}
