import { RefObject } from 'react'
import type { Prediction } from '../types'

interface VideoFeedProps {
  videoRef: RefObject<HTMLVideoElement | null>
  canvasRef: RefObject<HTMLCanvasElement | null> // 훅에서 전달받은 캡처용 ref
  landmarkCanvasRef: RefObject<HTMLCanvasElement | null>
  isRunning: boolean
  currentPrediction: Prediction | null
  predictionStatus: string
  onVideoEnded?: () => void
}

export default function VideoFeed({
  videoRef, canvasRef, landmarkCanvasRef,
  isRunning, currentPrediction, predictionStatus, onVideoEnded
}: VideoFeedProps) {
  
  const confidence = currentPrediction?.confidence 
    ? `${Math.round(currentPrediction.confidence * 100)}%` 
    : ''

  return (
    <div className="relative h-full w-full overflow-hidden bg-[#020617]">
      {/* 1. 비디오: 사용자가 편하게 보도록 좌우 반전 적용 */}
      <video
        ref={videoRef as RefObject<HTMLVideoElement>}
        autoPlay
        playsInline
        onEnded={onVideoEnded}
        className="h-full w-full object-contain"
        style={{ transform: 'scaleX(-1)', display: isRunning ? 'block' : 'none' }}
      />

      {/* 💡 2. 데이터 캡처용 캔버스 (추가된 핵심 코드!) 
          - 이 캔버스는 화면에 보일 필요가 없으므로 className="hidden" 처리합니다. 
          - 백엔드로 이미지를 전송하기 위해 내부적으로만 사용됩니다. */}
      <canvas
        ref={canvasRef as RefObject<HTMLCanvasElement>}
        className="hidden"
      />

      {/* 3. 랜드마크 캔버스: 비디오와 똑같이 반전시켜 뼈대를 그립니다. */}
      <canvas
        ref={landmarkCanvasRef as RefObject<HTMLCanvasElement>}
        width={640}
        height={480}
        style={{ transform: 'scaleX(-1)' }}
        className="pointer-events-none absolute inset-0 h-full w-full"
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