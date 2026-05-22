import { RefObject } from 'react'
import type { Prediction } from '../types'
import type { LiveSegmentStatus } from '../hooks/useSignLanguage'

interface VideoFeedProps {
  videoRef: RefObject<HTMLVideoElement | null>
  canvasRef: RefObject<HTMLCanvasElement | null> // 훅에서 전달받은 캡처용 ref
  landmarkCanvasRef: RefObject<HTMLCanvasElement | null>
  isRunning: boolean
  isDemoMode?: boolean
  currentPrediction: Prediction | null
  predictionStatus: string
  onVideoEnded?: () => void
  camFps?: number
  sendFps?: number
  liveSegmentStatus?: LiveSegmentStatus
}

const STATE_LABEL: Record<LiveSegmentStatus['state'], string> = {
  idle: 'IDLE',
  waiting_for_motion: 'WAIT',
  collecting: 'COLLECT',
  about_to_finalize: 'FINALIZE',
}

const STATE_COLOR: Record<LiveSegmentStatus['state'], { dot: string; border: string; text: string }> = {
  idle: { dot: '#94a3b8', border: 'rgba(148,163,184,0.4)', text: '#cbd5e1' },
  waiting_for_motion: { dot: '#f87171', border: 'rgba(248,113,113,0.55)', text: '#fecaca' },
  collecting: { dot: '#22c55e', border: 'rgba(34,197,94,0.55)', text: '#bbf7d0' },
  about_to_finalize: { dot: '#fbbf24', border: 'rgba(251,191,36,0.7)', text: '#fde68a' },
}

function formatBytes(bytes?: number): string {
  if (!bytes || bytes <= 0) return '--'
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)}MB`
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)}KB`
  return `${bytes}B`
}

function formatDualCandidate(
  item?: { label: string | null; display_label?: string | null; confidence: number },
): string {
  if (!item?.label) return '--'
  const label = item.display_label || item.label
  return `${label} ${Math.round((item.confidence || 0) * 100)}%`
}

function SegmentationOverlay({ status }: { status: LiveSegmentStatus }) {
  const color = STATE_COLOR[status.state]
  const stateLabel = STATE_LABEL[status.state]
  const v = status.velocity
  const vMax = Math.max(status.resumeThreshold * 2.5, 0.15)
  const vRatio = v === null ? 0 : Math.min(v / vMax, 1)
  const pauseRatio = Math.min(status.pauseFrames / Math.max(status.pauseMinFrames, 1), 1)
  const resumeRatio = Math.min(status.resumeFrames / Math.max(status.resumeMinFrames, 1), 1)
  const segRatio = Math.min(status.segmentFrameCount / Math.max(status.segmentMaxFrames, 1), 1)

  return (
    <div
      style={{
        position: 'absolute',
        top: 12,
        right: 12,
        zIndex: 50,
        minWidth: 220,
        background: 'rgba(2,6,23,0.92)',
        backdropFilter: 'blur(6px)',
        border: `2px solid ${color.border}`,
        borderRadius: 8,
        padding: '8px 10px',
        fontFamily: 'var(--font-mono, monospace)',
        fontSize: 11,
        color: '#e2e8f0',
        pointerEvents: 'none',
        boxShadow: status.state === 'about_to_finalize' ? `0 0 12px ${color.border}` : undefined,
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 6 }}>
        <span
          style={{
            display: 'inline-block',
            width: 8,
            height: 8,
            borderRadius: '50%',
            background: color.dot,
            boxShadow: `0 0 6px ${color.dot}`,
          }}
        />
        <span style={{ color: color.text, fontWeight: 700, letterSpacing: '0.05em' }}>{stateLabel}</span>
        <span style={{ marginLeft: 'auto', color: '#94a3b8', fontSize: 10 }}>
          #{status.segmentCount}
        </span>
      </div>

      <div style={{ marginBottom: 4, color: '#94a3b8', fontSize: 10 }}>
        SEGMENT {status.segmentFrameCount}/{status.segmentMaxFrames}
        <span style={{ marginLeft: 4, color: '#64748b' }}>(min {status.segmentMinFrames})</span>
      </div>
      <div style={{ height: 4, background: 'rgba(148,163,184,0.15)', borderRadius: 2, overflow: 'hidden', marginBottom: 6, position: 'relative' }}>
        <div
          style={{
            height: '100%',
            width: `${segRatio * 100}%`,
            background: color.dot,
            transition: 'width 80ms linear',
          }}
        />
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: `${(status.segmentMinFrames / status.segmentMaxFrames) * 100}%`,
            width: 1,
            height: '100%',
            background: 'rgba(255,255,255,0.4)',
          }}
        />
      </div>

      <div style={{ marginBottom: 4, color: '#94a3b8', fontSize: 10 }}>
        VELOCITY {v === null ? '—' : v.toFixed(4)}
      </div>
      <div style={{ height: 4, background: 'rgba(148,163,184,0.15)', borderRadius: 2, overflow: 'hidden', marginBottom: 6, position: 'relative' }}>
        <div
          style={{
            height: '100%',
            width: `${vRatio * 100}%`,
            background: v !== null && v < status.pauseThreshold ? '#fbbf24' : v !== null && v > status.resumeThreshold ? '#22c55e' : '#64748b',
            transition: 'width 80ms linear',
          }}
        />
        <div
          style={{
            position: 'absolute',
            top: -2,
            left: `${(status.pauseThreshold / vMax) * 100}%`,
            width: 1,
            height: 8,
            background: '#fbbf24',
          }}
          title="pause threshold"
        />
        <div
          style={{
            position: 'absolute',
            top: -2,
            left: `${(status.resumeThreshold / vMax) * 100}%`,
            width: 1,
            height: 8,
            background: '#22c55e',
          }}
          title="resume threshold"
        />
      </div>

      <div style={{ display: 'flex', gap: 8, fontSize: 10, color: '#94a3b8' }}>
        <span>
          PAUSE {status.pauseFrames}/{status.pauseMinFrames}
        </span>
        <span style={{ flex: 1, height: 3, background: 'rgba(148,163,184,0.15)', borderRadius: 2, overflow: 'hidden', alignSelf: 'center' }}>
          <span
            style={{
              display: 'block',
              height: '100%',
              width: `${pauseRatio * 100}%`,
              background: '#fbbf24',
            }}
          />
        </span>
      </div>
      <div style={{ display: 'flex', gap: 8, fontSize: 10, color: '#94a3b8', marginTop: 2 }}>
        <span>
          RESUME {status.resumeFrames}/{status.resumeMinFrames}
        </span>
        <span style={{ flex: 1, height: 3, background: 'rgba(148,163,184,0.15)', borderRadius: 2, overflow: 'hidden', alignSelf: 'center' }}>
          <span
            style={{
              display: 'block',
              height: '100%',
              width: `${resumeRatio * 100}%`,
              background: '#22c55e',
            }}
          />
        </span>
      </div>
      <div style={{ display: 'flex', gap: 8, fontSize: 10, color: '#94a3b8', marginTop: 2 }}>
        <span>
          HANDS↓ {status.handsDownElapsedSec.toFixed(2)}s/{status.handsDownRequiredSec.toFixed(1)}s
        </span>
        <span style={{ flex: 1, height: 3, background: 'rgba(148,163,184,0.15)', borderRadius: 2, overflow: 'hidden', alignSelf: 'center' }}>
          <span
            style={{
              display: 'block',
              height: '100%',
              width: `${Math.min(status.handsDownElapsedSec / Math.max(status.handsDownRequiredSec, 0.01), 1) * 100}%`,
              background: '#a78bfa',
            }}
          />
        </span>
      </div>

      {status.lastFinalize && (
        <div
          style={{
            marginTop: 8,
            paddingTop: 6,
            borderTop: '1px solid rgba(148,163,184,0.2)',
            fontSize: 10,
            color: '#94a3b8',
          }}
        >
          <div>
            LAST →{' '}
            <span style={{ color: status.lastFinalize.label ? '#bef264' : '#f87171', fontWeight: 700 }}>
              {status.lastFinalize.label ?? '(low conf)'}
            </span>
            {status.lastFinalize.label && (
              <span style={{ color: '#cbd5e1' }}> {(status.lastFinalize.confidence * 100).toFixed(0)}%</span>
            )}
          </div>
          <div style={{ color: '#64748b' }}>
            {status.lastFinalize.reason}
            {status.lastFinalize.segmentFrames !== null && ` · ${status.lastFinalize.segmentFrames}f`}
          </div>
        </div>
      )}
    </div>
  )
}

export default function VideoFeed({
  videoRef, canvasRef, landmarkCanvasRef,
  isRunning, isDemoMode, currentPrediction, predictionStatus, onVideoEnded,
  camFps, sendFps, liveSegmentStatus,
}: VideoFeedProps) {

  const confidence = currentPrediction?.confidence
    ? `${Math.round(currentPrediction.confidence * 100)}%`
    : ''

  const showLiveHud = Boolean(isRunning && liveSegmentStatus && !isDemoMode)
  const segBorderColor = showLiveHud && liveSegmentStatus
    ? STATE_COLOR[liveSegmentStatus.state].border
    : 'transparent'
  const processingMode = currentPrediction?.processing_mode
  const modeLabel = processingMode === 'client_mediapipe' ? 'CLIENT MP' : 'SERVER MP'
  const modeColor = processingMode === 'client_mediapipe' ? '#bef264' : '#67e8f9'
  const serverMs = currentPrediction?.process_ms
  const clientMs = currentPrediction?.client_mediapipe_ms
  const uploadBytes = currentPrediction?.upload_bytes
  const scenario = currentPrediction?.scenario
  const wordCandidate = scenario?.word
  const sentenceCandidate = scenario?.sentence
  const scenarioText = currentPrediction?.scenario_text || scenario?.scenario_text
  const fusionCandidate = scenario?.fusion_candidates?.[0]

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

      {showLiveHud && liveSegmentStatus && (
        <div
          aria-hidden
          style={{
            position: 'absolute',
            inset: 0,
            pointerEvents: 'none',
            border: `3px solid ${segBorderColor}`,
            borderRadius: 12,
            transition: 'border-color 120ms ease',
            boxShadow:
              liveSegmentStatus.state === 'about_to_finalize'
                ? `inset 0 0 30px ${segBorderColor}`
                : undefined,
          }}
        />
      )}

      {isRunning && camFps !== undefined && (
        <div style={{
          position: 'absolute', top: 12, left: 12,
          background: 'rgba(0,0,0,0.55)', backdropFilter: 'blur(4px)',
          borderRadius: 6, padding: '4px 10px',
          fontFamily: 'var(--font-mono, monospace)',
          fontSize: 13, fontWeight: 500, letterSpacing: '0.04em',
          border: `0.5px solid ${camFps >= 24 ? 'rgba(0,255,136,0.25)' : camFps >= 15 ? 'rgba(255,204,0,0.25)' : 'rgba(255,80,80,0.25)'}`,
          display: 'flex', alignItems: 'baseline', gap: 6,
          pointerEvents: 'none',
        }}>
          <span style={{ color: camFps >= 24 ? '#00ff88' : camFps >= 15 ? '#ffcc00' : '#ff5050' }}>
            {camFps}
          </span>
          <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.45)' }}>CAM</span>
          {sendFps !== undefined && (
            <>
              <span style={{ color: 'rgba(255,255,255,0.2)', fontSize: 10 }}>|</span>
              <span style={{ color: sendFps >= 12 ? '#00ff88' : sendFps >= 8 ? '#ffcc00' : '#ff5050' }}>
                {sendFps}
              </span>
              <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.45)' }}>SEND</span>
            </>
          )}
          <span style={{ fontSize: 9, color: 'rgba(255,255,255,0.3)', marginLeft: 2 }}>FPS</span>
        </div>
      )}

      {isRunning && currentPrediction && (
        <div
          style={{
            position: 'absolute',
            top: 48,
            left: 12,
            zIndex: 55,
            display: 'grid',
            gridTemplateColumns: 'auto auto',
            gap: '4px 10px',
            minWidth: 180,
            padding: '8px 10px',
            borderRadius: 6,
            border: '1px solid rgba(148,163,184,0.24)',
            background: 'rgba(2,6,23,0.76)',
            backdropFilter: 'blur(4px)',
            color: '#cbd5e1',
            fontFamily: 'var(--font-mono, monospace)',
            fontSize: 10,
            pointerEvents: 'none',
          }}
        >
          <span style={{ color: '#64748b' }}>MODE</span>
          <span style={{ color: modeColor, fontWeight: 800 }}>{modeLabel}</span>
          <span style={{ color: '#64748b' }}>SERVER</span>
          <span>{typeof serverMs === 'number' ? `${serverMs.toFixed(1)}ms` : '--'}</span>
          <span style={{ color: '#64748b' }}>CLIENT</span>
          <span>{typeof clientMs === 'number' ? `${clientMs.toFixed(1)}ms` : '--'}</span>
          <span style={{ color: '#64748b' }}>UPLOAD</span>
          <span>{formatBytes(uploadBytes)}</span>
          {(wordCandidate || sentenceCandidate) && (
            <>
              <span style={{ color: '#64748b' }}>WORD</span>
              <span>{formatDualCandidate(wordCandidate)}</span>
              <span style={{ color: '#64748b' }}>SEN</span>
              <span>{formatDualCandidate(sentenceCandidate)}</span>
            </>
          )}
          {scenarioText && (
            <>
              <span style={{ color: '#64748b' }}>TEXT</span>
              <span style={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {scenarioText}
              </span>
            </>
          )}
          {fusionCandidate && (
            <>
              <span style={{ color: '#64748b' }}>FUSE</span>
              <span>
                {fusionCandidate.key} {Math.round(fusionCandidate.score * 100)}%
              </span>
            </>
          )}
        </div>
      )}

      {showLiveHud && liveSegmentStatus && (
        <SegmentationOverlay status={liveSegmentStatus} />
      )}

      {isRunning && isDemoMode && (
        <div
          style={{
            position: 'absolute',
            top: 12,
            right: 12,
            zIndex: 50,
            padding: '4px 10px',
            background: 'rgba(2,6,23,0.78)',
            border: '1px solid rgba(148,163,184,0.4)',
            borderRadius: 6,
            fontFamily: 'var(--font-mono, monospace)',
            fontSize: 11,
            color: '#cbd5e1',
            pointerEvents: 'none',
            letterSpacing: '0.05em',
          }}
        >
          DEMO MODE · RGB segmentation
        </div>
      )}

      {isRunning && (
        <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between rounded-lg border border-white/10 bg-slate-950/80 px-4 py-3 backdrop-blur">
          <span className="text-sm font-bold text-slate-200">{predictionStatus || '인식 준비 중'}</span>
          {confidence && <span className="text-xs font-black text-cyan-200">{confidence}</span>}
        </div>
      )}
    </div>
  )
}
