// 클라이언트(브라우저) MediaPipe 런타임.
//
// ?mp=client 모드에서 사용. 브라우저에서 직접 랜드마크를 뽑아(왕복 지연 제거)
// 서버 MediaPipe 없이 실시간으로 오버레이가 따라가게 한다.
//
// @mediapipe/tasks-vision(PoseLandmarker + HandLandmarker)을 쓴다.
// detectForVideo가 동기라 기존 detect() 인터페이스에 그대로 맞는다.
//
// 출력 형식은 서버(Holistic)와 동일한 {pose:33, left_hand:21, right_hand:21}
// (각 점은 정규화된 [x, y, z]) — 백엔드 landmarks_payload_to_frame 및
// 프론트 drawLandmarks가 그대로 받는다.

import { FilesetResolver, PoseLandmarker, HandLandmarker } from '@mediapipe/tasks-vision'

const WASM_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm'
const POSE_MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
const HAND_MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'

export interface HolisticLandmarks {
  pose: number[][]
  left_hand: number[][]
  right_hand: number[][]
}

export interface ClientMediaPipeResult {
  landmarks: HolisticLandmarks
  hasHand: boolean
  hasPose: boolean
  processMs: number
}

export interface ClientMediaPipeRuntime {
  detect(canvas: HTMLCanvasElement | null, timestampMs: number): ClientMediaPipeResult
}

let runtimePromise: Promise<ClientMediaPipeRuntime> | null = null

const EMPTY: HolisticLandmarks = { pose: [], left_hand: [], right_hand: [] }

const toXYZ = (lms: Array<{ x: number; y: number; z: number }> | undefined): number[][] =>
  lms ? lms.map((p) => [p.x, p.y, p.z]) : []

async function createLandmarkers(delegate: 'GPU' | 'CPU') {
  const vision = await FilesetResolver.forVisionTasks(WASM_BASE)
  const pose = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: POSE_MODEL, delegate },
    runningMode: 'VIDEO',
    numPoses: 1,
  })
  const hand = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: HAND_MODEL, delegate },
    runningMode: 'VIDEO',
    numHands: 2,
    // 기본값 0.5는 손이 안 보일 때 얼굴·팔꿈치를 손으로 오탐함 → 0.6으로 올려 오탐 감소
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.6,
  })
  return { pose, hand }
}

export async function getClientMediaPipeRuntime(): Promise<ClientMediaPipeRuntime> {
  if (runtimePromise) return runtimePromise

  runtimePromise = (async () => {
    // GPU(WebGL) 우선, 실패하면 CPU로 폴백
    let pose: PoseLandmarker
    let hand: HandLandmarker
    try {
      const lm = await createLandmarkers('GPU')
      pose = lm.pose
      hand = lm.hand
    } catch {
      const lm = await createLandmarkers('CPU')
      pose = lm.pose
      hand = lm.hand
    }

    // detectForVideo는 타임스탬프가 단조 증가해야 함. 동일 ts 재사용 방지.
    let lastTs = 0

    return {
      detect(canvas: HTMLCanvasElement | null, timestampMs: number): ClientMediaPipeResult {
        if (!canvas) return { landmarks: EMPTY, hasHand: false, hasPose: false, processMs: 0 }
        const t0 = performance.now()
        let ts = Math.round(timestampMs)
        if (ts <= lastTs) ts = lastTs + 1
        lastTs = ts

        const poseRes = pose.detectForVideo(canvas, ts)
        const handRes = hand.detectForVideo(canvas, ts)

        const poseLm = toXYZ(poseRes.landmarks?.[0])
        let left_hand: number[][] = []
        let right_hand: number[][] = []
        handRes.landmarks?.forEach((lm, i) => {
          const label = handRes.handednesses?.[i]?.[0]?.categoryName
          const pts = toXYZ(lm)
          // MediaPipe handedness는 Holistic과 동일 규약 → left/right 그대로 매핑
          if (label === 'Left') left_hand = pts
          else if (label === 'Right') right_hand = pts
        })

        return {
          landmarks: { pose: poseLm, left_hand, right_hand },
          hasHand: left_hand.length > 0 || right_hand.length > 0,
          hasPose: poseLm.length > 0,
          processMs: performance.now() - t0,
        }
      },
    }
  })()

  return runtimePromise
}
