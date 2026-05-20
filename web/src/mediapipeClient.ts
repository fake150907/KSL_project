type LandmarkPoint = {
  x: number
  y: number
  z: number
  visibility?: number
}

type LandmarkPayload = {
  pose: LandmarkPoint[]
  left_hand: LandmarkPoint[]
  right_hand: LandmarkPoint[]
}

type ClientMediaPipeResult = {
  landmarks: LandmarkPayload
  hasHand: boolean
  hasPose: boolean
  processMs: number
}

type ClientMediaPipeRuntime = {
  detect: (source: HTMLCanvasElement, timestampMs: number) => ClientMediaPipeResult
}

const emptyLandmarks = (): LandmarkPayload => ({
  pose: [],
  left_hand: [],
  right_hand: [],
})

export async function getClientMediaPipeRuntime(): Promise<ClientMediaPipeRuntime> {
  return {
    detect: () => ({
      landmarks: emptyLandmarks(),
      hasHand: false,
      hasPose: false,
      processMs: 0,
    }),
  }
}
