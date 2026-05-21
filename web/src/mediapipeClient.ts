// Stub for client-side MediaPipe runtime.
//
// The full implementation is not bundled in handover_web_realz03_20260520.
// REALZ03 demos use server-side MediaPipe (default, mp=server) so this code
// path is unused unless the URL contains ?mp=client. If someone toggles
// client mode, getClientMediaPipeRuntime() throws and useSignLanguage logs
// the error; predictions then stop for that frame but server mode keeps
// working.

export interface ClientMediaPipeResult {
  landmarks: number[]
  hasHand: boolean
  hasPose: boolean
  processMs: number
}

export interface ClientMediaPipeRuntime {
  detect(canvas: HTMLCanvasElement | null, timestampMs: number): ClientMediaPipeResult
}

export async function getClientMediaPipeRuntime(): Promise<ClientMediaPipeRuntime> {
  throw new Error(
    'Client-side MediaPipe runtime is not bundled in this build. ' +
      'Use the default server MediaPipe mode (remove ?mp=client from the URL).',
  )
}
