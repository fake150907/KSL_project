import { apiFetch } from './client'
import type { Prediction } from '../types'

export async function predictLandmarks(
  landmarksPayload: Record<string, unknown>,
): Promise<Prediction> {
  return apiFetch<Prediction>('/api/predict_landmarks', {
    method: 'POST',
    body: JSON.stringify(landmarksPayload),
  })
}

export async function glossToText(glosses: string[]): Promise<{ text: string }> {
  return apiFetch<{ text: string }>('/api/gloss_to_text', {
    method: 'POST',
    body: JSON.stringify({ glosses }),
  })
}
