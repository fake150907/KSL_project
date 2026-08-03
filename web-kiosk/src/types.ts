export interface Prediction {
  label: string | null
  confidence: number
  timestamp: number
  has_hand?: boolean
  window_filled?: boolean
  window_progress?: number
  window_size?: number
  missing_frames?: number
  max_missing_frames?: number
  top_predictions?: Array<{ label: string; display_label?: string | null; confidence: number }>
  display_label?: string | null
  scenario_mode?: boolean
  processing_mode?: 'server_mediapipe' | 'client_mediapipe'
  process_ms?: number
  client_mediapipe_ms?: number
  upload_bytes?: number
  scenario_text?: string
  scenario?: {
    word?: {
      label: string | null
      display_label?: string | null
      confidence: number
      top: Array<{ label: string; display_label?: string | null; confidence: number }>
    }
    sentence?: {
      label: string | null
      display_label?: string | null
      confidence: number
      top: Array<{ label: string; display_label?: string | null; confidence: number }>
    }
    scenario_text?: string | null
    lookup_hit?: boolean
    lookup_key?: string | null
    lookup_source?: string | null
    lookup_score?: number | null
    fusion_candidates?: Array<{
      key: string
      text: string
      score: number
      source: string
    }>
  }
}

export interface ChatMessage {
  id: string
  sender: 'citizen' | 'agent'
  text: string
  timestamp: Date
  label?: string
}
