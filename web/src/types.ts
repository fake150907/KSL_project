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
  top_predictions?: Array<{ label: string; confidence: number }>
}

export interface ChatMessage {
  id: string
  sender: 'patient' | 'doctor'
  text: string
  timestamp: Date
  label?: string
}

export interface DoctorNote {
  id: string
  text: string
  tag: '문의' | '확인' | '처리'
  timestamp: Date
}
