/**
 * 🤖 백엔드 AI 모델이 프론트엔드로 돌려주는 '수어 예측 결과' 데이터 규격입니다.
 */
export interface Prediction {
  /** 인식된 수어 단어 (예: '우유', '아프다') */
  label: string | null
  /** 인공지능이 해당 단어라고 확신하는 확률 (0.0 ~ 1.0) */
  confidence: number
  /** 예측이 완료된 시점의 타임스탬프 */
  timestamp: number
  /** 화면 안에 민원인의 손이 감지되었는지 여부 */
  has_hand?: boolean
  /** 뼈대 데이터(프레임)가 모델이 요구하는 만큼 꽉 찼는지 여부 */
  window_filled?: boolean
  /** 현재까지 수집된 프레임 개수 */
  window_progress?: number
  /** 모델이 한 번에 분석하는 프레임의 크기 (기본 16) */
  window_size?: number
  /** 손을 놓친 프레임 횟수 */
  missing_frames?: number
  /** 손을 놓쳤을 때 허용하는 최대 프레임 횟수 (이 값을 넘으면 초기화) */
  max_missing_frames?: number
  /** 1등뿐만 아니라 2등, 3등 예측 결과까지 담고 있는 배열 */
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

/**
 * 💬 민원인과 상담원이 주고받는 채팅 메시지의 데이터 규격입니다.
 */
export interface ChatMessage {
  /** 메시지의 고유 식별자 (보통 타임스탬프와 난수 조합) */
  id: string
  /** 메시지를 보낸 사람 ('citizen' = 민원인, 'agent' = 상담원) */
  sender: 'citizen' | 'agent'
  /** 화면에 표시될 메시지 내용 */
  text: string
  /** 메시지를 주고받은 시간 */
  timestamp: Date
  /** 메시지 말풍선 위에 작게 띄워줄 라벨 (예: '수어 번역') */
  label?: string
}

/**
 * 상담원 대시보드 좌측에 작성하는 상담 메모의 데이터 규격입니다.
 */
export interface AgentNote {
  id: string
  text: string
  /** 메모의 카테고리 (색상 분류용) */
  tag: '문의' | '확인' | '처리'
  timestamp: Date
}
