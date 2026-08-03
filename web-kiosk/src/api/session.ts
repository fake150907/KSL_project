import { apiFetch } from './client'
import type { ChatMessage } from '../types'

// ── citizen-session ──────────────────────────────────────
export interface CitizenData {
  name: string
  phone: string
  dob?: string
  gender?: string
  [key: string]: string | undefined
}

// 백엔드 실제 응답 구조: { waiting, citizenData, updatedAt, _db_session_id }
export interface CitizenSessionResponse {
  waiting: boolean
  citizenData: CitizenData | null
  updatedAt: number | null
  _db_session_id: string | null
}

export async function getCitizenSession(): Promise<CitizenSessionResponse> {
  return apiFetch<CitizenSessionResponse>('/api/citizen-session')
}

export async function setCitizenSession(citizenData: CitizenData): Promise<void> {
  await apiFetch('/api/citizen-session', {
    method: 'POST',
    body: JSON.stringify({ citizenData }),
  })
}

// ── messages ─────────────────────────────────────────────
export interface RawMessage {
  id: string
  sender: 'citizen' | 'agent'
  text: string
  timestamp: string
  label?: string
}

export async function fetchMessages(): Promise<ChatMessage[]> {
  const data = await apiFetch<{ messages: RawMessage[] }>('/api/messages')
  return data.messages.map((m) => ({
    ...m,
    timestamp: new Date(m.timestamp),
  }))
}

export async function postMessage(message: Omit<ChatMessage, 'timestamp'> & { timestamp: Date }): Promise<void> {
  await apiFetch('/api/messages', {
    method: 'POST',
    body: JSON.stringify({
      ...message,
      timestamp: message.timestamp.toISOString(),
    }),
  })
}

// ── session-state ─────────────────────────────────────────
export interface SessionState {
  ended: boolean
}

export async function getSessionState(): Promise<SessionState> {
  return apiFetch<SessionState>('/api/session-state')
}
