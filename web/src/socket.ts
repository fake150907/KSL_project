import { io } from 'socket.io-client'

const SOCKET_URL = import.meta.env.VITE_SOCKET_URL || undefined
const SIGNALING_TOKEN = import.meta.env.VITE_SIGNALING_TOKEN || undefined

export const socket = io(SOCKET_URL, {
  autoConnect: true,
  reconnection: true,
  auth: SIGNALING_TOKEN ? { token: SIGNALING_TOKEN } : undefined,
})

let currentRole: 'kiosk' | 'agent' | null = null
let currentBranchId: string | null = null

const BRANCH_KEY = 'ksl-branch-id'

// 새로고침 후에도 지점 유지 (로그인 시 저장됨)
try {
  currentBranchId = localStorage.getItem(BRANCH_KEY)
} catch {
  currentBranchId = null
}

/** 로그인 시 지점 ID를 설정. 이후 모든 register에 함께 전송된다. */
export function setBranchId(branchId: string | null) {
  currentBranchId = branchId
  try {
    if (branchId) localStorage.setItem(BRANCH_KEY, branchId)
    else localStorage.removeItem(BRANCH_KEY)
  } catch {
    /* ignore */
  }
}

socket.on('connect', () => {
  if (currentRole) {
    console.log(`[socket] re-register role: ${currentRole} @ ${currentBranchId ?? '_default'}`)
    socket.emit('register', { role: currentRole, branchId: currentBranchId, token: SIGNALING_TOKEN })
  }
})

export function registerRole(role: 'kiosk' | 'agent') {
  currentRole = role
  if (socket.connected) {
    socket.emit('register', { role, branchId: currentBranchId, token: SIGNALING_TOKEN })
  }
}
