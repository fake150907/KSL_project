import { io } from 'socket.io-client'

const SOCKET_URL = import.meta.env.VITE_SOCKET_URL || undefined

export const socket = io(SOCKET_URL, {
  autoConnect: true,
  reconnection: true,
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
    socket.emit('register', { role: currentRole, branchId: currentBranchId })
  }
})

export function registerRole(role: 'kiosk' | 'agent') {
  currentRole = role
  if (socket.connected) {
    socket.emit('register', { role, branchId: currentBranchId })
  }
}
