import { io } from 'socket.io-client'

const SOCKET_URL = import.meta.env.VITE_SOCKET_URL || undefined
const SIGNALING_TOKEN = import.meta.env.VITE_SIGNALING_TOKEN || undefined

export const socket = io(SOCKET_URL, {
  autoConnect: true,
  reconnection: true,
  auth: SIGNALING_TOKEN ? { token: SIGNALING_TOKEN } : undefined,
})

let currentRole: 'kiosk' | 'agent' | null = null

socket.on('connect', () => {
  if (currentRole) {
    console.log(`[socket] re-register role: ${currentRole}`)
    socket.emit('register', { role: currentRole, token: SIGNALING_TOKEN })
  }
})

export function registerRole(role: 'kiosk' | 'agent') {
  currentRole = role
  if (socket.connected) {
    socket.emit('register', { role, token: SIGNALING_TOKEN })
  }
}
