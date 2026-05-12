import { io } from 'socket.io-client'

const SOCKET_URL = import.meta.env.VITE_SOCKET_URL || undefined

export const socket = io(SOCKET_URL, {
  autoConnect: true,
  reconnection: true,
})

let currentRole: 'kiosk' | 'doctor' | null = null

socket.on('connect', () => {
  if (currentRole) {
    console.log(`[socket] re-register role: ${currentRole}`)
    socket.emit('register', { role: currentRole })
  }
})

export function registerRole(role: 'kiosk' | 'doctor') {
  currentRole = role
  if (socket.connected) {
    socket.emit('register', { role })
  }
}
