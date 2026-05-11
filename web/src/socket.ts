import { io } from 'socket.io-client'

// 스마트폰 접속을 위해 동적 IP(window.location.hostname) 적용
const SOCKET_PORT = import.meta.env.VITE_SOCKET_PORT || '3001'
const SOCKET_URL = import.meta.env.VITE_SOCKET_URL || `${window.location.protocol}//${window.location.hostname}:${SOCKET_PORT}`

export const socket = io(SOCKET_URL, {
  autoConnect: true,
  reconnection: true,
})

// 💡 핵심 1: 소켓이 끊어지더라도 내가 무슨 역할이었는지 기억하기 위한 메모장
let currentRole: 'kiosk' | 'doctor' | null = null

// 💡 핵심 2: 스마트폰 화면이 꺼지거나 와이파이가 바뀌어 소켓이 재연결될 때, 자동으로 방에 다시 찾아 들어가기
socket.on('connect', () => {
  if (currentRole) {
    console.log(`[소켓 재연결] ${currentRole} 역할 자동 재등록`);
    socket.emit('register', { role: currentRole })
  }
})

export function registerRole(role: 'kiosk' | 'doctor') {
  currentRole = role // 역할 메모하기
  if (socket.connected) {
    socket.emit('register', { role })
  }
}
