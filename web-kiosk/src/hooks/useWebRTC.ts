import { useRef, useState, useCallback, useEffect } from 'react'
import { socket } from '../socket'
import { PEER_CONNECTION_CONFIG } from './webrtcConfig'

type Role = 'kiosk' | 'agent'

interface UseWebRTCOptions {
  role: Role
  // 키오스크에서 이미 열린 카메라 스트림을 주입 (중복 스트림 방지)
  getExternalStream?: () => MediaStream | null
}

export function useWebRTC({ role, getExternalStream }: UseWebRTCOptions) {
  const localRef  = useRef<HTMLVideoElement>(null)
  const remoteRef = useRef<HTMLVideoElement>(null)
  const pcRef     = useRef<RTCPeerConnection | null>(null)
  const localStreamRef = useRef<MediaStream | null>(null)
  const candidateQueue = useRef<RTCIceCandidateInit[]>([])

  const [isConnected,  setIsConnected]  = useState(false)
  const [isConnecting, setIsConnecting] = useState(false)
  const [error,        setError]        = useState<string | null>(null)

  const opponent: Role = role === 'agent' ? 'kiosk' : 'agent'

  const createPeerConnection = useCallback(() => {
    if (pcRef.current) pcRef.current.close()

    const pc = new RTCPeerConnection(PEER_CONNECTION_CONFIG)
    pcRef.current = pc

    pc.ontrack = (event) => {
      if (remoteRef.current && event.streams[0]) {
        remoteRef.current.srcObject = event.streams[0]
      }
    }

    pc.onicecandidate = (event) => {
      if (event.candidate) {
        socket.emit('webrtc_ice_candidate', { target: opponent, candidate: event.candidate })
      }
    }

    pc.onconnectionstatechange = () => {
      if (pc.connectionState === 'connected') {
        setIsConnected(true)
        setIsConnecting(false)
        setError(null)
      } else if (pc.connectionState === 'disconnected' || pc.connectionState === 'failed') {
        setIsConnected(false)
        if (pc.connectionState === 'failed') {
          setError('연결 실패. 네트워크를 확인하거나 다시 시도해주세요.')
        }
      } else if (pc.connectionState === 'closed') {
        setIsConnected(false)
      }
    }

    return pc
  }, [opponent])

  const processCandidateQueue = useCallback(async () => {
    if (!pcRef.current?.remoteDescription) return
    while (candidateQueue.current.length > 0) {
      const c = candidateQueue.current.shift()
      if (c) {
        try { await pcRef.current.addIceCandidate(new RTCIceCandidate(c)) } catch { /* 무시 */ }
      }
    }
  }, [])

  // 스트림 획득 — 외부 주입 우선, 없으면 직접 요청
  const getLocalStream = useCallback(async () => {
    // 키오스크: useKioskSignLanguage가 이미 연 스트림 재사용
    if (getExternalStream) {
      const external = getExternalStream()
      if (external) {
        localStreamRef.current = external
        if (localRef.current) localRef.current.srcObject = external
        return external
      }
    }

    // 상담원 또는 외부 스트림 없을 때: 직접 열기
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: true,
      })
      localStreamRef.current = stream
      if (localRef.current) localRef.current.srcObject = stream
      return stream
    } catch (err) {
      const msg = err instanceof Error ? err.message : '카메라/마이크 접근 실패'
      setError(msg)
      throw err
    }
  }, [getExternalStream])

  const startCall = useCallback(async () => {
    // kiosk가 offer를 먼저 보냄 (AgentDashboard가 offer를 받는 구조)
    setIsConnecting(true)
    setError(null)
    candidateQueue.current = []
    try {
      const stream = await getLocalStream()
      const pc = createPeerConnection()
      stream.getTracks().forEach((track) => pc.addTrack(track, stream))
      const offer = await pc.createOffer({ offerToReceiveAudio: true, offerToReceiveVideo: true })
      await pc.setLocalDescription(offer)
      socket.emit('webrtc_offer', { target: opponent, offer })
    } catch (err) {
      setIsConnecting(false)
      console.error('[WebRTC] Offer 실패:', err)
    }
  }, [role, getLocalStream, createPeerConnection])

  const endCall = useCallback(() => {
    pcRef.current?.close()
    pcRef.current = null
    candidateQueue.current = []
    // 외부 주입 스트림은 멈추지 않음 (useKioskSignLanguage가 관리)
    if (!getExternalStream) {
      localStreamRef.current?.getTracks().forEach((t) => t.stop())
    }
    localStreamRef.current = null
    if (localRef.current)  localRef.current.srcObject  = null
    if (remoteRef.current) remoteRef.current.srcObject = null
    setIsConnected(false)
    setIsConnecting(false)
  }, [getExternalStream])

  useEffect(() => {
    const handleOffer = async ({ offer }: { offer: RTCSessionDescriptionInit }) => {
      if (role !== 'agent') return
      setIsConnecting(true)
      candidateQueue.current = []
      try {
        const stream = await getLocalStream()
        const pc = createPeerConnection()
        stream.getTracks().forEach((track) => pc.addTrack(track, stream))
        await pc.setRemoteDescription(new RTCSessionDescription(offer))
        await processCandidateQueue()
        const answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        socket.emit('webrtc_answer', { target: opponent, answer })
      } catch (err) {
        setIsConnecting(false)
        console.error('[WebRTC] Answer 실패:', err)
      }
    }

    const handleAnswer = async ({ answer }: { answer: RTCSessionDescriptionInit }) => {
      if (role !== 'kiosk') return
      try {
        await pcRef.current?.setRemoteDescription(new RTCSessionDescription(answer))
        await processCandidateQueue()
      } catch (err) {
        console.error('[WebRTC] setRemoteDescription 실패:', err)
      }
    }

    const handleIceCandidate = async ({ candidate }: { candidate: RTCIceCandidateInit }) => {
      try {
        if (pcRef.current?.remoteDescription) {
          await pcRef.current.addIceCandidate(new RTCIceCandidate(candidate))
        } else {
          candidateQueue.current.push(candidate)
        }
      } catch { /* 무시 */ }
    }

    socket.on('webrtc_offer',         handleOffer)
    socket.on('webrtc_answer',        handleAnswer)
    socket.on('webrtc_ice_candidate', handleIceCandidate)

    return () => {
      socket.off('webrtc_offer',         handleOffer)
      socket.off('webrtc_answer',        handleAnswer)
      socket.off('webrtc_ice_candidate', handleIceCandidate)
    }
  }, [role, getLocalStream, createPeerConnection, processCandidateQueue])

  useEffect(() => () => endCall(), [endCall])

  return { localRef, remoteRef, startCall, endCall, isConnected, isConnecting, error }
}
