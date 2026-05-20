/**
 * useWebRTC.ts
 *
 * 키오스크(민원인) ↔ 상담원 간 WebRTC 영상통화 훅
 *
 * 사용법:
 * const { localRef, remoteRef, startCall, endCall, isConnected } = useWebRTC({ role: 'kiosk' })
 *
 * role: 'kiosk' | 'agent'
 * - 'agent' 쪽이 Offer를 먼저 보냄 (발신자)
 * - 'kiosk'  쪽이 Answer를 보냄    (수신자)
 */

import { useRef, useState, useCallback, useEffect } from "react";
import { socket } from "../socket"; // 기존 socket.io 인스턴스
import { PEER_CONNECTION_CONFIG } from "./webrtcConfig";

type Role = "kiosk" | "agent";

interface UseWebRTCOptions {
  role: Role;
}

export function useWebRTC({ role }: UseWebRTCOptions) {
  const localRef  = useRef<HTMLVideoElement>(null);
  const remoteRef = useRef<HTMLVideoElement>(null);
  const pcRef     = useRef<RTCPeerConnection | null>(null);
  const localStreamRef = useRef<MediaStream | null>(null);
  
  // 💡 ICE Candidate 임시 보관소 (버퍼 큐)
  const candidateQueue = useRef<RTCIceCandidateInit[]>([]);

  const [isConnected, setIsConnected]   = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError]               = useState<string | null>(null);

  const opponent: Role = role === "agent" ? "kiosk" : "agent";

  // ─────────────────────────────────────────
  // PeerConnection 생성 + 공통 이벤트 설정
  // ─────────────────────────────────────────
  const createPeerConnection = useCallback(() => {
    if (pcRef.current) {
      pcRef.current.close();
    }

    const pc = new RTCPeerConnection(PEER_CONNECTION_CONFIG);
    pcRef.current = pc;

    // 원격 스트림 수신
    pc.ontrack = (event) => {
      if (remoteRef.current && event.streams[0]) {
        remoteRef.current.srcObject = event.streams[0];
      }
    };

    // ICE candidate 생성 시 시그널링 서버로 전송
    pc.onicecandidate = (event) => {
      if (event.candidate) {
        socket.emit("webrtc_ice_candidate", {
          target: opponent,
          candidate: event.candidate,
        });
      }
    };

    // 연결 상태 변화 감지
    pc.onconnectionstatechange = () => {
      console.log(`[WebRTC] 연결 상태: ${pc.connectionState}`);
      if (pc.connectionState === "connected") {
        setIsConnected(true);
        setIsConnecting(false);
        setError(null);
      } else if (
        pc.connectionState === "disconnected" ||
        pc.connectionState === "failed"
      ) {
        setIsConnected(false);
        if (pc.connectionState === "failed") {
          setError("연결 실패. 네트워크를 확인하거나 다시 시도해주세요.");
        }
      } else if (pc.connectionState === "closed") {
        setIsConnected(false);
      }
    };

    return pc;
  }, [opponent]);

  // ─────────────────────────────────────────
  // 버퍼에 쌓인 ICE Candidate 일괄 처리기
  // ─────────────────────────────────────────
  const processCandidateQueue = async () => {
    if (!pcRef.current || !pcRef.current.remoteDescription) return;
    
    while (candidateQueue.current.length > 0) {
      const candidateInit = candidateQueue.current.shift();
      if (candidateInit) {
        try {
          await pcRef.current.addIceCandidate(new RTCIceCandidate(candidateInit));
        } catch (err) {
          console.error("[WebRTC] 대기 중인 ICE candidate 추가 실패:", err);
        }
      }
    }
  };

  // ─────────────────────────────────────────
  // 로컬 미디어 가져오기 (카메라 + 마이크)
  // ─────────────────────────────────────────
  const getLocalStream = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: true,
      });
      localStreamRef.current = stream;
      if (localRef.current) {
        localRef.current.srcObject = stream;
      }
      return stream;
    } catch (err) {
      const msg = err instanceof Error ? err.message : "카메라/마이크 접근 실패";
      setError(msg);
      throw err;
    }
  }, []);

  // ─────────────────────────────────────────
  // 통화 시작 (agent 쪽에서 호출)
  // ─────────────────────────────────────────
  const startCall = useCallback(async () => {
    if (role !== "agent") return;
    
    setIsConnecting(true);
    setError(null);
    candidateQueue.current = []; // 큐 초기화

    try {
      const stream = await getLocalStream();
      const pc = createPeerConnection();

      stream.getTracks().forEach((track) => pc.addTrack(track, stream));

      const offer = await pc.createOffer({
        offerToReceiveAudio: true,
        offerToReceiveVideo: true,
      });
      await pc.setLocalDescription(offer);

      socket.emit("webrtc_offer", { target: "kiosk", offer });
    } catch (err) {
      setIsConnecting(false);
      console.error("[WebRTC] Offer 생성 실패:", err);
    }
  }, [role, getLocalStream, createPeerConnection]);

  // ─────────────────────────────────────────
  // 통화 종료
  // ─────────────────────────────────────────
  const endCall = useCallback(() => {
    pcRef.current?.close();
    pcRef.current = null;
    candidateQueue.current = []; // 큐 비우기
    localStreamRef.current?.getTracks().forEach((t) => t.stop());
    localStreamRef.current = null;
    if (localRef.current)  localRef.current.srcObject  = null;
    if (remoteRef.current) remoteRef.current.srcObject = null;
    setIsConnected(false);
    setIsConnecting(false);
  }, []);

  // ─────────────────────────────────────────
  // 시그널링 이벤트 수신
  // ─────────────────────────────────────────
  useEffect(() => {
    // Offer 수신 (kiosk 쪽)
    const handleOffer = async ({ offer }: { offer: RTCSessionDescriptionInit }) => {
      if (role !== "kiosk") return;
      setIsConnecting(true);
      candidateQueue.current = []; // 큐 초기화

      try {
        const stream = await getLocalStream();
        const pc = createPeerConnection();
        stream.getTracks().forEach((track) => pc.addTrack(track, stream));

        // Remote Description 설정
        await pc.setRemoteDescription(new RTCSessionDescription(offer));
        // 💡 설정 완료 후 큐에 밀려있던 자재(Candidate) 일괄 적용
        await processCandidateQueue();

        const answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);

        socket.emit("webrtc_answer", { target: "agent", answer });
      } catch (err) {
        setIsConnecting(false);
        console.error("[WebRTC] Answer 생성 실패:", err);
      }
    };

    // Answer 수신 (agent 쪽)
    const handleAnswer = async ({ answer }: { answer: RTCSessionDescriptionInit }) => {
      if (role !== "agent") return;
      try {
        // Remote Description 설정
        await pcRef.current?.setRemoteDescription(new RTCSessionDescription(answer));
        // 💡 설정 완료 후 큐에 밀려있던 자재(Candidate) 일괄 적용
        await processCandidateQueue();
      } catch (err) {
        console.error("[WebRTC] setRemoteDescription 실패:", err);
      }
    };

    // ICE Candidate 수신
    const handleIceCandidate = async ({ candidate }: { candidate: RTCIceCandidateInit }) => {
      try {
        if (pcRef.current && pcRef.current.remoteDescription) {
          // 도면(RemoteDescription)이 확정된 상태라면 즉시 자재 추가
          await pcRef.current.addIceCandidate(new RTCIceCandidate(candidate));
        } else {
          // 도면이 아직 안 왔다면 임시 창고(큐)에 자재 보관
          candidateQueue.current.push(candidate);
        }
      } catch (err) {
        console.error("[WebRTC] ICE candidate 추가 실패:", err);
      }
    };

    socket.on("webrtc_offer",         handleOffer);
    socket.on("webrtc_answer",        handleAnswer);
    socket.on("webrtc_ice_candidate", handleIceCandidate);

    return () => {
      socket.off("webrtc_offer",         handleOffer);
      socket.off("webrtc_answer",        handleAnswer);
      socket.off("webrtc_ice_candidate", handleIceCandidate);
    };
  }, [role, getLocalStream, createPeerConnection]);

  useEffect(() => {
    return () => endCall();
  }, [endCall]);

  return { localRef, remoteRef, startCall, endCall, isConnected, isConnecting, error };
}