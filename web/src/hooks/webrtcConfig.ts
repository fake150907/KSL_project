/**
 * webrtcConfig.ts
 *
 * 모바일(LTE/5G) ↔ PC 간 WebRTC 연결이 안 되는 가장 흔한 원인:
 * TURN 서버 없이 STUN만 쓰면 대칭형 NAT(Symmetric NAT)를 못 뚫음.
 *
 * 아래 설정:
 * - STUN: Google 공개 STUN (무료, 속도 빠름)
 * - TURN: open-relay.metered.ca 무료 TURN
 *   → 실제 서비스라면 metered.ca / Twilio / Xirsys 유료 TURN으로 교체 권장
 */

export const ICE_SERVERS: RTCIceServer[] = [
  // STUN: 같은 네트워크 or 단순 NAT 환경에서 직접 P2P
  { urls: "stun:stun.l.google.com:19302" },
  { urls: "stun:stun1.l.google.com:19302" },

  // TURN: 모바일 데이터 ↔ 공유기 등 NAT 통과 불가 환경용 릴레이
  {
    urls: "turn:openrelay.metered.ca:80",
    username: "openrelayproject",
    credential: "openrelayproject",
  },
  {
    urls: "turn:openrelay.metered.ca:443",
    username: "openrelayproject",
    credential: "openrelayproject",
  },
  {
    urls: "turn:openrelay.metered.ca:443?transport=tcp",
    username: "openrelayproject",
    credential: "openrelayproject",
  },
];

export const PEER_CONNECTION_CONFIG: RTCConfiguration = {
  iceServers: ICE_SERVERS,
  // relay 전용 모드: TURN만 쓰고 싶으면 "relay"로 바꾸면 됨 (디버깅용)
  iceTransportPolicy: "all",
  // 번들 정책: 오디오+비디오를 한 연결로 묶어서 포트 절약
  bundlePolicy: "max-bundle",
  rtcpMuxPolicy: "require",
};

/**
 * 사용 예시 (프론트엔드):
 *
 * import { PEER_CONNECTION_CONFIG } from './webrtcConfig'
 *
 * const pc = new RTCPeerConnection(PEER_CONNECTION_CONFIG)
 */