import { Server } from "socket.io";
import http from "http";

const server = http.createServer();

const io = new Server(server, {
  cors: {
    origin: true,
    methods: ["GET", "POST"],
    credentials: true
  }
});

// 연결 상태 관리
const connectedUsers = {}; // socket.id -> role
const roleToSocketId = {}; // role -> socket.id (최신 1개만 유지)
let pendingPatient = null;

// ─────────────────────────────────────────
// ICE candidate 큐: offer/answer 이전에
// candidate가 먼저 도착할 때를 대비해 보관
// ─────────────────────────────────────────
const iceCandidateQueue = {}; // targetSocketId -> candidate[]

function flushIceCandidates(targetSocketId) {
  const queue = iceCandidateQueue[targetSocketId];
  if (!queue || queue.length === 0) return;
  const targetSocket = io.sockets.sockets.get(targetSocketId);
  if (targetSocket) {
    queue.forEach((candidate) => {
      targetSocket.emit("webrtc_ice_candidate", { candidate });
    });
  }
  delete iceCandidateQueue[targetSocketId];
}

io.on("connection", (socket) => {
  console.log(`[연결됨] Socket ID: ${socket.id}`);

  // 1. 역할 등록
  socket.on("register", ({ role }) => {
    connectedUsers[socket.id] = role;
    roleToSocketId[role] = socket.id; // 최신 소켓 ID 갱신
    socket.join(role);
    console.log(`[등록됨] ${role} (Socket ID: ${socket.id})`);

    if (role === "doctor" && pendingPatient) {
      socket.emit("patient_arrived", { patientData: pendingPatient });
      pendingPatient = null;
    }
  });

  // 2. 환자 접수 알림
  socket.on("patient_arrived", ({ patientData }) => {
    console.log(`[환자 도착]`, patientData);
    const doctorRoom = io.sockets.adapter.rooms.get("doctor");
    if (doctorRoom && doctorRoom.size > 0) {
      io.to("doctor").emit("patient_arrived", { patientData });
    } else {
      console.log("[대기] 의사 미접속 → 환자 정보 보관");
      pendingPatient = patientData;
    }
  });

  // 3. 의사 진료 시작
  socket.on("doctor_ready", () => {
    console.log("[진료 시작] → 키오스크 신호 전송");
    io.to("kiosk").emit("doctor_ready");
  });

  // 4. 세션 초기화
  socket.on("session_reset", () => {
    console.log("[세션 종료]");
    io.to("doctor").emit("session_reset");
    io.to("kiosk").emit("session_reset");
    pendingPatient = null;
  });

  socket.on("session_end", () => {
    console.log("[session_end]");
    io.to("doctor").emit("session_end");
    io.to("kiosk").emit("session_end");
    pendingPatient = null;
  });

  // ✅ 수정됨: 채팅 메시지 중계 (의사 ↔ 환자 양방향)
  socket.on("chat_message", (msg) => {
    const senderRole = connectedUsers[socket.id]; // 'doctor' | 'kiosk'
    const targetRole = senderRole === "doctor" ? "kiosk" : "doctor";
    console.log(`[채팅] ${senderRole} → ${targetRole}: ${msg.text}`);
    io.to(targetRole).emit("chat_message", msg);
  });

  socket.on("diagnosis_summary_saved", (payload) => {
    console.log("[diagnosis_summary_saved]");
    io.to("doctor").emit("diagnosis_summary_saved", payload);
  });

  // ─────────────────────────────────────────
  // WebRTC 시그널링
  // target: 'kiosk' | 'doctor' (role 이름)
  // → roleToSocketId로 정확한 소켓에만 전송
  // ─────────────────────────────────────────

  socket.on("webrtc_offer", ({ target, offer }) => {
    const targetSocketId = roleToSocketId[target];
    console.log(`[WebRTC] Offer → ${target} (${targetSocketId})`);
    if (!targetSocketId) {
      console.warn(`[WebRTC] Offer 대상 없음: ${target}`);
      return;
    }
    const targetSocket = io.sockets.sockets.get(targetSocketId);
    if (targetSocket) {
      // offer를 받는 쪽에 발신자 socket.id도 함께 전달
      targetSocket.emit("webrtc_offer", { offer, fromSocketId: socket.id });
      // 큐에 쌓인 candidate 있으면 즉시 전달
      flushIceCandidates(targetSocketId);
    }
  });

  socket.on("webrtc_answer", ({ target, answer }) => {
    const targetSocketId = roleToSocketId[target];
    console.log(`[WebRTC] Answer → ${target} (${targetSocketId})`);
    if (!targetSocketId) {
      console.warn(`[WebRTC] Answer 대상 없음: ${target}`);
      return;
    }
    const targetSocket = io.sockets.sockets.get(targetSocketId);
    if (targetSocket) {
      targetSocket.emit("webrtc_answer", { answer, fromSocketId: socket.id });
      flushIceCandidates(targetSocketId);
    }
  });

  socket.on("webrtc_ice_candidate", ({ target, candidate }) => {
    const targetSocketId = roleToSocketId[target];
    if (!targetSocketId) return;
    const targetSocket = io.sockets.sockets.get(targetSocketId);
    if (targetSocket) {
      targetSocket.emit("webrtc_ice_candidate", { candidate });
    } else {
      // 대상 소켓이 아직 준비 안 됐으면 큐에 보관
      if (!iceCandidateQueue[targetSocketId]) {
        iceCandidateQueue[targetSocketId] = [];
      }
      iceCandidateQueue[targetSocketId].push(candidate);
    }
  });

  // 5. 연결 종료
  socket.on("disconnect", () => {
    const role = connectedUsers[socket.id];
    console.log(`[연결 종료] ${role ?? "unknown"} (${socket.id})`);
    delete connectedUsers[socket.id];
    if (role && roleToSocketId[role] === socket.id) {
      delete roleToSocketId[role];
    }
    delete iceCandidateQueue[socket.id];
  });
});

const PORT = process.env.PORT || 5001;
server.listen(PORT, () => {
  console.log(`🚀 시그널링 서버 포트 ${PORT}`);
});
