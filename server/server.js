import { Server } from "socket.io";
import http from "http";

const server = http.createServer((req, res) => {
  // Railway/Cloud Run 헬스체크 / 생존 확인용 (socket.io 요청은 아래 io가 가로챔)
  if (req.url === "/health" || req.url === "/") {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ status: "ok", service: "signaling" }));
    return;
  }
  res.writeHead(404);
  res.end();
});

const io = new Server(server, {
  cors: {
    origin: true,
    methods: ["GET", "POST"],
    credentials: true
  }
});

// ─────────────────────────────────────────
// 지점(branch)별 연결 관리
// 한 시그널링 서버가 여러 동사무소를 처리하므로 branchId로 분리한다.
// 같은 지점의 키오스크↔상담원만 서로 연결되고, 다른 지점과 섞이지 않는다.
// ─────────────────────────────────────────
const connectedUsers = {};        // socket.id -> { role, branchId }
const roleToSocketId = {};        // "branchId:role" -> socket.id (해당 지점·역할 최신 1개)
const pendingCitizenByBranch = {};// branchId -> 대기 중 민원인 데이터

const iceCandidateQueue = {};     // targetSocketId -> candidate[]

function rk(branchId, role) {
  return `${branchId || "_default"}:${role}`;
}
function branchOf(socketId) {
  return connectedUsers[socketId]?.branchId || "_default";
}

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

  // 1. 역할 + 지점 등록
  socket.on("register", ({ role, branchId }) => {
    const bid = branchId || "_default";
    connectedUsers[socket.id] = { role, branchId: bid };
    roleToSocketId[rk(bid, role)] = socket.id; // 같은 지점·역할 최신 소켓 갱신
    socket.join(rk(bid, role));
    console.log(`[등록됨] ${role} @ ${bid} (Socket ID: ${socket.id})`);

    if (role === "agent") {
      // 상담원이 (재)등록되면(새로고침 포함) 같은 지점 키오스크에 영상 offer 재요청.
      // register가 먼저 처리돼 roleToSocketId가 갱신된 뒤라 offer 라우팅이 보장됨.
      io.to(rk(bid, "kiosk")).emit("request_offer");
    }

    if (role === "agent" && pendingCitizenByBranch[bid]) {
      socket.emit("citizen_arrived", { citizenData: pendingCitizenByBranch[bid] });
      delete pendingCitizenByBranch[bid];
    }
  });

  // 2. 민원인 접수 알림 (같은 지점 상담원에게만)
  socket.on("citizen_arrived", ({ citizenData }) => {
    const bid = branchOf(socket.id);
    console.log(`[민원인 도착] @ ${bid}`, citizenData);
    const agentRoom = io.sockets.adapter.rooms.get(rk(bid, "agent"));
    if (agentRoom && agentRoom.size > 0) {
      io.to(rk(bid, "agent")).emit("citizen_arrived", { citizenData });
    } else {
      console.log(`[대기] ${bid} 상담원 미접속 → 민원인 정보 보관`);
      pendingCitizenByBranch[bid] = citizenData;
    }
  });

  // 3. 상담원 상담 시작
  socket.on("agent_ready", () => {
    const bid = branchOf(socket.id);
    console.log(`[상담 시작] @ ${bid} → 키오스크 신호 전송`);
    io.to(rk(bid, "kiosk")).emit("agent_ready");
  });

  // 상담원 재접속/새로고침 시 같은 지점 키오스크에 offer 재요청
  socket.on("request_offer", () => {
    const bid = branchOf(socket.id);
    io.to(rk(bid, "kiosk")).emit("request_offer");
  });

  // 4. 세션 초기화/종료 (같은 지점에만)
  socket.on("session_reset", () => {
    const bid = branchOf(socket.id);
    console.log(`[세션 종료] @ ${bid}`);
    io.to(rk(bid, "agent")).emit("session_reset");
    io.to(rk(bid, "kiosk")).emit("session_reset");
    delete pendingCitizenByBranch[bid];
  });

  socket.on("session_end", () => {
    const bid = branchOf(socket.id);
    io.to(rk(bid, "agent")).emit("session_end");
    io.to(rk(bid, "kiosk")).emit("session_end");
    delete pendingCitizenByBranch[bid];
  });

  // 채팅 메시지 중계 (같은 지점 상담원↔민원인 양방향)
  socket.on("chat_message", (msg) => {
    const me = connectedUsers[socket.id];
    if (!me) return;
    const targetRole = me.role === "agent" ? "kiosk" : "agent";
    console.log(`[채팅] @ ${me.branchId} ${me.role} → ${targetRole}: ${msg.text}`);
    io.to(rk(me.branchId, targetRole)).emit("chat_message", msg);
  });

  socket.on("consultation_summary_saved", (payload) => {
    const bid = branchOf(socket.id);
    io.to(rk(bid, "agent")).emit("consultation_summary_saved", payload);
  });

  socket.on("agent_voice_status", (payload) => {
    const bid = branchOf(socket.id);
    io.to(rk(bid, "kiosk")).emit("agent_voice_status", payload);
  });

  // ─────────────────────────────────────────
  // WebRTC 시그널링 — target은 역할('kiosk'|'agent'),
  // 발신자의 지점(branchId) 안에서만 라우팅
  // ─────────────────────────────────────────
  socket.on("webrtc_offer", ({ target, offer }) => {
    const bid = branchOf(socket.id);
    const targetSocketId = roleToSocketId[rk(bid, target)];
    console.log(`[WebRTC] Offer → ${rk(bid, target)} (${targetSocketId})`);
    if (!targetSocketId) {
      console.warn(`[WebRTC] Offer 대상 없음: ${rk(bid, target)}`);
      return;
    }
    const targetSocket = io.sockets.sockets.get(targetSocketId);
    if (targetSocket) {
      targetSocket.emit("webrtc_offer", { offer, fromSocketId: socket.id });
      flushIceCandidates(targetSocketId);
    }
  });

  socket.on("webrtc_answer", ({ target, answer }) => {
    const bid = branchOf(socket.id);
    const targetSocketId = roleToSocketId[rk(bid, target)];
    console.log(`[WebRTC] Answer → ${rk(bid, target)} (${targetSocketId})`);
    if (!targetSocketId) {
      console.warn(`[WebRTC] Answer 대상 없음: ${rk(bid, target)}`);
      return;
    }
    const targetSocket = io.sockets.sockets.get(targetSocketId);
    if (targetSocket) {
      targetSocket.emit("webrtc_answer", { answer, fromSocketId: socket.id });
      flushIceCandidates(targetSocketId);
    }
  });

  socket.on("webrtc_ice_candidate", ({ target, candidate }) => {
    const bid = branchOf(socket.id);
    const targetSocketId = roleToSocketId[rk(bid, target)];
    if (!targetSocketId) return;
    const targetSocket = io.sockets.sockets.get(targetSocketId);
    if (targetSocket) {
      targetSocket.emit("webrtc_ice_candidate", { candidate });
    } else {
      if (!iceCandidateQueue[targetSocketId]) {
        iceCandidateQueue[targetSocketId] = [];
      }
      iceCandidateQueue[targetSocketId].push(candidate);
    }
  });

  // 5. 연결 종료
  socket.on("disconnect", () => {
    const me = connectedUsers[socket.id];
    console.log(`[연결 종료] ${me ? `${me.role}@${me.branchId}` : "unknown"} (${socket.id})`);
    if (me) {
      const key = rk(me.branchId, me.role);
      if (roleToSocketId[key] === socket.id) {
        delete roleToSocketId[key];
      }
    }
    delete connectedUsers[socket.id];
    delete iceCandidateQueue[socket.id];
  });
});

const PORT = process.env.PORT || 5001;
server.listen(PORT, () => {
  console.log(`🚀 시그널링 서버 포트 ${PORT}`);
});
