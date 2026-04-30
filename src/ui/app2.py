"""Role: Streamlit MVP UI for realtime sign-text and voice-text conversation.

Simplified version: Queue-based auto-sync, no Fragment complexity.

Run:
  streamlit run src/ui/app2.py
"""

from __future__ import annotations

import queue
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from src.data.dataset import load_npz
from src.data.keypoint_utils import mediapipe_landmarks_to_frame, sequence_to_tensor
from src.models.model_baseline import flatten_sequences
from src.services.stt_service import listen_once
from src.services.tts_service import speak_text
from src.utils.config import load_config


st.set_page_config(page_title="Sign MVP", layout="wide")
config = load_config("config/default.yaml")

# Session-state에 queue 저장 (Streamlit rerun 시에도 유지)
if "prediction_queue" not in st.session_state:
    st.session_state["prediction_queue"] = queue.Queue()

prediction_queue = st.session_state["prediction_queue"]


def setup_auto_sync() -> None:
    """최상단에서 매번 queue 확인"""
    process_prediction_queue()


@dataclass
class SharedRealtimeState:
    label: str = "대기 중"
    confidence: float = 0.0
    status: str = "카메라를 시작하세요."
    frames_ready: int = 0
    frames_target: int = 0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def update(self, label: str, confidence: float, status: str, frames_ready: int = 0, frames_target: int = 0) -> None:
        with self._lock:
            self.label = label
            self.confidence = confidence
            self.status = status
            self.frames_ready = frames_ready
            self.frames_target = frames_target

    def snapshot(self) -> tuple[str, float, str, int, int]:
        with self._lock:
            return self.label, self.confidence, self.status, self.frames_ready, self.frames_target


REALTIME_STATE = SharedRealtimeState()


def init_session_state() -> None:
    st.session_state.setdefault("sign_text", "")
    st.session_state.setdefault("sign_text_input", "")
    st.session_state.setdefault("sign_text_version", 0)
    st.session_state.setdefault("speech_text", "")
    st.session_state.setdefault("speech_text_input", "")
    st.session_state.setdefault("speech_text_version", 0)
    st.session_state.setdefault("last_prediction", "")
    st.session_state.setdefault("auto_commit_sign", True)
    st.session_state.setdefault("frame_stride", 2)
    st.session_state.setdefault("window_size", 16)
    st.session_state.setdefault("stable_min_count", 2)
    st.session_state.setdefault("draw_pose", False)
    st.session_state.setdefault("draw_hands", True)


@st.cache_resource
def get_font(size: int = 34) -> ImageFont.FreeTypeFont | None:
    for candidate in (Path("C:/Windows/Fonts/malgunbd.ttf"), Path("C:/Windows/Fonts/malgun.ttf")):
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return None


@st.cache_resource
def load_baseline_bundle(checkpoint: str) -> tuple[Any, list[str], int]:
    bundle = joblib.load(checkpoint)
    model = bundle["model"]
    labels = [str(x) for x in bundle["labels"]]
    features = int(getattr(model, "n_features_in_", 0))
    return model, labels, features


@st.cache_resource
def load_sequence_bundle(checkpoint: str) -> tuple[Any, list[str], int]:
    import torch
    from src.models.model_sequence import build_sequence_model

    bundle = torch.load(checkpoint, map_location="cpu")
    train_config = bundle.get("config", {}).get("train", {})
    model = build_sequence_model(
        input_size=bundle["input_size"],
        num_classes=len(bundle["labels"]),
        hidden_size=int(train_config.get("hidden_size", 64)),
        num_layers=int(train_config.get("num_layers", 1)),
        dropout=float(train_config.get("dropout", 0.1)),
        rnn_type=str(train_config.get("rnn_type", "gru")),
        model_type=str(bundle.get("model_type", train_config.get("model_type", "rnn"))),
        conv_channels=int(train_config.get("conv_channels", 128)),
        num_heads=int(train_config.get("num_heads", 4)),
        sequence_length=int(bundle.get("config", {}).get("data", {}).get("sequence_length", 32)),
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    labels = [str(x) for x in bundle["labels"]]
    return model, labels, int(bundle["input_size"])


@st.cache_resource
def get_mediapipe_runtime() -> tuple[Any, Any]:
    import mediapipe as mp
    return mp.solutions.holistic, mp.solutions.drawing_utils


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


def expected_feature_count(model_type: str) -> int:
    ckpt_dir = Path(config["paths"]["checkpoints_dir"])
    seq_len = int(config["data"]["sequence_length"])
    if model_type == "baseline":
        _, _, total_features = load_baseline_bundle(str(ckpt_dir / "baseline.joblib"))
        return total_features // seq_len if total_features else 0
    _, _, input_size = load_sequence_bundle(str(ckpt_dir / "sequence_model.pt"))
    return input_size


def predict_live(model_type: str, tensor: np.ndarray) -> tuple[str, float]:
    ckpt_dir = Path(config["paths"]["checkpoints_dir"])
    if model_type == "baseline":
        model, labels, _ = load_baseline_bundle(str(ckpt_dir / "baseline.joblib"))
        flat = flatten_sequences(tensor[None, ...])
        pred = int(model.predict(flat)[0])
        conf = float(np.max(model.predict_proba(flat)[0])) if hasattr(model, "predict_proba") else 1.0
        return labels[pred], conf

    model, labels, _ = load_sequence_bundle(str(ckpt_dir / "sequence_model.pt"))
    import torch
    with torch.no_grad():
        logits = model(torch.tensor(tensor[None, ...], dtype=torch.float32))[0].numpy()
    probs = softmax(logits)
    pred = int(np.argmax(probs))
    return labels[pred], float(probs[pred])


def stable_label(history: deque[str], min_count: int = 3) -> str:
    if not history:
        return ""
    label, count = Counter(history).most_common(1)[0]
    return label if count >= min_count else history[-1]


def align_tensor_features(tensor: np.ndarray, expected: int) -> np.ndarray:
    if expected <= 0 or tensor.shape[1] == expected:
        return tensor
    if tensor.shape[1] > expected:
        return tensor[:, :expected]
    aligned = np.zeros((tensor.shape[0], expected), dtype=tensor.dtype)
    aligned[:, : tensor.shape[1]] = tensor
    return aligned


def draw_text(frame: np.ndarray, text: str, position: tuple[int, int], color: tuple[int, int, int]) -> np.ndarray:
    if not text:
        return frame
    font = get_font()
    if font is None:
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        return frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    x, y = position
    draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def append_sign_text(label: str) -> None:
    current = st.session_state.get("sign_text", "").strip()
    updated = f"{current} {label}".strip() if current else label
    st.session_state["sign_text"] = updated
    st.session_state["sign_text_input"] = updated
    st.session_state["sign_text_version"] = int(st.session_state.get("sign_text_version", 0)) + 1


def sync_sign_text_from_input() -> None:
    key = f"sign_text_input_{int(st.session_state.get('sign_text_version', 0))}"
    st.session_state["sign_text"] = st.session_state.get(key, "")


def sync_speech_text_from_input() -> None:
    key = f"speech_text_input_{int(st.session_state.get('speech_text_version', 0))}"
    st.session_state["speech_text"] = st.session_state.get(key, "")


def process_prediction_queue() -> None:
    """Main thread에서 queue를 주기적으로 확인하고 session_state에 반영"""
    auto_commit = st.session_state.get("auto_commit_sign", True)
    print(f"[PROCESS] auto_commit={auto_commit}, queue_size={prediction_queue.qsize()}")

    if not auto_commit:
        print("[PROCESS] auto_commit disabled, returning")
        return

    try:
        # Non-blocking: queue에 있는 모든 예측을 처리
        while True:
            label, conf = prediction_queue.get_nowait()
            print(f"[PROCESS] Got from queue: '{label}' ({conf:.2f})")
            if label and conf > 0:
                print(f"[PROCESS] Appending to text...")
                append_sign_text(label)
                st.session_state["last_prediction"] = label
                print(f"[AUTO] Applied: {label} ({conf:.2f})")
            else:
                print(f"[PROCESS] Invalid: label='{label}', conf={conf}")
    except queue.Empty:
        print(f"[PROCESS] Queue empty")
        pass


def make_video_processor_factory(
    model_type: str,
    confidence_threshold: float,
    frame_stride: int,
    window_size: int,
    stable_min_count: int,
    draw_pose: bool,
    draw_hands: bool,
):
    sequence_length = int(config["data"]["sequence_length"])
    expected_features = expected_feature_count(model_type)
    preprocess_config = config.get("preprocess", {})
    landmark_layout = str(preprocess_config.get("landmark_layout", "mediapipe_xyz"))
    normalize_mode = str(preprocess_config.get("normalize_mode", "mediapipe_xyz_body"))

    class SignVideoProcessor:
        def __init__(self) -> None:
            self.mp_holistic, self.drawer = get_mediapipe_runtime()
            self.window: deque[np.ndarray] = deque(maxlen=window_size)
            self.predictions: deque[str] = deque(maxlen=5)
            self.holistic = None
            self.frame_index = 0
            self.last_result_frame: np.ndarray | None = None
            self.last_queued_label = ""
            self.last_queue_at = 0.0

        def _ensure_holistic(self) -> None:
            if self.holistic is None:
                self.holistic = self.mp_holistic.Holistic(
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            self._ensure_holistic()
            image = frame.to_ndarray(format="bgr24")
            self.frame_index += 1
            if frame_stride > 1 and self.frame_index % frame_stride != 0:
                if self.last_result_frame is not None:
                    return av.VideoFrame.from_ndarray(self.last_result_frame.copy(), format="bgr24")
                return av.VideoFrame.from_ndarray(image, format="bgr24")

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(rgb)

            has_hand = bool(results.left_hand_landmarks or results.right_hand_landmarks)
            if has_hand:
                frame_points = mediapipe_landmarks_to_frame(results, layout=landmark_layout)
                if frame_points.size:
                    self.window.append(frame_points)
                    REALTIME_STATE.update(
                        "대기 중",
                        0.0,
                        f"프레임 수집 중 {len(self.window)}/{self.window.maxlen}",
                        len(self.window),
                        self.window.maxlen,
                    )
            else:
                self.window.clear()
                self.predictions.clear()
                REALTIME_STATE.update("대기 중", 0.0, "손을 카메라에 보여주세요.", 0, self.window.maxlen)

            if len(self.window) == self.window.maxlen and has_hand:
                tensor = sequence_to_tensor(list(self.window), sequence_length, normalize_mode=normalize_mode)
                tensor = align_tensor_features(tensor, expected_features)
                label, conf = predict_live(model_type, tensor)
                if conf >= confidence_threshold:
                    self.predictions.append(label)
                    shown = stable_label(self.predictions, min_count=stable_min_count)
                    REALTIME_STATE.update(shown, conf, "실시간 수어 인식 중", len(self.window), self.window.maxlen)

                    # Queue에 추가 (Thread-safe)
                    if shown and shown not in ("대기 중", "알 수 없음"):
                        now = time.time()
                        # 같은 라벨이면 0.5초 쿨다운, 다른 라벨이면 항상 queue에 추가
                        if shown != self.last_queued_label or now - self.last_queue_at >= 0.5:
                            prediction_queue.put((shown, conf))
                            self.last_queued_label = shown
                            self.last_queue_at = now
                            print(f"[QUEUE] Enqueued: {shown} ({conf:.2f})")
                else:
                    shown = label
                    REALTIME_STATE.update(shown, conf, "후보는 잡혔지만 신뢰도가 낮습니다.", len(self.window), self.window.maxlen)
                image = draw_text(image, f"{shown} ({conf:.2f})", (20, 20), (30, 220, 30))
            else:
                label, conf, status, _, _ = REALTIME_STATE.snapshot()
                image = draw_text(image, status if label == "대기 중" else f"{label} ({conf:.2f})", (20, 20), (30, 220, 30))

            if draw_pose and results.pose_landmarks:
                self.drawer.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
            if draw_hands and results.left_hand_landmarks:
                self.drawer.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
            if draw_hands and results.right_hand_landmarks:
                self.drawer.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

            self.last_result_frame = image.copy()
            return av.VideoFrame.from_ndarray(image, format="bgr24")

    return SignVideoProcessor


def render_conversation_board() -> None:
    board_left, board_right = st.columns(2)
    with board_left:
        st.subheader("수어하는 사람")
        sign_key = f"sign_text_input_{int(st.session_state.get('sign_text_version', 0))}"
        st.session_state[sign_key] = st.session_state.get("sign_text", "")
        st.text_area("수어 -> 텍스트", height=160, key=sign_key, on_change=sync_sign_text_from_input)
        cols = st.columns(2)
        if cols[0].button("수어 텍스트 지우기", use_container_width=True):
            st.session_state["sign_text"] = ""
            st.session_state["sign_text_version"] = int(st.session_state.get("sign_text_version", 0)) + 1
            st.rerun()
        if cols[1].button("수어 문장 읽기", use_container_width=True):
            speak_text(st.session_state["sign_text"], backend=config["services"]["tts_backend"])

    with board_right:
        st.subheader("말하는 사람")
        speech_key = f"speech_text_input_{int(st.session_state.get('speech_text_version', 0))}"
        st.session_state[speech_key] = st.session_state.get("speech_text", "")
        st.text_area("음성 -> 텍스트", height=160, key=speech_key, on_change=sync_speech_text_from_input)
        cols = st.columns(2)
        if cols[0].button("한 번 듣기", use_container_width=True):
            st.session_state["speech_text"] = listen_once(backend=config["services"]["stt_backend"])
            st.session_state["speech_text_version"] = int(st.session_state.get("speech_text_version", 0)) + 1
            st.rerun()
        if cols[1].button("음성 문장 읽기", use_container_width=True):
            speak_text(st.session_state["speech_text"], backend=config["services"]["tts_backend"])


def render_realtime_status() -> None:
    label, conf, status, frames_ready, frames_target = REALTIME_STATE.snapshot()
    status_cols = st.columns([1.3, 0.9, 1.1, 1.2])
    status_cols[0].metric("실시간 수어 후보", label)
    status_cols[1].metric("신뢰도", f"{conf:.2f}")
    status_cols[2].metric("수집 프레임", f"{frames_ready}/{frames_target}" if frames_target else "0/0")
    status_cols[3].write(status)


def render_offline_panel() -> None:
    with st.expander("오프라인 샘플 점검", expanded=False):
        model_type = st.radio("오프라인 모델", ["baseline", "sequence"], horizontal=True, key="offline_model")
        processed_path = Path(config["paths"]["processed_npz"])
        if not processed_path.exists():
            st.info("처리된 데이터셋이 없습니다.")
            return
        data = load_npz(str(processed_path))
        idx = st.slider("Sample index", 0, max(0, len(data.X) - 1), 0, key="offline_idx")
        ckpt_name = "baseline.joblib" if model_type == "baseline" else "sequence_model.pt"
        ckpt = Path(config["paths"]["checkpoints_dir"]) / ckpt_name
        if not ckpt.exists():
            st.warning(f"Checkpoint not found: {ckpt}")
            return
        st.success(f"Checkpoint ready: {ckpt.name}")
        if st.button("선택 샘플 예측", key="offline_predict"):
            label, conf = predict_live(model_type, data.X[idx])
            st.session_state["last_prediction"] = label
            st.metric("Predicted expression", label, f"{conf:.2f}")
            st.write("True label:", data.labels[int(data.y[idx])])


init_session_state()

camera_width_percent = st.session_state.get("camera_width_percent", 60)
st.markdown(
    f"""
    <style>
    .stApp .main .block-container {{
        max-width: 1700px;
        padding-top: 1rem;
    }}
    .stApp video {{
        width: 100% !important;
        max-width: 100% !important;
        height: auto !important;
        border-radius: 12px;
        background: #111;
    }}
    .stApp [data-testid="stSelectbox"] {{
        max-width: 100%;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("MediaPipe Sign Interpreter MVP")
st.caption("수어하는 사람과 말하는 사람이 같은 화면의 텍스트를 보며 대화하는 MVP")

with st.sidebar:
    st.subheader("실시간 설정")
    model_type = st.radio("실시간 모델", ["baseline", "sequence"], horizontal=True, key="realtime_model")
    confidence_threshold = st.slider("실시간 신뢰도 기준", 0.25, 0.90, 0.45, 0.05)
    st.session_state["frame_stride"] = st.slider("프레임 간격", 1, 4, int(st.session_state["frame_stride"]), 1)
    st.session_state["window_size"] = st.slider("인식 시작 프레임 수", 8, 32, int(st.session_state["window_size"]), 4)
    st.session_state["stable_min_count"] = st.slider("같은 예측 반복 최소 횟수", 1, 4, int(st.session_state["stable_min_count"]), 1)
    st.session_state["auto_commit_sign"] = st.checkbox("수어 텍스트 자동 반영", value=bool(st.session_state["auto_commit_sign"]))
    st.session_state["draw_pose"] = st.checkbox("포즈 선 그리기", value=bool(st.session_state["draw_pose"]))
    st.session_state["draw_hands"] = st.checkbox("손 선 그리기", value=bool(st.session_state["draw_hands"]))
    st.session_state["camera_width_percent"] = st.slider("카메라 폭", 40, 100, max(70, camera_width_percent), 5)
    st.info("현재 인식 가능한 수어는 `가다, 감사, 괜찮다, 배고프다, 병원, 아프다, 우유, 자다` 입니다.")

st.subheader("실시간 수어 카메라")
st.caption("가장 먼저 카메라를 시작하고, 손을 가슴~얼굴 높이에서 화면 중앙에 보여주세요.")
with st.spinner("실시간 수어 엔진을 준비하는 중입니다..."):
    get_mediapipe_runtime()

camera_ratio = max(4, int(st.session_state["camera_width_percent"]))
status_ratio = max(12, 100 - camera_ratio)
camera_area, status_area = st.columns([camera_ratio, status_ratio], vertical_alignment="top")

with camera_area:
    webrtc_streamer(
        key=f"sign-webrtc-{model_type}-{confidence_threshold:.2f}",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={
            "video": {"width": {"ideal": 640}, "height": {"ideal": 360}, "frameRate": {"ideal": 15}, "facingMode": "user"},
            "audio": False,
        },
        async_processing=True,
        video_processor_factory=make_video_processor_factory(
            model_type=model_type,
            confidence_threshold=confidence_threshold,
            frame_stride=int(st.session_state["frame_stride"]),
            window_size=int(st.session_state["window_size"]),
            stable_min_count=int(st.session_state["stable_min_count"]),
            draw_pose=bool(st.session_state["draw_pose"]),
            draw_hands=bool(st.session_state["draw_hands"]),
        ),
    )

st.divider()
with status_area:
    st.subheader("실시간 상태")
    render_realtime_status()
    st.caption("카메라가 시작되면 이 값들이 계속 바뀝니다.")
    if st.session_state.get("last_prediction"):
        st.caption(f"마지막 자동 반영: {st.session_state['last_prediction']}")

st.divider()

# 최상단에서 queue 확인 (매 페이지 로드마다)
setup_auto_sync()

# 대화 보드 렌더링
render_conversation_board()

render_offline_panel()