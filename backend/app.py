from __future__ import annotations

import io
import json
import os
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_from_directory, session
from flask_cors import CORS

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import mediapipe as mp
except ImportError:
    mp = None

try:
    import torch
except ImportError:
    torch = None

try:
    from PIL import Image
except ImportError:
    Image = None

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from auth.routes import auth_bp, login_required
from config import Config
from notification.routes import notification_bp
from src.services.gloss_to_text_service import gloss_to_text
from src.utils.config import load_config
from summary.routes import summary_bp

try:
    from src.data.keypoint_utils import mediapipe_landmarks_to_frame, sequence_to_tensor
    from src.models.model_sequence import build_sequence_model
except ImportError as exc:
    print(f"Optional vision imports failed: {exc}")
    mediapipe_landmarks_to_frame = None
    sequence_to_tensor = None
    build_sequence_model = None


app = Flask(__name__)
app.secret_key = Config.FLASK_SECRET_KEY
CORS(app, supports_credentials=True)
app.register_blueprint(auth_bp)
app.register_blueprint(summary_bp)
app.register_blueprint(notification_bp)

memory_windows: dict[str, list[np.ndarray]] = {}
memory_misses: dict[str, int] = {}
memory_predictions: dict[str, deque] = {}
gloss_to_text_last_called_at: dict[str, float] = {}
GLOSS_TO_TEXT_MIN_INTERVAL_SECONDS = 6.0

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = os.environ.get("SIGN_CONFIG", str(ROOT_DIR / "config" / "web_demo.yaml"))

try:
    config = load_config(CONFIG_PATH)
    print(f"Loaded config from {CONFIG_PATH}")
except Exception as exc:
    print(f"Config load failed from {CONFIG_PATH}: {exc}, using defaults")
    config = {
        "paths": {"checkpoints_dir": "src/models"},
        "data": {"sequence_length": 32},
        "preprocess": {"feature_dims": 3, "normalize": True},
        "realtime": {
            "landmark_layout": "mediapipe_xyz",
            "confidence_threshold": 0.35,
            "stable_min_count": 1,
            "max_missing_frames": 3,
            "temperature": 0.9,
            "tta_enabled": True,
        },
    }

CHECKPOINT_DIR = ROOT_DIR / config.get("paths", {}).get("checkpoints_dir", "outputs/checkpoints")
MODEL_FILES = {
    "cnn_gru": CHECKPOINT_DIR / "best_cnngru_3000.pt",
}

sequence_models: dict[str, torch.nn.Module] = {}
sequence_labels: dict[str, list[str]] = {}
mp_holistic: Any = None
mp_holistic_instance: Any = None
mp_holistic_lock = threading.Lock()
mp_drawing: Any = None


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


def load_label_map(path: Path) -> list[str]:
    idx_to_label_path = ROOT_DIR / "model_results" / "idx_to_label_3000.json"
    if idx_to_label_path.exists():
        with idx_to_label_path.open("r", encoding="utf-8") as f:
            idx_to_label = json.load(f)
        labels: list[str] = []
        for key, value in sorted(idx_to_label.items(), key=lambda item: int(item[0])):
            if isinstance(value, dict):
                labels.append(str(value.get("label") or value.get("word_id") or key))
            else:
                labels.append(str(value))
        return labels

    label_map_path = ROOT_DIR / "model_results" / "label_map_word_id_3000.json"
    if not label_map_path.exists():
        label_map_path = path.with_name("label_map.json")
    if not label_map_path.exists():
        label_map_path = ROOT_DIR / "model_results" / "label_map.json"
    with label_map_path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)
    return [label for label, _ in sorted(label_map.items(), key=lambda item: int(item[1]))]


def load_sequence_checkpoint(path: Path) -> tuple[torch.nn.Module, list[str]]:
    if torch is None or build_sequence_model is None:
        raise RuntimeError("Vision model dependencies are not installed")
    bundle = torch.load(path, map_location="cpu")
    if isinstance(bundle, dict) and "model_state_dict" in bundle:
        state_dict = bundle["model_state_dict"]
        labels = load_label_map(path)
        model_type = "cnn_gru_3000"
        input_size = int(bundle.get("feature_dim", 225))
        num_classes = int(bundle.get("num_classes", len(labels)))
        train_config = {
            "hidden_size": 128,
            "num_layers": 2,
            "dropout": 0.3,
            "model_type": model_type,
            "conv_channels": 64,
        }
    elif isinstance(bundle, dict) and "state_dict" in bundle:
        state_dict = bundle["state_dict"]
        labels = [str(x) for x in bundle["labels"]]
        train_config = bundle.get("config", {}).get("train", {})
        model_type = str(bundle.get("model_type", train_config.get("model_type", "rnn")))
        input_size = int(bundle["input_size"])
        num_classes = len(labels)
    else:
        state_dict = bundle
        labels = load_label_map(path)
        train_config = {"hidden_size": 64, "dropout": 0.5, "model_type": "cnn_1d", "conv_channels": 64}
        model_type = "cnn_1d"
        input_size = int(state_dict["cnn.0.weight"].shape[1])
        num_classes = int(state_dict["classifier.3.weight"].shape[0])

    model = build_sequence_model(
        input_size=input_size,
        num_classes=num_classes,
        hidden_size=int(train_config.get("hidden_size", 64)),
        num_layers=int(train_config.get("num_layers", 1)),
        dropout=float(train_config.get("dropout", 0.1)),
        rnn_type=str(train_config.get("rnn_type", "gru")),
        model_type=model_type,
        conv_channels=int(train_config.get("conv_channels", 128)),
        num_heads=int(train_config.get("num_heads", 4)),
        sequence_length=int(
            bundle.get("config", {}).get("data", {}).get("sequence_length", config["data"]["sequence_length"])
            if isinstance(bundle, dict)
            else config["data"]["sequence_length"]
        ),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model, labels


def load_models() -> None:
    global sequence_models, sequence_labels, mp_holistic, mp_holistic_instance, mp_drawing

    if mp is None or torch is None or build_sequence_model is None:
        print("Vision dependencies are not installed; /api/predict will return 503")
        return

    try:
        mp_holistic = mp.solutions.holistic.Holistic
        mp_holistic_instance = mp_holistic(
            model_complexity=0,
            smooth_landmarks=False,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2,
        )
        mp_drawing = mp.solutions.drawing_utils
        print("Loaded MediaPipe")
    except Exception as exc:
        print(f"Failed to load MediaPipe: {exc}")

    for model_key, path in MODEL_FILES.items():
        try:
            if path.exists():
                model, labels = load_sequence_checkpoint(path)
                sequence_models[model_key] = model
                sequence_labels[model_key] = labels
                print(f"Loaded {model_key} model from {path}")
        except Exception as exc:
            print(f"Failed to load {model_key} model from {path}: {exc}")


def get_session_window(client_id: str) -> list[np.ndarray]:
    if client_id not in memory_windows:
        memory_windows[client_id] = []
    return memory_windows[client_id]


def get_session_misses(client_id: str) -> int:
    return int(memory_misses.get(client_id, 0))


def set_session_misses(client_id: str, misses: int) -> None:
    memory_misses[client_id] = max(0, misses)


def get_prediction_history(client_id: str) -> deque:
    if client_id not in memory_predictions:
        memory_predictions[client_id] = deque(maxlen=5)
    return memory_predictions[client_id]


def summarize_prediction_history(history: deque, min_count: int = 1) -> tuple[str | None, float]:
    if not history:
        return None, 0.0
    counts = Counter(item["label"] for item in history)
    label, count = counts.most_common(1)[0]
    if count < min_count:
        latest = history[-1]
        return latest["label"], float(latest["confidence"])
    confidences = [float(item["confidence"]) for item in history if item["label"] == label]
    return label, float(np.mean(confidences)) if confidences else 0.0


def normalize_model_type(model_type: str) -> str:
    aliases = {
        "sequence": "cnn_gru",
        "cnn-gru": "cnn_gru",
        "cnn+gru": "cnn_gru",
        "cnn_gru": "cnn_gru",
        "cnn_1d": "cnn_1d",
        "cnn-1d": "cnn_1d",
        "1d-cnn": "cnn_1d",
        "lstm": "lstm",
    }
    return aliases.get(model_type.lower(), "cnn_gru")


def get_sequence_model_bundle(model_type: str) -> tuple[Any, list[str] | None]:
    selected = normalize_model_type(model_type)
    if selected not in sequence_models:
        selected = "cnn_gru"
    model = sequence_models.get(selected)
    labels = sequence_labels.get(selected)
    return model, labels


def top_predictions_from_probs(
    probs: np.ndarray,
    labels: list[str],
    top_k: int = 3,
) -> tuple[str | None, float, list[dict[str, float | str]]]:
    if probs.size == 0 or not labels:
        return None, 0.0, []
    pred = int(np.argmax(probs))
    top_idx = np.argsort(probs)[::-1][:top_k]
    top = [{"label": labels[i], "confidence": float(probs[i])} for i in top_idx]
    return labels[pred], float(probs[pred]), top


def predict_live(
    model_type: str,
    tensor: np.ndarray,
    temperature: float | None = None,
) -> tuple[str | None, float, list[dict[str, float | str]]]:
    model, labels = get_sequence_model_bundle(model_type)
    if not model or not labels:
        return None, 0.0, []
    with torch.no_grad():
        logits = model(torch.tensor(tensor[None, ...], dtype=torch.float32))[0].numpy()
    temp = max(float(temperature or config.get("realtime", {}).get("temperature", 1.0)), 1e-6)
    probs = softmax(logits / temp)
    return top_predictions_from_probs(probs, labels)


def align_tensor_features(tensor: np.ndarray, expected: int) -> np.ndarray:
    if expected <= 0 or tensor.shape[1] == expected:
        return tensor
    if tensor.shape[1] > expected:
        return tensor[:, :expected]
    aligned = np.zeros((tensor.shape[0], expected), dtype=tensor.dtype)
    aligned[:, : tensor.shape[1]] = tensor
    return aligned


def smooth_segment_frames(frames: list[np.ndarray]) -> list[np.ndarray]:
    if len(frames) < 3:
        return frames
    smoothed: list[np.ndarray] = []
    for idx, frame in enumerate(frames):
        if idx == 0 or idx == len(frames) - 1:
            smoothed.append(frame)
            continue
        prev_frame = frames[idx - 1]
        next_frame = frames[idx + 1]
        smoothed.append((prev_frame * 0.2 + frame * 0.6 + next_frame * 0.2).astype(np.float32))
    return smoothed


def center_trim_segment_frames(frames: list[np.ndarray], trim_ratio: float = 0.1) -> list[np.ndarray]:
    if len(frames) < 12:
        return frames
    trim = max(1, int(round(len(frames) * trim_ratio)))
    if len(frames) - trim * 2 < 8:
        return frames
    return frames[trim:-trim]


def build_segment_tta_variants(frames: list[np.ndarray]) -> list[list[np.ndarray]]:
    variants = [frames, smooth_segment_frames(frames)]
    trimmed = center_trim_segment_frames(frames)
    if len(trimmed) != len(frames):
        variants.append(trimmed)
        variants.append(smooth_segment_frames(trimmed))
    return variants


def frames_to_model_tensor(model_type: str, frames: list[np.ndarray]) -> np.ndarray:
    sequence_length = int(config["data"]["sequence_length"])
    tensor = sequence_to_tensor(frames, sequence_length)
    selected_model = sequence_models.get(model_type) or sequence_models.get("cnn_gru")
    if selected_model is not None:
        expected_features = int(selected_model.input_size)
        tensor = align_tensor_features(tensor, expected_features)
    return tensor


def predict_sequence_frames(
    model_type: str,
    frames: list[np.ndarray],
    temperature: float | None = None,
    use_tta: bool | None = None,
) -> tuple[str | None, float, list[dict[str, float | str]], int]:
    model, labels = get_sequence_model_bundle(model_type)
    if not model or not labels:
        return None, 0.0, [], 0

    temp = max(float(temperature or config.get("realtime", {}).get("temperature", 1.0)), 1e-6)
    tta_enabled = bool(config.get("realtime", {}).get("tta_enabled", True)) if use_tta is None else bool(use_tta)
    variants = build_segment_tta_variants(frames) if tta_enabled else [smooth_segment_frames(frames)]
    probs_list: list[np.ndarray] = []

    with torch.no_grad():
        for variant in variants:
            tensor = frames_to_model_tensor(model_type, variant)
            logits = model(torch.tensor(tensor[None, ...], dtype=torch.float32))[0].numpy()
            probs_list.append(softmax(logits / temp))

    if not probs_list:
        return None, 0.0, [], 0
    avg_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
    label, conf, top = top_predictions_from_probs(avg_probs, labels)
    return label, conf, top, len(probs_list)


def generate_mjpeg_frames(camera_index: int = 0):
    if cv2 is None or np is None:
        yield b"--frame\r\nContent-Type: text/plain\r\n\r\nCamera dependencies are not installed\r\n"
        return

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "Camera not available",
            (130, 240),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        ok, buffer = cv2.imencode(".jpg", frame)
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        return

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
    finally:
        cap.release()


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/api/gloss_to_text", methods=["POST"])
def api_gloss_to_text():
    data = request.get_json(force=True, silent=True) or {}
    client_id = str(data.get("client_id") or request.remote_addr or "default")
    now = time.monotonic()
    last_called_at = gloss_to_text_last_called_at.get(client_id, 0.0)
    elapsed = now - last_called_at
    if elapsed < GLOSS_TO_TEXT_MIN_INTERVAL_SECONDS:
        retry_after = round(GLOSS_TO_TEXT_MIN_INTERVAL_SECONDS - elapsed, 2)
        return (
            jsonify(
                {
                    "error": "gloss_to_text API는 6초에 한 번만 호출할 수 있습니다.",
                    "retry_after": retry_after,
                }
            ),
            429,
        )
    gloss_to_text_last_called_at[client_id] = now

    gloss_value = data.get("gloss", "")
    if isinstance(gloss_value, list):
        gloss = " + ".join(str(item).strip() for item in gloss_value if str(item).strip())
    else:
        gloss = str(gloss_value).strip()
    if not gloss:
        return jsonify({"error": "gloss field is required"}), 400

    provider = data.get("provider")
    model = data.get("model")
    result = gloss_to_text(
        gloss,
        provider=str(provider).strip() if provider else None,
        model=str(model).strip() if model else None,
    )
    return jsonify({"gloss": gloss, "text": result}), 200


@app.route("/api/kakao/login", methods=["GET"])
def kakao_login():
    from urllib.parse import urlencode
    from flask import redirect as flask_redirect

    redirect_uri = request.args.get("redirect_uri", "")
    if not Config.KAKAO_REST_API_KEY:
        return jsonify({"error": "KAKAO_REST_API_KEY가 설정되어 있지 않습니다."}), 500
    if not redirect_uri:
        return jsonify({"error": "redirect_uri가 필요합니다."}), 400

    params = urlencode({"client_id": Config.KAKAO_REST_API_KEY, "redirect_uri": redirect_uri, "response_type": "code"})
    return flask_redirect(f"https://kauth.kakao.com/oauth/authorize?{params}")


@app.route("/api/kakao/token", methods=["POST"])
def kakao_token():
    data = request.get_json(silent=True) or {}
    code = str(data.get("code", "")).strip()
    redirect_uri = str(data.get("redirect_uri", "")).strip()

    if not Config.KAKAO_REST_API_KEY:
        return jsonify({"error": "KAKAO_REST_API_KEY가 설정되어 있지 않습니다."}), 500
    if not code:
        return jsonify({"error": "카카오 인가 코드(code)가 필요합니다."}), 400
    if not redirect_uri:
        return jsonify({"error": "redirect_uri가 필요합니다."}), 400

    try:
        import requests

        token_payload = {
            "grant_type": "authorization_code",
            "client_id": Config.KAKAO_REST_API_KEY,
            "redirect_uri": redirect_uri,
            "code": code,
        }
        if Config.KAKAO_CLIENT_SECRET:
            token_payload["client_secret"] = Config.KAKAO_CLIENT_SECRET

        response = requests.post(
            "https://kauth.kakao.com/oauth/token",
            headers={"Content-Type": "application/x-www-form-urlencoded;charset=utf-8"},
            data=token_payload,
            timeout=10,
        )
        result = response.json()
        if response.status_code != 200:
            error_msg = result.get("error_description") or result.get("msg") or result.get("error") or str(result)
            return jsonify({"error": str(error_msg)}), response.status_code
        return jsonify(
            {
                "access_token": result.get("access_token"),
                "refresh_token": result.get("refresh_token"),
                "expires_in": result.get("expires_in"),
            }
        ), 200
    except Exception as exc:
        return jsonify({"error": f"카카오 토큰 발급 실패: {exc}"}), 502


@app.route("/validation_demos/<path:filename>", methods=["GET"])
def validation_demo_video(filename: str):
    return send_from_directory(ROOT_DIR / "data" / "raw" / "validation_mp4", filename)


@app.route("/video_feed", methods=["GET"])
@login_required
def video_feed():
    camera_index = int(request.args.get("camera", 0))
    return Response(
        generate_mjpeg_frames(camera_index),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    session.modified = False
    request_started_at = time.perf_counter()
    try:
        if cv2 is None or Image is None or mp is None or torch is None or mediapipe_landmarks_to_frame is None or sequence_to_tensor is None:
            return jsonify({"error": "Vision dependencies are not installed"}), 503
        if mp_holistic_instance is None:
            return jsonify({"error": "MediaPipe is not available"}), 503
        if "frame" not in request.files:
            return jsonify({"error": "No frame provided"}), 400

        model_type = normalize_model_type(request.form.get("model_type", "sequence"))
        landmark_layout = request.form.get("landmark_layout", "mediapipe_xyz")
        if landmark_layout != "mediapipe_xyz":
            return jsonify({"error": f"Unsupported landmark_layout: {landmark_layout}"}), 400

        client_id = request.form.get("client_id", "default")
        frame_id = request.form.get("frame_id", "")
        confidence_threshold = float(
            request.form.get(
                "confidence_threshold",
                config.get("realtime", {}).get("confidence_threshold", 0.35),
            )
        )
        requested_window_size = int(
            request.form.get(
                "window_size",
                config.get("realtime", {}).get("window_size", config["data"]["sequence_length"]),
            )
        )
        window_size = max(8, min(requested_window_size, int(config["data"]["sequence_length"])))
        stable_min_count = int(
            request.form.get("stable_min_count", config.get("realtime", {}).get("stable_min_count", 2))
        )
        max_missing_frames = int(
            request.form.get("max_missing_frames", config.get("realtime", {}).get("max_missing_frames", 3))
        )
        min_segment_frames = int(
            request.form.get("min_segment_frames", config.get("realtime", {}).get("min_segment_frames", 8))
        )
        temperature = float(request.form.get("temperature", config.get("realtime", {}).get("temperature", 0.9)))
        use_tta = request.form.get("tta_enabled", str(config.get("realtime", {}).get("tta_enabled", True))).lower() not in {
            "false",
            "0",
            "no",
        }
        run_model = request.form.get("run_model", "true").lower() != "false"

        frame_file = request.files["frame"]
        image_pil = Image.open(io.BytesIO(frame_file.read())).convert("RGB")
        image_rgb = np.array(image_pil)
        height, width = image_rgb.shape[:2]
        scale = min(640 / width, 480 / height, 1)
        if scale < 1:
            image_rgb = cv2.resize(
                image_rgb,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        processed_height, processed_width = image_rgb.shape[:2]

        with mp_holistic_lock:
            results = mp_holistic_instance.process(image_rgb)

        has_hand = bool(results.left_hand_landmarks or results.right_hand_landmarks)
        has_pose = bool(results.pose_landmarks)
        landmarks = {"left_hand": [], "right_hand": [], "pose": []}

        if results.left_hand_landmarks:
            landmarks["left_hand"] = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        if results.right_hand_landmarks:
            landmarks["right_hand"] = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        if results.pose_landmarks:
            landmarks["pose"] = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]

        window = get_session_window(client_id)
        prediction: dict[str, Any] = {
            "label": None,
            "confidence": 0.0,
            "has_hand": has_hand,
            "has_pose": has_pose,
            "landmarks": landmarks,
            "window_progress": len(window),
            "window_size": window_size,
            "missing_frames": get_session_misses(client_id),
            "max_missing_frames": max_missing_frames,
            "min_segment_frames": min_segment_frames,
            "temperature": temperature,
            "tta_enabled": use_tta,
            "top_predictions": [],
            "frame_id": frame_id,
            "landmark_layout": landmark_layout,
            "model_type": model_type,
            "input_size": {"width": int(width), "height": int(height)},
            "processed_size": {"width": int(processed_width), "height": int(processed_height)},
        }

        if has_hand:
            set_session_misses(client_id, 0)
            prediction["missing_frames"] = 0
            frame_points = mediapipe_landmarks_to_frame(results)
            window.append(frame_points)
            prediction["window_progress"] = len(window)
            prediction["window_size"] = window_size
            prediction["window_filled"] = False
            prediction["segmenting"] = True
            prediction["status"] = "수어 단어 구간 수집 중"
        else:
            misses = get_session_misses(client_id) + 1
            set_session_misses(client_id, misses)
            prediction["missing_frames"] = misses
            if misses > max_missing_frames:
                segment_frames = list(window)
                window.clear()
                memory_predictions.pop(client_id, None)
                prediction["window_progress"] = 0
                prediction["segmenting"] = False

                if len(segment_frames) >= min_segment_frames:
                    label, conf, top_predictions, tta_count = predict_sequence_frames(
                        model_type,
                        segment_frames,
                        temperature=temperature,
                        use_tta=use_tta,
                    )
                    prediction["top_predictions"] = top_predictions
                    prediction["raw_label"] = label
                    prediction["raw_confidence"] = conf
                    prediction["tta_count"] = tta_count
                    prediction["confidence"] = conf
                    prediction["window_filled"] = True
                    prediction["segment_finalized"] = True
                    prediction["segment_frames"] = len(segment_frames)

                    if label and conf >= confidence_threshold:
                        prediction["label"] = label
                        prediction["below_threshold"] = False
                    else:
                        prediction["label"] = None
                        prediction["below_threshold"] = True
                        prediction["status"] = "수어 구간은 잡혔지만 신뢰도가 낮습니다."
                elif segment_frames:
                    prediction["segment_finalized"] = True
                    prediction["segment_frames"] = len(segment_frames)
                    prediction["status"] = "수어 구간이 너무 짧아 예측하지 않았습니다."
            else:
                prediction["window_progress"] = len(window)

        top_log = json.dumps(prediction.get("top_predictions", []), ensure_ascii=False)
        process_ms = (time.perf_counter() - request_started_at) * 1000
        prediction["process_ms"] = round(process_ms, 1)
        print(
            "Prediction: "
            f"frame_id={frame_id}, "
            f"process_ms={process_ms:.1f}, "
            f"input={width}x{height}, "
            f"processed={processed_width}x{processed_height}, "
            f"label={prediction.get('label')}, "
            f"conf={prediction.get('confidence'):.2f}, "
            f"raw={prediction.get('raw_label')}/{prediction.get('raw_confidence')}, "
            f"has_hand={prediction.get('has_hand')}, "
            f"has_pose={prediction.get('has_pose')}, "
            f"window={prediction.get('window_progress')}/{prediction.get('window_size')}, "
            f"miss={prediction.get('missing_frames')}/{prediction.get('max_missing_frames')}, "
            f"window_filled={prediction.get('window_filled')}, "
            f"temp={prediction.get('temperature')}, "
            f"tta={prediction.get('tta_count')}, "
            f"top={top_log}",
            flush=True,
        )
        return jsonify({"frame_id": frame_id, "prediction": prediction}), 200
    except Exception as exc:
        print(f"Prediction error: {exc}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    load_models()
    app.run(debug=False, host="127.0.0.1", port=5000, use_reloader=False, threaded=True)
