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

import cv2
import mediapipe as mp
import numpy as np
import torch
from flask import Flask, Response, jsonify, request, session
from flask_cors import CORS
from PIL import Image

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.keypoint_utils import mediapipe_landmarks_to_frame, sequence_to_tensor
from src.models.model_sequence import build_sequence_model
from src.utils.config import load_config


app = Flask(__name__)
app.secret_key = "sign-interpreter-secret"
CORS(app)

memory_windows: dict[str, list[np.ndarray]] = {}
memory_misses: dict[str, int] = {}
memory_predictions: dict[str, deque] = {}

CONFIG_PATH = os.environ.get("SIGN_CONFIG", "config/web_demo.yaml")

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
            "stable_min_count": 2,
            "max_missing_frames": 3,
        },
    }

ROOT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT_DIR / config.get("paths", {}).get("checkpoints_dir", "outputs/checkpoints")
WEB_MODEL_DIR = CHECKPOINT_DIR / "web_models"
SEQUENCE_MODEL = CHECKPOINT_DIR / "sequence_model.pt"
MODEL_FILES = {
    "cnn_gru": WEB_MODEL_DIR / "sequence_model_cnn_gru_FILE01_03-FILE10_12_valacc0.8571.pt",
    "lstm": WEB_MODEL_DIR / "sequence_model_lstm_FILE01_03-FILE10_12_valacc0.8095.pt",
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


def load_sequence_checkpoint(path: Path) -> tuple[torch.nn.Module, list[str]]:
    bundle = torch.load(path, map_location="cpu")
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
        sequence_length=int(
            bundle.get("config", {})
            .get("data", {})
            .get("sequence_length", config["data"]["sequence_length"])
        ),
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, [str(x) for x in bundle["labels"]]


def load_models() -> None:
    global sequence_models, sequence_labels, mp_holistic, mp_holistic_instance, mp_drawing

    try:
        mp_holistic = mp.solutions.holistic.Holistic
        mp_holistic_instance = mp_holistic(
            model_complexity=0,
            smooth_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
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

    if "cnn_gru" not in sequence_models and SEQUENCE_MODEL.exists():
        try:
            model, labels = load_sequence_checkpoint(SEQUENCE_MODEL)
            sequence_models["cnn_gru"] = model
            sequence_labels["cnn_gru"] = labels
            print(f"Loaded cnn_gru fallback model from {SEQUENCE_MODEL}")
        except Exception as exc:
            print(f"Failed to load sequence fallback model: {exc}")


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
        memory_predictions[client_id] = deque(maxlen=12)
    return memory_predictions[client_id]


def summarize_prediction_history(history: deque, min_count: int = 2) -> tuple[str | None, float]:
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
        "lstm": "lstm",
    }
    return aliases.get(model_type.lower(), "cnn_gru")


def predict_live(model_type: str, tensor: np.ndarray) -> tuple[str | None, float, list[dict[str, float | str]]]:
    selected = normalize_model_type(model_type)
    if selected not in sequence_models:
        selected = "cnn_gru"
    model = sequence_models.get(selected)
    labels = sequence_labels.get(selected)
    if not model or not labels:
        return None, 0.0, []
    with torch.no_grad():
        logits = model(torch.tensor(tensor[None, ...], dtype=torch.float32))[0].numpy()
    probs = softmax(logits)
    pred = int(np.argmax(probs))
    top_idx = np.argsort(probs)[::-1][:3]
    top = [{"label": labels[i], "confidence": float(probs[i])} for i in top_idx]
    return labels[pred], float(probs[pred]), top


def align_tensor_features(tensor: np.ndarray, expected: int) -> np.ndarray:
    if expected <= 0 or tensor.shape[1] == expected:
        return tensor
    if tensor.shape[1] > expected:
        return tensor[:, :expected]
    aligned = np.zeros((tensor.shape[0], expected), dtype=tensor.dtype)
    aligned[:, : tensor.shape[1]] = tensor
    return aligned


def generate_mjpeg_frames(camera_index: int = 0):
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


@app.route("/video_feed", methods=["GET"])
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
        window_size = int(config["data"]["sequence_length"])
        stable_min_count = int(
            request.form.get("stable_min_count", config.get("realtime", {}).get("stable_min_count", 2))
        )
        max_missing_frames = int(
            request.form.get("max_missing_frames", config.get("realtime", {}).get("max_missing_frames", 3))
        )

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
            "landmarks": landmarks,
            "window_progress": len(window),
            "window_size": window_size,
            "missing_frames": get_session_misses(client_id),
            "max_missing_frames": max_missing_frames,
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
            if len(window) > window_size:
                window.pop(0)

            if len(window) == window_size:
                sequence_length = int(config["data"]["sequence_length"])
                tensor = sequence_to_tensor(window, sequence_length)

                selected_model = sequence_models.get(model_type) or sequence_models.get("cnn_gru")
                if selected_model is not None:
                    expected_features = int(selected_model.input_size)
                    tensor = align_tensor_features(tensor, expected_features)

                label, conf, top_predictions = predict_live(model_type, tensor)
                prediction["top_predictions"] = top_predictions
                prediction["raw_label"] = label
                prediction["raw_confidence"] = conf

                if label:
                    history = get_prediction_history(client_id)
                    history.append({"label": label, "confidence": conf})
                    stable, stable_conf = summarize_prediction_history(history, stable_min_count)
                else:
                    stable, stable_conf = None, conf

                display_label = stable if stable else label
                display_confidence = stable_conf if stable else conf
                prediction["confidence"] = display_confidence
                if display_label and display_confidence >= confidence_threshold:
                    prediction["label"] = display_label
                    prediction["below_threshold"] = False
                else:
                    prediction["label"] = None
                    prediction["below_threshold"] = True
                prediction["window_filled"] = True
            else:
                prediction["window_progress"] = len(window)
                prediction["window_size"] = window_size
        else:
            misses = get_session_misses(client_id) + 1
            set_session_misses(client_id, misses)
            prediction["missing_frames"] = misses
            if misses > max_missing_frames:
                window.clear()
                memory_predictions.pop(client_id, None)
                prediction["window_progress"] = 0
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
            f"window={prediction.get('window_progress')}/{prediction.get('window_size')}, "
            f"miss={prediction.get('missing_frames')}/{prediction.get('max_missing_frames')}, "
            f"window_filled={prediction.get('window_filled')}, "
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
