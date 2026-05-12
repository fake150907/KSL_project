from __future__ import annotations

import io
import json
import time
from collections import Counter, deque
from typing import Any

import cv2
import numpy as np
import torch
from flask import Blueprint, Response, jsonify, request
from PIL import Image

from auth.routes import login_required
from vision.model_loader import (
    config,
    mp_holistic_instance,
    mp_holistic_lock,
    sequence_models,
    sequence_labels,
)

vision_bp = Blueprint("vision", __name__)

memory_windows: dict[str, list[np.ndarray]] = {}
memory_misses: dict[str, int] = {}
memory_predictions: dict[str, deque] = {}

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


def _normalize_model_type(model_type: str) -> str:
    aliases = {
        "sequence": "cnn_gru", "cnn-gru": "cnn_gru",
        "cnn+gru": "cnn_gru", "cnn_gru": "cnn_gru", "lstm": "lstm",
    }
    return aliases.get(model_type.lower(), "cnn_gru")


def _align_tensor_features(tensor: np.ndarray, expected: int) -> np.ndarray:
    if expected <= 0 or tensor.shape[1] == expected:
        return tensor
    if tensor.shape[1] > expected:
        return tensor[:, :expected]
    aligned = np.zeros((tensor.shape[0], expected), dtype=tensor.dtype)
    aligned[:, : tensor.shape[1]] = tensor
    return aligned


def _predict_live(model_type: str, tensor: np.ndarray) -> tuple[str | None, float, list[dict]]:
    selected = _normalize_model_type(model_type)
    if selected not in sequence_models:
        selected = "cnn_gru"
    model = sequence_models.get(selected)
    labels = sequence_labels.get(selected)
    if not model or not labels:
        return None, 0.0, []
    with torch.no_grad():
        logits = model(torch.tensor(tensor[None, ...], dtype=torch.float32))[0].numpy()
    probs = _softmax(logits)
    pred = int(np.argmax(probs))
    top_idx = np.argsort(probs)[::-1][:3]
    top = [{"label": labels[i], "confidence": float(probs[i])} for i in top_idx]
    return labels[pred], float(probs[pred]), top


def _summarize_history(history: deque, min_count: int = 2) -> tuple[str | None, float]:
    if not history:
        return None, 0.0
    counts = Counter(item["label"] for item in history)
    label, count = counts.most_common(1)[0]
    if count < min_count:
        latest = history[-1]
        return latest["label"], float(latest["confidence"])
    confidences = [float(item["confidence"]) for item in history if item["label"] == label]
    return label, float(np.mean(confidences)) if confidences else 0.0

@vision_bp.route("/api/health", methods=["GET"])
def health():
    """헬스 체크 (인증 불필요)."""
    return jsonify({"status": "ok"}), 200


@vision_bp.route("/video_feed", methods=["GET"])
@login_required
def video_feed():
    def _generate(camera_index: int):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Camera not available", (130, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            return
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                ok, buf = cv2.imencode(".jpg", frame)
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        finally:
            cap.release()

    camera_index = int(request.args.get("camera", 0))
    return Response(_generate(camera_index), mimetype="multipart/x-mixed-replace; boundary=frame")


@vision_bp.route("/api/predict", methods=["POST"])
@login_required
def predict():
    from src.data.keypoint_utils import mediapipe_landmarks_to_frame, sequence_to_tensor

    t0 = time.perf_counter()
    try:
        if mp_holistic_instance is None:
            return jsonify({"error": "MediaPipe is not available"}), 503
        if "frame" not in request.files:
            return jsonify({"error": "No frame provided"}), 400

        model_type   = _normalize_model_type(request.form.get("model_type", "sequence"))
        layout       = request.form.get("landmark_layout", "mediapipe_xyz")
        if layout != "mediapipe_xyz":
            return jsonify({"error": f"Unsupported landmark_layout: {layout}"}), 400

        client_id    = request.form.get("client_id", "default")
        frame_id     = request.form.get("frame_id", "")
        conf_thresh  = float(request.form.get("confidence_threshold",
                             config.get("realtime", {}).get("confidence_threshold", 0.35)))
        window_size  = int(config["data"]["sequence_length"])
        stable_min   = int(request.form.get("stable_min_count",
                           config.get("realtime", {}).get("stable_min_count", 2)))
        max_miss     = int(request.form.get("max_missing_frames",
                           config.get("realtime", {}).get("max_missing_frames", 3)))

        image_pil = Image.open(io.BytesIO(request.files["frame"].read())).convert("RGB")
        image_rgb = np.array(image_pil)
        h, w = image_rgb.shape[:2]
        scale = min(640 / w, 480 / h, 1)
        if scale < 1:
            image_rgb = cv2.resize(image_rgb, (int(w * scale), int(h * scale)),
                                   interpolation=cv2.INTER_AREA)
        ph, pw = image_rgb.shape[:2]

        with mp_holistic_lock:
            results = mp_holistic_instance.process(image_rgb)

        has_hand  = bool(results.left_hand_landmarks or results.right_hand_landmarks)
        landmarks = {"left_hand": [], "right_hand": [], "pose": []}
        if results.left_hand_landmarks:
            landmarks["left_hand"]  = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        if results.right_hand_landmarks:
            landmarks["right_hand"] = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        if results.pose_landmarks:
            landmarks["pose"]       = [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]

        window = memory_windows.setdefault(client_id, [])
        pred: dict[str, Any] = {
            "label": None, "confidence": 0.0, "has_hand": has_hand,
            "landmarks": landmarks,
            "window_progress": len(window), "window_size": window_size,
            "missing_frames": memory_misses.get(client_id, 0),
            "max_missing_frames": max_miss,
            "top_predictions": [], "frame_id": frame_id,
            "landmark_layout": layout, "model_type": model_type,
            "input_size": {"width": w, "height": h},
            "processed_size": {"width": pw, "height": ph},
        }

        if has_hand:
            memory_misses[client_id] = 0
            pred["missing_frames"]   = 0
            window.append(mediapipe_landmarks_to_frame(results))
            if len(window) > window_size:
                window.pop(0)

            if len(window) == window_size:
                tensor = sequence_to_tensor(window, int(config["data"]["sequence_length"]))
                sel_model = sequence_models.get(model_type) or sequence_models.get("cnn_gru")
                if sel_model:
                    tensor = _align_tensor_features(tensor, int(sel_model.input_size))

                label, conf, top = _predict_live(model_type, tensor)
                pred["top_predictions"] = top
                pred["raw_label"] = label
                pred["raw_confidence"] = conf

                if label:
                    history = memory_predictions.setdefault(client_id, deque(maxlen=12))
                    history.append({"label": label, "confidence": conf})
                    stable, s_conf = _summarize_history(history, stable_min)
                else:
                    stable, s_conf = None, conf

                disp_label = stable or label
                disp_conf  = s_conf if stable else conf
                pred["confidence"] = disp_conf
                pred["label"]          = disp_label if disp_label and disp_conf >= conf_thresh else None
                pred["below_threshold"] = not bool(pred["label"])
                pred["window_filled"]  = True
            else:
                pred["window_progress"] = len(window)
        else:
            misses = memory_misses.get(client_id, 0) + 1
            memory_misses[client_id] = max(0, misses)
            pred["missing_frames"] = misses
            if misses > max_miss:
                window.clear()
                memory_predictions.pop(client_id, None)
                pred["window_progress"] = 0
            else:
                pred["window_progress"] = len(window)

        pred["process_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        print(f"Prediction: frame_id={frame_id}, label={pred.get('label')}, "
              f"conf={pred.get('confidence', 0):.2f}, has_hand={has_hand}", flush=True)
        return jsonify({"frame_id": frame_id, "prediction": pred}), 200

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500
