"""Role: realtime webcam sign-expression inference with MediaPipe.

Input: webcam frames
Output: predicted expression overlaid on screen
Example:
  python -m src.inference.realtime_sign_inference --model baseline
  python -m src.inference.realtime_sign_inference --model baseline --camera_index 1
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.data.keypoint_utils import mediapipe_landmarks_to_frame, sequence_to_tensor
from src.inference.infer import predict_baseline, predict_sequence
from src.services.tts_service import speak_text
from src.utils.config import load_config


def stable_label(history: deque[str], min_count: int = 3) -> str:
    if not history:
        return ""
    label, count = Counter(history).most_common(1)[0]
    return label if count >= min_count else history[-1]


def expected_feature_count(checkpoint: Path, model_type: str, sequence_length: int) -> int | None:
    try:
        if model_type == "baseline":
            import joblib

            bundle = joblib.load(checkpoint)
            model = bundle.get("model")
            total = getattr(model, "n_features_in_", None)
            if total:
                return int(total) // sequence_length
        meta_path = Path("data/processed/sign_word_subset.meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return int(meta.get("feature_count"))
    except Exception:
        return None
    return None


def align_tensor_features(tensor: np.ndarray, expected: int | None) -> np.ndarray:
    if expected is None or tensor.shape[1] == expected:
        return tensor
    if tensor.shape[1] > expected:
        return tensor[:, :expected]
    aligned = np.zeros((tensor.shape[0], expected), dtype=tensor.dtype)
    aligned[:, : tensor.shape[1]] = tensor
    return aligned


def draw_text(frame: np.ndarray, text: str, position: tuple[int, int], color: tuple[int, int, int] = (30, 220, 30)) -> np.ndarray:
    if not text:
        return frame

    font_candidates = [
        Path("C:/Windows/Fonts/malgunbd.ttf"),
        Path("C:/Windows/Fonts/malgun.ttf"),
    ]
    font = None
    for candidate in font_candidates:
        if candidate.exists():
            font = ImageFont.truetype(str(candidate), 32)
            break

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


def open_camera(camera_index: int) -> cv2.VideoCapture:
    backends = []
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    backends.append(None)

    for backend in backends:
        cap = cv2.VideoCapture(camera_index, backend) if backend is not None else cv2.VideoCapture(camera_index)
        if cap.isOpened():
            return cap
        cap.release()
    raise RuntimeError(f"Could not open webcam index {camera_index}.")


def run_realtime(config: dict, model_type: str = "baseline", speak: bool = False, camera_index: int = 0) -> None:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("mediapipe is required for realtime inference. Install requirements.txt.") from exc

    cap = open_camera(camera_index)

    mp_holistic = mp.solutions.holistic
    drawer = mp.solutions.drawing_utils
    window = deque(maxlen=int(config["realtime"]["window_size"]))
    predictions = deque(maxlen=5)
    last_spoken = ""
    last_spoken_at = 0.0
    checkpoint = Path(config["paths"]["checkpoints_dir"]) / ("baseline.joblib" if model_type == "baseline" else "sequence_model.pt")
    seq_len = int(config["data"]["sequence_length"])
    expected_features = expected_feature_count(checkpoint, model_type, seq_len)
    preprocess_config = config.get("preprocess", {})
    landmark_layout = str(preprocess_config.get("landmark_layout", "mediapipe_xyz"))
    normalize_mode = str(preprocess_config.get("normalize_mode", "mediapipe_xyz_body"))

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            frame_points = mediapipe_landmarks_to_frame(results, layout=landmark_layout)
            if frame_points.size:
                window.append(frame_points)
            if len(window) == window.maxlen:
                tensor = sequence_to_tensor(list(window), seq_len, normalize_mode=normalize_mode)
                tensor = align_tensor_features(tensor, expected_features)
                label, conf = (
                    predict_baseline(str(checkpoint), tensor)
                    if model_type == "baseline"
                    else predict_sequence(str(checkpoint), tensor)
                )
                if conf >= float(config["realtime"]["confidence_threshold"]):
                    predictions.append(label)
                shown = stable_label(predictions)
                frame = draw_text(frame, f"{shown} ({conf:.2f})", (20, 20))
                now = time.time()
                if speak and shown and shown != last_spoken and now - last_spoken_at > float(config["realtime"]["cooldown_seconds"]):
                    speak_text(shown, backend=config["services"]["tts_backend"])
                    last_spoken = shown
                    last_spoken_at = now

            if results.pose_landmarks:
                drawer.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                drawer.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                drawer.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            cv2.imshow("Sign MVP Realtime", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", choices=["baseline", "sequence"], default="baseline")
    parser.add_argument("--speak", action="store_true")
    parser.add_argument("--camera_index", type=int, default=0)
    args = parser.parse_args()
    run_realtime(load_config(args.config), args.model, args.speak, args.camera_index)


if __name__ == "__main__":
    main()