from __future__ import annotations

import io
import json
import os
import re
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request, send_from_directory, session
from flask_cors import CORS

VISION_DISABLED = (
    os.environ.get("DISABLE_VISION", "").lower() in {"1", "true", "yes"}
)

if VISION_DISABLED:
    np = None
    cv2 = None
    mp = None
    torch = None
    Image = None
else:
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

if VISION_DISABLED:
    mediapipe_landmarks_to_frame = None
    sequence_to_tensor = None
    build_sequence_model = None
else:
    try:
        from src.data.keypoint_utils import mediapipe_landmarks_to_frame, sequence_to_tensor
    except ImportError as exc:
        print(f"Optional keypoint imports failed: {exc}")
        mediapipe_landmarks_to_frame = None
        sequence_to_tensor = None

    try:
        from src.models.model_sequence import build_sequence_model
    except ImportError as exc:
        print(f"Optional sequence model imports failed: {exc}")
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
patient_session_lock = threading.Lock()
patient_session: dict[str, Any] = {
    "waiting": False,
    "patientData": None,
    "updatedAt": None,
}
chat_messages_lock = threading.Lock()
chat_messages: list[dict[str, Any]] = []
session_state_lock = threading.Lock()
session_state: dict[str, Any] = {
    "ended": False,
    "updatedAt": None,
}

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
    # 이전 best_cnngru_3000.pt 폐기됨 (2026-05-20). v7 운영 ckpt 2개만 유지.
    "word_v2":      CHECKPOINT_DIR / "word_stage2.pt",          # ⭐ 운영 WORD
    "sentence_v2":  CHECKPOINT_DIR / "sentence_scenario12.pt",  # ⭐ 운영 SENTENCE (demo 12-class)
}

# ⭐ 시나리오 클래스 indices (운영 시 restricted 마스킹용)
SCENARIO_SEN_IDS = [
    "SEN0109", "SEN0110", "SEN0169", "SEN0170", "SEN0175", "SEN0176",
    "SEN0278", "SEN0279", "SEN0322", "SEN0354", "SEN0355", "SEN1817",
]
SCENARIO_SEN_INDICES: list[int] = []
SCENARIO_WORD_IDS = ["WORD0579", "WORD0602", "WORD1174", "WORD1282",
                     "WORD2317", "WORD2318", "WORD2492", "WORD2493"]
SCENARIO_WORD_INDICES: list[int] = []  # load_models에서 label_map 로드 후 채움
SCENARIO_LOOKUP: dict[str, str] = {}   # load_models에서 scenario_lookup.json 로드 후 채움
LABEL_DISPLAY_MAP: dict[str, str] = {}
SINGLE_SENTENCE_MIN_CONFIDENCE = 0.55
PAIR_MIN_FUSION_SCORE = 0.35
PAIR_MIN_WORD_CONFIDENCE = 0.20
PAIR_MIN_SENTENCE_CONFIDENCE = 0.12

# Scenario fusion priors from the 2026-05-20 scenario12 fine-tune report.
# Unknown WORD scenario classes use the WORD model's overall validation Top-1.
SCENARIO_DEFAULT_WORD_ACC = 0.9655
SCENARIO_DEFAULT_SENTENCE_ACC = 0.8722
SCENARIO_LABEL_ACC: dict[str, float] = {
    "SEN0109": 1.00,
    "SEN0110": 0.40,
    "SEN0169": 1.00,
    "SEN0170": 0.667,
    "SEN0175": 1.00,
    "SEN0176": 0.867,
    "SEN0278": 0.667,
    "SEN0279": 1.00,
    "SEN0322": 0.933,
    "SEN0354": 0.933,
    "SEN0355": 1.00,
    "SEN1817": 1.00,
}

sequence_models: dict[str, torch.nn.Module] = {}
sequence_labels: dict[str, list[str]] = {}
mp_holistic: Any = None
mp_holistic_instance: Any = None
mp_holistic_lock = threading.Lock()
model_load_lock = threading.Lock()
model_load_attempted = False
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


def load_label_map_for_ckpt(path: Path) -> list[str]:
    """⭐ ckpt 이름에 'sentence' 포함되면 SEN label_map, 그 외엔 WORD label_map."""
    name = path.stem.lower()
    if "scenario12" in name:
        sen12_map_path = ROOT_DIR / "model_results" / "label_map_sen_scenario12.json"
        if sen12_map_path.exists():
            with sen12_map_path.open("r", encoding="utf-8") as f:
                sen12_map = json.load(f)
            return [label for label, _ in sorted(sen12_map.items(), key=lambda x: int(x[1]))]
    if "sentence" in name or "_sen" in name:
        sen_map_path = ROOT_DIR / "model_results" / "label_map_sen_id_2000.json"
        if sen_map_path.exists():
            with sen_map_path.open("r", encoding="utf-8") as f:
                sen_map = json.load(f)
            return [label for label, _ in sorted(sen_map.items(), key=lambda x: int(x[1]))]
    word_map_path = ROOT_DIR / "model_results" / "label_map_word_id_3000.json"
    if word_map_path.exists():
        with word_map_path.open("r", encoding="utf-8") as f:
            word_map = json.load(f)
        return [label for label, _ in sorted(word_map.items(), key=lambda x: int(x[1]))]
    return load_label_map(path)


def load_label_display_map() -> dict[str, str]:
    display: dict[str, str] = {}
    idx_to_label_path = ROOT_DIR / "model_results" / "idx_to_label_3000.json"
    if idx_to_label_path.exists():
        try:
            with idx_to_label_path.open("r", encoding="utf-8") as f:
                idx_to_label = json.load(f)
            for value in idx_to_label.values():
                if isinstance(value, dict):
                    word_id = str(value.get("word_id") or "").strip()
                    label = str(value.get("label") or "").strip()
                    if word_id and label:
                        display[word_id] = label
        except Exception as exc:
            print(f"Failed to load WORD display labels: {exc}")
    sen_display_path = ROOT_DIR / "model_results" / "idx_to_label_sen_2000.json"
    if sen_display_path.exists():
        try:
            with sen_display_path.open("r", encoding="utf-8") as f:
                sen_display = json.load(f)
            for key, value in sen_display.items():
                if isinstance(value, dict):
                    sen_id = str(value.get("sen_id") or value.get("label_id") or key).strip()
                    label = str(value.get("label") or value.get("text") or "").strip()
                else:
                    sen_id = str(key).strip()
                    label = str(value).strip()
                if sen_id.isdigit():
                    sen_id = f"SEN{int(sen_id) + 1:04d}"
                if sen_id and label:
                    display[sen_id] = label
        except Exception as exc:
            print(f"Failed to load SENTENCE display labels: {exc}")
    sen12_display_path = ROOT_DIR / "model_results" / "idx_to_label_sen_scenario12.json"
    if sen12_display_path.exists():
        try:
            with sen12_display_path.open("r", encoding="utf-8") as f:
                sen12_display = json.load(f)
            for key, value in sen12_display.items():
                if isinstance(value, dict):
                    sen_id = str(value.get("sen_id") or value.get("label") or key).strip()
                    label = str(value.get("display") or value.get("text") or "").strip()
                else:
                    sen_id = str(key).strip()
                    label = str(value).strip()
                if sen_id and label:
                    display[sen_id] = label
        except Exception as exc:
            print(f"Failed to load scenario12 display labels: {exc}")
    return display


def display_label_for(label: str | None) -> str | None:
    if not label:
        return None
    if label.startswith("SEN") and label in SCENARIO_LOOKUP:
        return SCENARIO_LOOKUP[label]
    return LABEL_DISPLAY_MAP.get(label, SCENARIO_LOOKUP.get(label, label))


def scenario_label_acc(label: str | None) -> float:
    if not label:
        return 0.0
    if label in SCENARIO_LABEL_ACC:
        return SCENARIO_LABEL_ACC[label]
    if label.startswith("WORD"):
        return SCENARIO_DEFAULT_WORD_ACC
    if label.startswith("SEN"):
        return SCENARIO_DEFAULT_SENTENCE_ACC
    return 0.5


def load_sequence_checkpoint(path: Path) -> tuple[torch.nn.Module, list[str]]:
    if torch is None or build_sequence_model is None:
        raise RuntimeError("Vision model dependencies are not installed")
    bundle = torch.load(path, map_location="cpu")

    # ⭐ 신규 분기: 피어나 v7 CNNGRUAttn ckpt 형식
    if isinstance(bundle, dict) and "model_state" in bundle and "model_config" in bundle:
        try:
            from src.models.cnngru_attn import CNNGRUAttn
        except ImportError as exc:
            raise RuntimeError(f"CNNGRUAttn import failed: {exc}")
        model = CNNGRUAttn(**bundle["model_config"])
        model.load_state_dict(bundle["model_state"])
        model.eval()
        # frames_to_model_tensor 호환을 위해 input_size 속성 노출
        try:
            model.input_size = int(bundle["model_config"].get("input_dim", 225))
        except Exception:
            model.input_size = 225
        if isinstance(bundle.get("labels"), (list, tuple)):
            labels = [str(label) for label in bundle["labels"]]
        else:
            labels = load_label_map_for_ckpt(path)
        return model, labels

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
    global sequence_models, sequence_labels, mp_holistic, mp_holistic_instance, mp_drawing, LABEL_DISPLAY_MAP

    if mp is None:
        print("MediaPipe is not installed; /api/predict will return 503")
        return

    try:
        mp_holistic = mp.solutions.holistic.Holistic
        # ⭐ v7 §J.4: 학습-추론 옵션 일치 (model_complexity=1, smooth=True, conf=0.3)
        mp_holistic_instance = mp_holistic(
            static_image_mode=False,
            model_complexity=1,            # 0 → 1
            smooth_landmarks=True,         # False → True
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.3,  # 0.2 → 0.3
            min_tracking_confidence=0.3,   # 0.2 → 0.3
        )
        mp_drawing = mp.solutions.drawing_utils
        print("Loaded MediaPipe (v7 학습-추론 일치 옵션 적용)")
    except Exception as exc:
        print(f"Failed to load MediaPipe: {exc}")

    if torch is None or build_sequence_model is None:
        print("Sequence model dependencies are not installed; /api/predict will return landmarks only")
        return

    for model_key, path in MODEL_FILES.items():
        try:
            if path.exists():
                model, labels = load_sequence_checkpoint(path)
                sequence_models[model_key] = model
                sequence_labels[model_key] = labels
                print(f"Loaded {model_key} model from {path}")
        except Exception as exc:
            print(f"Failed to load {model_key} model from {path}: {exc}")

    # ⭐ 기존 프론트 호환: model_type="cnn_gru" / "sequence" 요청도 운영 WORD(word_v2)로 라우팅
    if "word_v2" in sequence_models:
        sequence_models["cnn_gru"] = sequence_models["word_v2"]
        sequence_labels["cnn_gru"] = sequence_labels["word_v2"]

    # ⭐ 시나리오 class indices 계산 (label_map 기반)
    global SCENARIO_WORD_INDICES, SCENARIO_SEN_INDICES, SCENARIO_LOOKUP
    word_labels = sequence_labels.get("word_v2") or sequence_labels.get("cnn_gru") or []
    if word_labels:
        word_label_to_idx = {label: i for i, label in enumerate(word_labels)}
        SCENARIO_WORD_INDICES = [word_label_to_idx[w] for w in SCENARIO_WORD_IDS if w in word_label_to_idx]
        print(f"Scenario WORD indices ({len(SCENARIO_WORD_INDICES)}/{len(SCENARIO_WORD_IDS)}): {SCENARIO_WORD_INDICES}")
    else:
        print("Warning: WORD label_map not available, SCENARIO_WORD_INDICES empty")

    sen_labels = sequence_labels.get("sentence_v2") or []
    if sen_labels:
        sen_label_to_idx = {label: i for i, label in enumerate(sen_labels)}
        SCENARIO_SEN_INDICES = [sen_label_to_idx[s] for s in SCENARIO_SEN_IDS if s in sen_label_to_idx]
        print(f"Scenario SEN indices ({len(SCENARIO_SEN_INDICES)}/{len(SCENARIO_SEN_IDS)}): {SCENARIO_SEN_INDICES}")
    else:
        print("Warning: SENTENCE label_map not available, SCENARIO_SEN_INDICES empty")

    # ⭐ scenario_lookup.json 로드
    lookup_path = Path(__file__).resolve().parent / "inference" / "scenario_lookup.json"
    if lookup_path.exists():
        try:
            with lookup_path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
            # 메타키 (_로 시작)는 제외하고 실제 lookup만 추출
            SCENARIO_LOOKUP = {k: v for k, v in raw.items() if not k.startswith("_")}
            print(f"Loaded scenario_lookup with {len(SCENARIO_LOOKUP)} entries")
        except Exception as exc:
            print(f"Failed to load scenario_lookup: {exc}")

    LABEL_DISPLAY_MAP = load_label_display_map()
    print(f"Loaded display labels with {len(LABEL_DISPLAY_MAP)} entries")


def ensure_models_loaded() -> None:
    global model_load_attempted
    if model_load_attempted:
        return
    with model_load_lock:
        if model_load_attempted:
            return
        load_models()
        model_load_attempted = True


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
        "word_v2": "word_v2",
        "sentence_v2": "sentence_v2",
        "word": "word_v2",
        "sentence": "sentence_v2",
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
    if model_type in sequence_models:
        return sequence_models.get(model_type), sequence_labels.get(model_type)
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
    top = [
        {
            "label": labels[i],
            "display_label": display_label_for(labels[i]),
            "confidence": float(probs[i]),
        }
        for i in top_idx
    ]
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
    restrict_indices: list[int] | None = None,  # ⭐ v7 시나리오 restricted 마스킹용
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
            # ⭐ Restricted 마스킹: 시나리오 class만 비교 (v7 핵심)
            if restrict_indices:
                mask = np.full_like(logits, -np.inf)
                for idx in restrict_indices:
                    if 0 <= idx < logits.shape[0]:
                        mask[idx] = 0.0
                logits = logits + mask
            probs_list.append(softmax(logits / temp))

    if not probs_list:
        return None, 0.0, [], 0
    avg_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
    label, conf, top = top_predictions_from_probs(avg_probs, labels)
    return label, conf, top, len(probs_list)


def predict_dual_scenario(
    frames: list[np.ndarray],
    temperature: float | None = None,
    use_tta: bool | None = None,
) -> dict[str, Any]:
    """⭐ v7 운영: WORD + SENTENCE 병렬 추론 + 시나리오 restricted Top-3 + lookup.

    반환:
      {
        "word":     {"label": "WORD0579", "confidence": 0.78, "top": [...]},
        "sentence": {"label": "SEN0322",  "confidence": 0.85, "top": [...]},
        "scenario_text": "복지카드를 잃어버렸어요",
        "lookup_hit": True,
      }
    """
    out: dict[str, Any] = {}

    # WORD
    if "word_v2" in sequence_models and SCENARIO_WORD_INDICES:
        w_label, w_conf, w_top, _ = predict_sequence_frames(
            "word_v2", frames, temperature=temperature, use_tta=use_tta,
            restrict_indices=SCENARIO_WORD_INDICES,
        )
        out["word"] = {
            "label": w_label,
            "display_label": display_label_for(w_label),
            "confidence": w_conf,
            "acc_prior": scenario_label_acc(w_label),
            "top": w_top,
        }
    else:
        out["word"] = {"label": None, "display_label": None, "confidence": 0.0, "acc_prior": 0.0, "top": []}

    # SENTENCE
    if "sentence_v2" in sequence_models and SCENARIO_SEN_INDICES:
        s_label, s_conf, s_top, _ = predict_sequence_frames(
            "sentence_v2", frames, temperature=temperature, use_tta=use_tta,
            restrict_indices=SCENARIO_SEN_INDICES,
        )
        out["sentence"] = {
            "label": s_label,
            "display_label": display_label_for(s_label),
            "confidence": s_conf,
            "acc_prior": scenario_label_acc(s_label),
            "top": s_top,
        }
    else:
        out["sentence"] = {"label": None, "display_label": None, "confidence": 0.0, "acc_prior": 0.0, "top": []}

    word_candidates = [
        item for item in (out["word"].get("top") or [])
        if item.get("label")
    ] or [
        {
            "label": out["word"]["label"],
            "display_label": out["word"].get("display_label"),
            "confidence": out["word"]["confidence"],
            "acc_prior": out["word"].get("acc_prior", 0.0),
        }
    ]
    sentence_candidates = [
        item for item in (out["sentence"].get("top") or [])
        if item.get("label")
    ] or [
        {
            "label": out["sentence"]["label"],
            "display_label": out["sentence"].get("display_label"),
            "confidence": out["sentence"]["confidence"],
            "acc_prior": out["sentence"].get("acc_prior", 0.0),
        }
    ]

    lookup_candidates: list[dict[str, Any]] = []
    for word_item in word_candidates[:3]:
        w_label = str(word_item.get("label") or "")
        if not w_label:
            continue
        w_conf = float(word_item.get("confidence") or 0.0)
        w_acc = scenario_label_acc(w_label)
        word_item["acc_prior"] = w_acc
        for sentence_item in sentence_candidates[:3]:
            s_label = str(sentence_item.get("label") or "")
            if not s_label:
                continue
            s_conf = float(sentence_item.get("confidence") or 0.0)
            s_acc = scenario_label_acc(s_label)
            sentence_item["acc_prior"] = s_acc
            for key in (f"{s_label}+{w_label}", f"{w_label}+{s_label}"):
                if key in SCENARIO_LOOKUP:
                    lookup_candidates.append({
                        "key": key,
                        "text": SCENARIO_LOOKUP[key],
                        "score": round(((w_conf * w_acc) * (s_conf * s_acc)) ** 0.5, 6),
                        "source": "word_sentence_pair",
                        "word": word_item,
                        "sentence": sentence_item,
                    })
                    break

    for left_index, left_item in enumerate(word_candidates[:3]):
        left_label = str(left_item.get("label") or "")
        if not left_label:
            continue
        left_conf = float(left_item.get("confidence") or 0.0)
        left_acc = scenario_label_acc(left_label)
        left_item["acc_prior"] = left_acc
        for right_item in word_candidates[left_index + 1:3]:
            right_label = str(right_item.get("label") or "")
            if not right_label or right_label == left_label:
                continue
            right_conf = float(right_item.get("confidence") or 0.0)
            right_acc = scenario_label_acc(right_label)
            right_item["acc_prior"] = right_acc
            for key in (f"{left_label}+{right_label}", f"{right_label}+{left_label}"):
                if key in SCENARIO_LOOKUP:
                    lookup_candidates.append({
                        "key": key,
                        "text": SCENARIO_LOOKUP[key],
                        "score": round(((left_conf * left_acc) * (right_conf * right_acc)) ** 0.5, 6),
                        "source": "word_word_pair",
                        "word": left_item,
                        "word_secondary": right_item,
                    })
                    break

    for source, items in (("single_sentence", sentence_candidates[:3]), ("single_word", word_candidates[:3])):
        for item in items:
            label = str(item.get("label") or "")
            if label in SCENARIO_LOOKUP:
                conf = float(item.get("confidence") or 0.0)
                if source == "single_sentence" and conf < SINGLE_SENTENCE_MIN_CONFIDENCE:
                    continue
                acc = scenario_label_acc(label)
                item["acc_prior"] = acc
                lookup_candidates.append({
                    "key": label,
                    "text": SCENARIO_LOOKUP[label],
                    "score": round(conf * acc, 6),
                    "source": source,
                    "word": item if source == "single_word" else None,
                    "sentence": item if source == "single_sentence" else None,
                })

    lookup_candidates.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    best_sentence_single = next(
        (
            item for item in lookup_candidates
            if item.get("source") == "single_sentence"
            and float(item.get("score") or 0.0) >= 0.20
        ),
        None,
    )
    best_word_single = next(
        (item for item in lookup_candidates if item.get("source") == "single_word"),
        None,
    )
    if best_sentence_single and best_word_single:
        sentence_score = float(best_sentence_single.get("score") or 0.0)
        word_score = float(best_word_single.get("score") or 0.0)
        raw_word_label = str((best_word_single.get("word") or {}).get("label") or "")
        if raw_word_label in SCENARIO_WORD_IDS and word_score < 0.75 and sentence_score >= word_score * 0.35:
            lookup_candidates = [
                best_sentence_single,
                *[item for item in lookup_candidates if item is not best_sentence_single],
            ]
    pair_candidates = [
        item for item in lookup_candidates
        if item.get("source") in {"word_sentence_pair", "word_word_pair"}
    ]
    if pair_candidates:
        eligible_pairs = []
        for item in pair_candidates:
            score = float(item.get("score") or 0.0)
            word_conf = float((item.get("word") or {}).get("confidence") or 0.0)
            if item.get("source") == "word_word_pair":
                second_conf = float((item.get("word_secondary") or {}).get("confidence") or 0.0)
                is_eligible = (
                    score >= PAIR_MIN_FUSION_SCORE
                    and word_conf >= PAIR_MIN_WORD_CONFIDENCE
                    and second_conf >= PAIR_MIN_WORD_CONFIDENCE
                )
            else:
                second_conf = float((item.get("sentence") or {}).get("confidence") or 0.0)
                is_eligible = (
                    score >= PAIR_MIN_FUSION_SCORE
                    and word_conf >= PAIR_MIN_WORD_CONFIDENCE
                    and second_conf >= PAIR_MIN_SENTENCE_CONFIDENCE
                )
            if is_eligible:
                eligible_pairs.append(item)
        best_pair = eligible_pairs[0] if eligible_pairs else None
    else:
        best_pair = None

    if best_pair:
        best_pair_score = float(best_pair.get("score") or 0.0)
        pair_word_conf = float((best_pair.get("word") or {}).get("confidence") or 0.0)
        pair_sentence_conf = float((best_pair.get("sentence") or {}).get("confidence") or 0.0)
        pair_word_secondary_conf = float((best_pair.get("word_secondary") or {}).get("confidence") or 0.0)
        best_pair["rule"] = {
            "fusion_score": best_pair_score,
            "min_fusion_score": PAIR_MIN_FUSION_SCORE,
            "word_confidence": pair_word_conf,
            "min_word_confidence": PAIR_MIN_WORD_CONFIDENCE,
            "sentence_confidence": pair_sentence_conf,
            "min_sentence_confidence": PAIR_MIN_SENTENCE_CONFIDENCE,
            "word_secondary_confidence": pair_word_secondary_conf,
        }
        lookup_candidates = [
            best_pair,
            *[item for item in lookup_candidates if item is not best_pair],
        ]
    out["fusion_candidates"] = lookup_candidates[:5]
    if lookup_candidates:
        best = lookup_candidates[0]
        out["scenario_text"] = best["text"]
        out["lookup_hit"] = True
        out["lookup_key"] = best["key"]
        out["lookup_source"] = best["source"]
        out["lookup_score"] = best["score"]
    else:
        out["scenario_text"] = None
        out["lookup_hit"] = False

    return out


def landmarks_payload_to_frame(landmarks: dict[str, Any]) -> np.ndarray:
    if np is None:
        raise RuntimeError("NumPy is not installed")

    layout = [
        ("pose", 33),
        ("left_hand", 21),
        ("right_hand", 21),
    ]
    points: list[list[float]] = []
    for key, expected_count in layout:
        raw_points = landmarks.get(key) if isinstance(landmarks, dict) else None
        if not isinstance(raw_points, list):
            raw_points = []
        for idx in range(expected_count):
            raw_point = raw_points[idx] if idx < len(raw_points) else None
            if isinstance(raw_point, list):
                xyz = [float(raw_point[i] or 0.0) if i < len(raw_point) else 0.0 for i in range(3)]
            elif isinstance(raw_point, dict):
                xyz = [float(raw_point.get(axis, 0.0) or 0.0) for axis in ("x", "y", "z")]
            else:
                xyz = [0.0, 0.0, 0.0]
            points.append(xyz)
    return np.asarray(points, dtype=np.float32)


def landmarks_have_points(points: Any) -> bool:
    if not isinstance(points, list):
        return False
    for point in points:
        if isinstance(point, list) and any(abs(float(value or 0.0)) > 1e-9 for value in point[:2]):
            return True
        if isinstance(point, dict) and (
            abs(float(point.get("x", 0.0) or 0.0)) > 1e-9
            or abs(float(point.get("y", 0.0) or 0.0)) > 1e-9
        ):
            return True
    return False


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


@app.route("/api/patient-session", methods=["GET", "POST", "DELETE"])
def api_patient_session():
    if request.method == "GET":
        with patient_session_lock:
            return jsonify(dict(patient_session)), 200

    if request.method == "DELETE":
        with patient_session_lock:
            patient_session.update({"waiting": False, "patientData": None, "updatedAt": None})
            return jsonify(dict(patient_session)), 200

    data = request.get_json(silent=True) or {}
    patient_data = data.get("patientData") or data.get("patient_data") or data
    if not isinstance(patient_data, dict):
        return jsonify({"error": "patientData must be an object"}), 400

    normalized = {
        "name": str(patient_data.get("name", "")).strip(),
        "dob": str(patient_data.get("dob", "")).strip(),
        "gender": str(patient_data.get("gender", "")).strip(),
        "phone": str(patient_data.get("phone", "")).strip(),
    }
    if not normalized["name"] or not normalized["phone"]:
        return jsonify({"error": "patient name and phone are required"}), 400

    with patient_session_lock:
        patient_session.update(
            {
                "waiting": True,
                "patientData": normalized,
                "updatedAt": int(time.time()),
            }
        )
        return jsonify(dict(patient_session)), 200


@app.route("/api/messages", methods=["GET", "POST", "DELETE"])
def api_messages():
    if request.method == "GET":
        with chat_messages_lock:
            return jsonify({"messages": list(chat_messages)}), 200

    if request.method == "DELETE":
        with chat_messages_lock:
            chat_messages.clear()
        return jsonify({"messages": []}), 200

    data = request.get_json(silent=True) or {}
    message_id = str(data.get("id", "")).strip()
    sender = str(data.get("sender", "")).strip()
    text = str(data.get("text", "")).strip()
    if not message_id or sender not in {"patient", "doctor"} or not text:
        return jsonify({"error": "id, sender, and text are required"}), 400

    message = {
        "id": message_id,
        "sender": sender,
        "text": text,
        "timestamp": str(data.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        "label": str(data.get("label", "")).strip(),
    }
    with chat_messages_lock:
        if not any(item.get("id") == message_id for item in chat_messages):
            chat_messages.append(message)
            del chat_messages[:-200]
    return jsonify({"message": message}), 200


@app.route("/api/session-state", methods=["GET", "POST", "DELETE"])
def api_session_state():
    if request.method == "GET":
        with session_state_lock:
            return jsonify(dict(session_state)), 200

    with session_state_lock:
        if request.method == "DELETE":
            session_state.update({"ended": False, "updatedAt": None})
        else:
            data = request.get_json(silent=True) or {}
            session_state.update({"ended": bool(data.get("ended")), "updatedAt": int(time.time())})
        return jsonify(dict(session_state)), 200


@app.route("/", methods=["GET"])
def root():
    return jsonify({"status": "ok", "service": "ksl-backend"}), 200


@app.route("/api/gloss_to_text", methods=["POST"])
def api_gloss_to_text():
    ensure_models_loaded()
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

    gloss_tokens = [token.strip() for token in re.split(r"\s*\+\s*", gloss) if token.strip()]
    lookup_candidates: list[str] = []
    if len(gloss_tokens) >= 2:
        for left in gloss_tokens:
            for right in gloss_tokens:
                if left == right:
                    continue
                lookup_candidates.append(f"{left}+{right}")
    lookup_candidates.extend(gloss_tokens)
    for key in lookup_candidates:
        if key in SCENARIO_LOOKUP:
            return jsonify({
                "gloss": gloss,
                "text": SCENARIO_LOOKUP[key],
                "lookup_hit": True,
                "lookup_key": key,
            }), 200

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
        ensure_models_loaded()
        if cv2 is None or Image is None or np is None or mp is None:
            return jsonify({"error": "Vision dependencies are not installed"}), 503
        if mp_holistic_instance is None:
            return jsonify({"error": "MediaPipe is not available"}), 503
        force_finalize = request.form.get("force_finalize", "false").lower() in {"true", "1", "yes"}
        if not force_finalize and "frame" not in request.files:
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
        sequence_length = int(config["data"]["sequence_length"])
        window_size = max(8, min(requested_window_size, sequence_length))
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
        upload_bytes = int(float(request.form.get("upload_bytes", 0) or 0))
        scenario_mode = request.form.get("scenario_mode", "false").lower() in {"true", "1", "yes", "resident"}
        demo_video_time_sec = request.form.get("demo_video_time_sec")
        demo_segment_start_sec = request.form.get("demo_segment_start_sec")
        demo_finalize_reason = request.form.get("demo_finalize_reason") or request.form.get("live_finalize_reason")
        if force_finalize:
            window = get_session_window(client_id)
            prediction: dict[str, Any] = {
                "label": None,
                "confidence": 0.0,
                "has_hand": None,
                "has_pose": None,
                "landmarks": {"left_hand": [], "right_hand": [], "pose": []},
                "window_progress": len(window),
                "window_size": window_size,
                "missing_frames": get_session_misses(client_id),
                "max_missing_frames": max_missing_frames,
                "min_segment_frames": min_segment_frames,
                "segment_frames": None,
                "sequence_length": sequence_length,
                "temperature": temperature,
                "tta_enabled": use_tta,
                "run_model": run_model,
                "force_finalize": True,
                "top_predictions": [],
                "frame_id": frame_id,
                "landmark_layout": landmark_layout,
                "model_type": model_type,
                "scenario_mode": scenario_mode,
                "processing_mode": "server_mediapipe",
                "upload_bytes": upload_bytes,
                "input_size": None,
                "processed_size": None,
                "demo_video_time_sec": demo_video_time_sec,
                "demo_segment_start_sec": demo_segment_start_sec,
                "demo_finalize_reason": demo_finalize_reason,
            }

            segment_frames = list(window)
            window.clear()
            set_session_misses(client_id, 0)
            memory_predictions.pop(client_id, None)
            prediction["window_progress"] = 0
            prediction["segmenting"] = False
            prediction["missing_frames"] = 0

            if len(segment_frames) >= min_segment_frames:
                label, conf, top_predictions, tta_count = predict_sequence_frames(
                    model_type,
                    segment_frames,
                    temperature=temperature,
                    use_tta=use_tta,
                )
                prediction["top_predictions"] = top_predictions
                prediction["raw_label"] = label
                prediction["display_label"] = display_label_for(label)
                prediction["raw_confidence"] = conf
                prediction["tta_count"] = tta_count
                prediction["confidence"] = conf
                prediction["window_filled"] = True
                prediction["segment_finalized"] = True
                prediction["segment_frames"] = len(segment_frames)

                # ⭐ v7 운영: 시나리오 dual model 추론 + Top-3 + lookup
                if scenario_mode and ("word_v2" in sequence_models or "sentence_v2" in sequence_models):
                    try:
                        dual = predict_dual_scenario(segment_frames, temperature=temperature, use_tta=use_tta)
                        prediction["scenario"] = dual
                        if dual.get("scenario_text"):
                            prediction["scenario_text"] = dual["scenario_text"]
                    except Exception as exc:
                        prediction["scenario_error"] = str(exc)

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

            process_ms = (time.perf_counter() - request_started_at) * 1000
            prediction["process_ms"] = round(process_ms, 1)
            print(
                "ForceFinalize: "
                f"frame_id={frame_id}, "
                f"video_time={demo_video_time_sec}, "
                f"segment_start={demo_segment_start_sec}, "
                f"reason={demo_finalize_reason}, "
                f"process_ms={process_ms:.1f}, "
                f"label={prediction.get('label')}, "
                f"conf={prediction.get('confidence'):.2f}, "
                f"raw={prediction.get('raw_label')}/{prediction.get('raw_confidence')}, "
                f"segment_frames={prediction.get('segment_frames')}, "
                f"min_segment_frames={prediction.get('min_segment_frames')}, "
                f"tta={prediction.get('tta_count')}, "
                f"scenario={json.dumps(prediction.get('scenario'), ensure_ascii=False)}, "
                f"top={json.dumps(prediction.get('top_predictions', []), ensure_ascii=False)}",
                flush=True,
            )
            return jsonify({"prediction": prediction, "frame_id": frame_id})

        frame_file = request.files["frame"]
        image_pil = Image.open(io.BytesIO(frame_file.read())).convert("RGB")
        image_rgb = np.array(image_pil)
        height, width = image_rgb.shape[:2]
        scale = min(640 / width, 360 / height, 1)
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
            "segment_frames": None,
            "sequence_length": sequence_length,
            "temperature": temperature,
            "tta_enabled": use_tta,
            "run_model": run_model,
            "force_finalize": force_finalize,
            "top_predictions": [],
            "frame_id": frame_id,
            "landmark_layout": landmark_layout,
            "model_type": model_type,
            "scenario_mode": scenario_mode,
            "processing_mode": "server_mediapipe",
            "upload_bytes": upload_bytes,
            "input_size": {"width": int(width), "height": int(height)},
            "processed_size": {"width": int(processed_width), "height": int(processed_height)},
        }

        can_collect_sequence = mediapipe_landmarks_to_frame is not None

        if has_hand and can_collect_sequence:
            set_session_misses(client_id, 0)
            prediction["missing_frames"] = 0
            frame_points = mediapipe_landmarks_to_frame(results)
            window.append(frame_points)
            prediction["window_progress"] = len(window)
            prediction["window_size"] = window_size
            prediction["window_filled"] = False
            prediction["segmenting"] = True
            prediction["status"] = "수어 단어 구간 수집 중"
        elif has_hand:
            set_session_misses(client_id, 0)
            prediction["missing_frames"] = 0
            prediction["window_progress"] = len(window)
            prediction["window_filled"] = False
            prediction["segmenting"] = True
            prediction["status"] = "MediaPipe 랜드마크 감지 중"
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
                    prediction["display_label"] = display_label_for(label)
                    prediction["raw_confidence"] = conf
                    prediction["tta_count"] = tta_count
                    prediction["confidence"] = conf
                    prediction["window_filled"] = True
                    prediction["segment_finalized"] = True
                    prediction["segment_frames"] = len(segment_frames)

                    # ⭐ v7 운영: 시나리오 dual model 추론 + Top-3 + lookup
                    if scenario_mode and ("word_v2" in sequence_models or "sentence_v2" in sequence_models):
                        try:
                            dual = predict_dual_scenario(segment_frames, temperature=temperature, use_tta=use_tta)
                            prediction["scenario"] = dual
                            if dual.get("scenario_text"):
                                prediction["scenario_text"] = dual["scenario_text"]
                        except Exception as exc:
                            prediction["scenario_error"] = str(exc)

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
            f"segment_frames={prediction.get('segment_frames')}, "
            f"min_segment_frames={prediction.get('min_segment_frames')}, "
            f"sequence_length={prediction.get('sequence_length')}, "
            f"miss={prediction.get('missing_frames')}/{prediction.get('max_missing_frames')}, "
            f"window_filled={prediction.get('window_filled')}, "
            f"run_model={prediction.get('run_model')}, "
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


@app.route("/api/predict_landmarks", methods=["POST"])
def predict_landmarks():
    request_started_at = time.perf_counter()
    try:
        ensure_models_loaded()
        if np is None or torch is None or sequence_to_tensor is None:
            return jsonify({"error": "Model dependencies are not installed"}), 503

        data = request.get_json(silent=True) or {}
        force_finalize = str(data.get("force_finalize", "false")).lower() in {"true", "1", "yes"}
        landmarks = data.get("landmarks") or {}
        if not force_finalize and not isinstance(landmarks, dict):
            return jsonify({"error": "landmarks must be an object"}), 400

        model_type = normalize_model_type(str(data.get("model_type", "sequence")))
        landmark_layout = str(data.get("landmark_layout", "mediapipe_xyz"))
        if landmark_layout != "mediapipe_xyz":
            return jsonify({"error": f"Unsupported landmark_layout: {landmark_layout}"}), 400

        client_id = str(data.get("client_id", "default"))
        frame_id = str(data.get("frame_id", ""))
        confidence_threshold = float(
            data.get(
                "confidence_threshold",
                config.get("realtime", {}).get("confidence_threshold", 0.35),
            )
        )
        requested_window_size = int(
            data.get(
                "window_size",
                config.get("realtime", {}).get("window_size", config["data"]["sequence_length"]),
            )
        )
        sequence_length = int(config["data"]["sequence_length"])
        window_size = max(8, min(requested_window_size, sequence_length))
        stable_min_count = int(data.get("stable_min_count", config.get("realtime", {}).get("stable_min_count", 2)))
        max_missing_frames = int(data.get("max_missing_frames", config.get("realtime", {}).get("max_missing_frames", 3)))
        min_segment_frames = int(data.get("min_segment_frames", config.get("realtime", {}).get("min_segment_frames", 8)))
        temperature = float(data.get("temperature", config.get("realtime", {}).get("temperature", 0.9)))
        use_tta = str(data.get("tta_enabled", config.get("realtime", {}).get("tta_enabled", True))).lower() not in {
            "false",
            "0",
            "no",
        }
        run_model = str(data.get("run_model", "true")).lower() != "false"
        scenario_mode = str(data.get("scenario_mode", "false")).lower() in {"true", "1", "yes", "resident"}
        client_mediapipe_ms = data.get("client_mediapipe_ms")
        client_payload_bytes = int(float(data.get("client_payload_bytes", 0) or 0))
        demo_video_time_sec = data.get("demo_video_time_sec")
        demo_segment_start_sec = data.get("demo_segment_start_sec")
        demo_finalize_reason = data.get("demo_finalize_reason") or data.get("live_finalize_reason")

        window = get_session_window(client_id)
        prediction: dict[str, Any] = {
            "label": None,
            "confidence": 0.0,
            "has_hand": None if force_finalize else False,
            "has_pose": None if force_finalize else False,
            "landmarks": landmarks if isinstance(landmarks, dict) else {"left_hand": [], "right_hand": [], "pose": []},
            "window_progress": len(window),
            "window_size": window_size,
            "missing_frames": get_session_misses(client_id),
            "max_missing_frames": max_missing_frames,
            "min_segment_frames": min_segment_frames,
            "segment_frames": None,
            "sequence_length": sequence_length,
            "temperature": temperature,
            "tta_enabled": use_tta,
            "run_model": run_model,
            "force_finalize": force_finalize,
            "top_predictions": [],
            "frame_id": frame_id,
            "landmark_layout": landmark_layout,
            "model_type": model_type,
            "scenario_mode": scenario_mode,
            "processing_mode": "client_mediapipe",
            "client_mediapipe_ms": client_mediapipe_ms,
            "upload_bytes": client_payload_bytes,
            "stable_min_count": stable_min_count,
        }

        if force_finalize:
            prediction["demo_video_time_sec"] = demo_video_time_sec
            prediction["demo_segment_start_sec"] = demo_segment_start_sec
            prediction["demo_finalize_reason"] = demo_finalize_reason
            segment_frames = list(window)
            window.clear()
            set_session_misses(client_id, 0)
            memory_predictions.pop(client_id, None)
            prediction["window_progress"] = 0
            prediction["segmenting"] = False
            prediction["missing_frames"] = 0

            if len(segment_frames) >= min_segment_frames:
                label, conf, top_predictions, tta_count = predict_sequence_frames(
                    model_type,
                    segment_frames,
                    temperature=temperature,
                    use_tta=use_tta,
                )
                prediction["top_predictions"] = top_predictions
                prediction["raw_label"] = label
                prediction["display_label"] = display_label_for(label)
                prediction["raw_confidence"] = conf
                prediction["tta_count"] = tta_count
                prediction["confidence"] = conf
                prediction["window_filled"] = True
                prediction["segment_finalized"] = True
                prediction["segment_frames"] = len(segment_frames)
                # ⭐ v7 운영: 시나리오 dual model 추론 + Top-3 + lookup
                if scenario_mode and ("word_v2" in sequence_models or "sentence_v2" in sequence_models):
                    try:
                        dual = predict_dual_scenario(segment_frames, temperature=temperature, use_tta=use_tta)
                        prediction["scenario"] = dual
                        if dual.get("scenario_text"):
                            prediction["scenario_text"] = dual["scenario_text"]
                    except Exception as exc:
                        prediction["scenario_error"] = str(exc)
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
            has_hand = landmarks_have_points(landmarks.get("left_hand")) or landmarks_have_points(landmarks.get("right_hand"))
            has_pose = landmarks_have_points(landmarks.get("pose"))
            prediction["has_hand"] = has_hand
            prediction["has_pose"] = has_pose

            if has_hand:
                set_session_misses(client_id, 0)
                prediction["missing_frames"] = 0
                window.append(landmarks_payload_to_frame(landmarks))
                prediction["window_progress"] = len(window)
                prediction["window_size"] = window_size
                prediction["window_filled"] = False
                prediction["segmenting"] = True
                prediction["status"] = "클라이언트 MediaPipe 랜드마크 수집 중"
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
                        prediction["display_label"] = display_label_for(label)
                        prediction["raw_confidence"] = conf
                        prediction["tta_count"] = tta_count
                        prediction["confidence"] = conf
                        prediction["window_filled"] = True
                        prediction["segment_finalized"] = True
                        prediction["segment_frames"] = len(segment_frames)
                        # ⭐ v7 운영: 시나리오 dual model 추론 + Top-3 + lookup
                        if scenario_mode and ("word_v2" in sequence_models or "sentence_v2" in sequence_models):
                            try:
                                dual = predict_dual_scenario(segment_frames, temperature=temperature, use_tta=use_tta)
                                prediction["scenario"] = dual
                                if dual.get("scenario_text"):
                                    prediction["scenario_text"] = dual["scenario_text"]
                            except Exception as exc:
                                prediction["scenario_error"] = str(exc)
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

        process_ms = (time.perf_counter() - request_started_at) * 1000
        prediction["process_ms"] = round(process_ms, 1)
        print(
            "ClientMediaPipe: "
            f"frame_id={frame_id}, "
            f"process_ms={process_ms:.1f}, "
            f"label={prediction.get('label')}, "
            f"conf={prediction.get('confidence'):.2f}, "
            f"has_hand={prediction.get('has_hand')}, "
            f"has_pose={prediction.get('has_pose')}, "
            f"window={prediction.get('window_progress')}/{prediction.get('window_size')}, "
            f"segment_frames={prediction.get('segment_frames')}, "
            f"force_finalize={force_finalize}",
            flush=True,
        )
        return jsonify({"frame_id": frame_id, "prediction": prediction}), 200
    except Exception as exc:
        print(f"Client MediaPipe prediction error: {exc}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    load_models()
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port, use_reloader=False, threaded=True)
