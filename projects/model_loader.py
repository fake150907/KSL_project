from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Any

import mediapipe as mp
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.models.model_sequence import build_sequence_model
from src.utils.config import load_config
from config import Config

try:
    config = load_config(Config.SIGN_CONFIG)
    print(f"Loaded config from {Config.SIGN_CONFIG}")
except Exception as exc:
    print(f"Config load failed: {exc}, using defaults")
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

ROOT_DIR = Path(__file__).resolve().parents[2]
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


def _load_sequence_checkpoint(path: Path) -> tuple[torch.nn.Module, list[str]]:
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
            bundle.get("config", {}).get("data", {}).get("sequence_length", config["data"]["sequence_length"])
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
                model, labels = _load_sequence_checkpoint(path)
                sequence_models[model_key] = model
                sequence_labels[model_key] = labels
                print(f"Loaded {model_key} from {path}")
        except Exception as exc:
            print(f"Failed to load {model_key}: {exc}")

    if "cnn_gru" not in sequence_models and SEQUENCE_MODEL.exists():
        try:
            model, labels = _load_sequence_checkpoint(SEQUENCE_MODEL)
            sequence_models["cnn_gru"] = model
            sequence_labels["cnn_gru"] = labels
            print(f"Loaded cnn_gru fallback from {SEQUENCE_MODEL}")
        except Exception as exc:
            print(f"Failed to load fallback model: {exc}")
