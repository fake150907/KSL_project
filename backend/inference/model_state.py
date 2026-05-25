from __future__ import annotations

import os
import threading
from collections import deque
from pathlib import Path
from typing import Any

VISION_DISABLED = os.environ.get("DISABLE_VISION", "").lower() in {"1", "true", "yes"}

if VISION_DISABLED:
    np = cv2 = mp = torch = Image = None
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

if VISION_DISABLED:
    mediapipe_landmarks_to_frame = sequence_to_tensor = build_sequence_model = None
else:
    try:
        from src.data.keypoint_utils import mediapipe_landmarks_to_frame, sequence_to_tensor
    except ImportError as exc:
        print(f"Optional keypoint imports failed: {exc}")
        mediapipe_landmarks_to_frame = sequence_to_tensor = None

    try:
        from src.models.model_sequence import build_sequence_model
    except ImportError as exc:
        print(f"Optional sequence model imports failed: {exc}")
        build_sequence_model = None

sequence_models: dict[str, Any] = {}
sequence_labels: dict[str, list[str]] = {}

mp_holistic: Any = None
mp_holistic_instance: Any = None
mp_holistic_lock = threading.Lock()
mp_drawing: Any = None

model_load_lock = threading.Lock()
model_load_attempted = False

memory_windows: dict[str, list] = {}
memory_misses: dict[str, int] = {}
memory_predictions: dict[str, deque] = {}
gloss_to_text_last_called_at: dict[str, float] = {}

SCENARIO_SEN_IDS = [
    "SEN0109", "SEN0110", "SEN0169", "SEN0170", "SEN0175", "SEN0176",
    "SEN0278", "SEN0279", "SEN0322", "SEN0354", "SEN0355", "SEN1817",
]
SCENARIO_WORD_IDS = [
    "WORD0579", "WORD0602", "WORD1174", "WORD1282",
    "WORD2317", "WORD2318", "WORD2492", "WORD2493",
]
SCENARIO_SEN_INDICES: list[int] = []
SCENARIO_WORD_INDICES: list[int] = []
SCENARIO_LOOKUP: dict[str, str] = {}
LABEL_DISPLAY_MAP: dict[str, str] = {}

SCENARIO_DEFAULT_WORD_ACC = 0.9655
SCENARIO_DEFAULT_SENTENCE_ACC = 0.8722
SCENARIO_LABEL_ACC: dict[str, float] = {
    "SEN0109": 1.00, "SEN0110": 0.40,  "SEN0169": 1.00,
    "SEN0170": 0.667,"SEN0175": 1.00,  "SEN0176": 0.867,
    "SEN0278": 0.667,"SEN0279": 1.00,  "SEN0322": 0.933,
    "SEN0354": 0.933,"SEN0355": 1.00,  "SEN1817": 1.00,
}

GLOSS_TO_TEXT_MIN_INTERVAL_SECONDS = 6.0
