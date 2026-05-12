"""Evaluate local videos with the MediaPipe-retrained sequence checkpoint.

This script is intended for the 1안(MediaPipe retraining) comparison step.
It extracts MediaPipe Holistic landmarks from mp4 files, builds 32-frame
windows, and reports final/average/majority predictions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

_MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", ".matplotlib")).resolve()
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import mediapipe as mp
import numpy as np
import pandas as pd
import torch

from src.data.keypoint_utils import mediapipe_landmarks_to_frame, sequence_to_tensor
from src.models.model_sequence import build_sequence_model
from src.utils.config import load_config


EXPECTED_BY_PREFIX = {
    "1": "가다",
    "2": "감사",
    "3": "괜찮다",
    "4": "배고프다",
    "5": "병원",
    "6": "아프다",
    "7": "우유",
    "8": "자다",
}


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


def load_sequence_checkpoint(checkpoint_path: Path) -> tuple[torch.nn.Module, list[str], int, int]:
    bundle = torch.load(checkpoint_path, map_location="cpu")
    train_config = bundle.get("config", {}).get("train", {})
    data_config = bundle.get("config", {}).get("data", {})
    sequence_length = int(data_config.get("sequence_length", 32))
    labels = [str(label) for label in bundle["labels"]]
    model = build_sequence_model(
        input_size=int(bundle["input_size"]),
        num_classes=len(labels),
        hidden_size=int(train_config.get("hidden_size", 64)),
        num_layers=int(train_config.get("num_layers", 1)),
        dropout=float(train_config.get("dropout", 0.1)),
        rnn_type=str(train_config.get("rnn_type", "gru")),
        model_type=str(bundle.get("model_type", train_config.get("model_type", "rnn"))),
        conv_channels=int(train_config.get("conv_channels", 128)),
        num_heads=int(train_config.get("num_heads", 4)),
        sequence_length=sequence_length,
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, labels, int(bundle["input_size"]), sequence_length


def expected_label_from_name(path: Path) -> str:
    prefix = path.stem.split("_", 1)[0]
    return EXPECTED_BY_PREFIX.get(prefix, "")


def extract_frames(video_path: Path, holistic: Any, frame_step: int) -> tuple[list[np.ndarray], dict[str, int | float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames: list[np.ndarray] = []
    read_frames = 0
    processed_frames = 0
    hand_frames = 0
    pose_frames = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if read_frames % frame_step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            if results.pose_landmarks:
                pose_frames += 1
            if results.left_hand_landmarks or results.right_hand_landmarks:
                hand_frames += 1
            frames.append(mediapipe_landmarks_to_frame(results))
            processed_frames += 1
        read_frames += 1

    cap.release()
    return frames, {
        "fps": fps,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "hand_frames": hand_frames,
        "pose_frames": pose_frames,
    }


def frames_to_tensor(
    frames: list[np.ndarray],
    sequence_length: int,
    expected_features: int,
    normalize: bool,
) -> np.ndarray:
    tensor = sequence_to_tensor(
        frames,
        sequence_length=sequence_length,
        feature_dims=3,
        normalize=normalize,
    )
    if tensor.shape[1] > expected_features:
        tensor = tensor[:, :expected_features]
    elif tensor.shape[1] < expected_features:
        aligned = np.zeros((tensor.shape[0], expected_features), dtype=tensor.dtype)
        aligned[:, : tensor.shape[1]] = tensor
        tensor = aligned
    return tensor


def predict_windows(
    model: torch.nn.Module,
    windows: list[list[np.ndarray]],
    labels: list[str],
    sequence_length: int,
    expected_features: int,
    normalize: bool,
    batch_size: int,
) -> list[np.ndarray]:
    tensors = [
        frames_to_tensor(window, sequence_length, expected_features, normalize)
        for window in windows
    ]
    probs_list: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(tensors), batch_size):
            batch = np.stack(tensors[start : start + batch_size]).astype(np.float32)
            logits = model(torch.tensor(batch, dtype=torch.float32)).numpy()
            batch_probs = np.stack([softmax(row) for row in logits])
            probs_list.extend(batch_probs)
    if probs_list and len(probs_list[0]) != len(labels):
        raise RuntimeError("Model output and labels length do not match.")
    return probs_list


def evaluate_video(
    video_path: Path,
    holistic: Any,
    model: torch.nn.Module,
    labels: list[str],
    sequence_length: int,
    expected_features: int,
    frame_step: int,
    window_stride: int,
    batch_size: int,
    normalize: bool,
) -> dict[str, Any]:
    frames, stats = extract_frames(video_path, holistic, frame_step)
    if not frames:
        raise RuntimeError(f"No usable frames extracted: {video_path}")

    if len(frames) < sequence_length:
        windows = [frames]
    else:
        starts = list(range(0, len(frames) - sequence_length + 1, max(1, window_stride)))
        last_start = len(frames) - sequence_length
        if starts[-1] != last_start:
            starts.append(last_start)
        windows = [frames[idx : idx + sequence_length] for idx in starts]

    probs_by_window = predict_windows(
        model=model,
        windows=windows,
        labels=labels,
        sequence_length=sequence_length,
        expected_features=expected_features,
        normalize=normalize,
        batch_size=batch_size,
    )
    top_indices = [int(np.argmax(probs)) for probs in probs_by_window]
    final_probs = probs_by_window[-1]
    average_probs = np.mean(np.stack(probs_by_window), axis=0)
    majority_idx = Counter(top_indices).most_common(1)[0][0]

    final_idx = int(np.argmax(final_probs))
    average_idx = int(np.argmax(average_probs))
    expected = expected_label_from_name(video_path)

    return {
        "file": video_path.name,
        "expected": expected,
        "final": labels[final_idx],
        "final_confidence": float(final_probs[final_idx]),
        "average": labels[average_idx],
        "average_confidence": float(average_probs[average_idx]),
        "majority": labels[majority_idx],
        "majority_ratio": float(top_indices.count(majority_idx) / len(top_indices)),
        "final_correct": labels[final_idx] == expected,
        "average_correct": labels[average_idx] == expected,
        "majority_correct": labels[majority_idx] == expected,
        "num_windows": len(windows),
        **stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/mediapipe.yaml")
    parser.add_argument("--video-dir", default="data/raw/label8_clip")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/sequence_model.pt")
    parser.add_argument("--output-dir", default="outputs/local_label8_mediapipe_eval")
    parser.add_argument("--pattern", default="*.mp4")
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--window-stride", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--min-detection-confidence", type=float, default=0.3)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.3)
    args = parser.parse_args()

    config = load_config(args.config)
    normalize = bool(config.get("preprocess", {}).get("normalize", True))
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, labels, expected_features, sequence_length = load_sequence_checkpoint(Path(args.checkpoint))
    video_paths = sorted(video_dir.glob(args.pattern))
    if not video_paths:
        raise RuntimeError(f"No videos matched {video_dir / args.pattern}")

    rows: list[dict[str, Any]] = []
    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=int(args.model_complexity),
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=float(args.min_detection_confidence),
        min_tracking_confidence=float(args.min_tracking_confidence),
    ) as holistic:
        for video_path in video_paths:
            print(f"Evaluating {video_path}", flush=True)
            rows.append(
                evaluate_video(
                    video_path=video_path,
                    holistic=holistic,
                    model=model,
                    labels=labels,
                    sequence_length=sequence_length,
                    expected_features=expected_features,
                    frame_step=max(1, int(args.frame_step)),
                    window_stride=max(1, int(args.window_stride)),
                    batch_size=max(1, int(args.batch_size)),
                    normalize=normalize,
                )
            )

    df = pd.DataFrame(rows)
    summary = {
        "video_dir": str(video_dir),
        "checkpoint": str(args.checkpoint),
        "labels": labels,
        "num_videos": int(len(df)),
        "sequence_length": int(sequence_length),
        "feature_count": int(expected_features),
        "final_accuracy": float(df["final_correct"].mean()),
        "average_accuracy": float(df["average_correct"].mean()),
        "majority_accuracy": float(df["majority_correct"].mean()),
        "rows": rows,
    }

    csv_path = output_dir / "local_label8_mediapipe_eval_summary.csv"
    json_path = output_dir / "local_label8_mediapipe_eval.json"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False, indent=2))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
