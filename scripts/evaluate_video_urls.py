"""Evaluate sign-language YouTube/video clips with the current sequence model.

Example:
  python scripts/evaluate_video_urls.py --clip 감사=https://youtu.be/... --clip 괜찮다=https://youtube.com/shorts/...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib"))

import cv2
import mediapipe as mp
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.keypoint_utils import mediapipe_landmarks_to_frame, sequence_to_tensor
from src.models.model_sequence import build_sequence_model
from src.utils.config import load_config


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


def safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣_-]+", "_", text).strip("_") or "clip"


def parse_clip(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--clip must be LABEL=URL")
    label, url = value.split("=", 1)
    label = label.strip()
    url = url.strip()
    if not label or not url:
        raise argparse.ArgumentTypeError("--clip must be LABEL=URL")
    return label, url


def download_video(label: str, url: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob(f"{safe_name(label)}.*"))
    if existing:
        return existing[0]

    try:
        import yt_dlp
    except ImportError as exc:
        raise RuntimeError("yt-dlp is required. Install with: python -m pip install yt-dlp") from exc

    output_template = str(out_dir / f"{safe_name(label)}.%(ext)s")
    ydl_opts: dict[str, Any] = {
        "format": "best[ext=mp4][vcodec!=none]/best[ext=mp4]/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "overwrites": False,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded = Path(ydl.prepare_filename(info))
    if downloaded.exists():
        return downloaded

    candidates = sorted(out_dir.glob(f"{safe_name(label)}.*"))
    if not candidates:
        raise FileNotFoundError(f"Downloaded video for {label} was not found")
    return candidates[0]


def load_sequence_model(config_path: str) -> tuple[torch.nn.Module, list[str], int, int]:
    config = load_config(config_path)
    checkpoint = ROOT / config["paths"]["checkpoints_dir"] / "sequence_model.pt"
    bundle = torch.load(checkpoint, map_location="cpu")
    train_config = bundle.get("config", {}).get("train", {})
    sequence_length = int(bundle.get("config", {}).get("data", {}).get("sequence_length", config["data"]["sequence_length"]))
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
        sequence_length=sequence_length,
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model, [str(x) for x in bundle["labels"]], int(bundle["input_size"]), sequence_length


def align_tensor_features(tensor: np.ndarray, expected: int) -> np.ndarray:
    if expected <= 0 or tensor.shape[1] == expected:
        return tensor
    if tensor.shape[1] > expected:
        return tensor[:, :expected]
    aligned = np.zeros((tensor.shape[0], expected), dtype=tensor.dtype)
    aligned[:, : tensor.shape[1]] = tensor
    return aligned


def predict_top(model: torch.nn.Module, labels: list[str], tensor: np.ndarray, top_k: int) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    with torch.no_grad():
        logits = model(torch.tensor(tensor[None, ...], dtype=torch.float32))[0].numpy()
    probs = softmax(logits)
    top_idx = np.argsort(probs)[::-1][:top_k]
    top = [{"label": labels[i], "confidence": float(probs[i])} for i in top_idx]
    return probs, top


def analyze_video(
    video_path: Path,
    expected_label: str,
    model: torch.nn.Module,
    labels: list[str],
    input_size: int,
    sequence_length: int,
    frame_step: int,
    top_k: int,
    mirror: bool,
) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    windows: list[dict[str, Any]] = []
    probs_list: list[np.ndarray] = []
    valid_frames: list[np.ndarray] = []
    total_frames = 0
    sampled_frames = 0
    hand_frames = 0

    holistic_cls = mp.solutions.holistic.Holistic
    with holistic_cls(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            total_frames += 1
            if total_frames % frame_step != 0:
                continue
            sampled_frames += 1
            if mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            has_hand = bool(results.left_hand_landmarks or results.right_hand_landmarks)
            if not has_hand:
                continue
            frame_points = mediapipe_landmarks_to_frame(results)
            if frame_points.size == 0:
                continue

            hand_frames += 1
            valid_frames.append(frame_points)
            if len(valid_frames) < sequence_length:
                continue

            window = valid_frames[-sequence_length:]
            tensor = sequence_to_tensor(window, sequence_length)
            tensor = align_tensor_features(tensor, input_size)
            probs, top = predict_top(model, labels, tensor, top_k)
            probs_list.append(probs)
            windows.append(
                {
                    "frame": total_frames,
                    "top": top,
                    "predicted": top[0]["label"],
                    "confidence": top[0]["confidence"],
                }
            )

    cap.release()

    if probs_list:
        avg_probs = np.mean(np.stack(probs_list), axis=0)
        avg_idx = np.argsort(avg_probs)[::-1][:top_k]
        average_top = [{"label": labels[i], "confidence": float(avg_probs[i])} for i in avg_idx]
        majority = Counter(str(w["predicted"]) for w in windows).most_common(top_k)
        majority_top = [{"label": label, "count": count, "ratio": count / len(windows)} for label, count in majority]
        final_top = windows[-1]["top"]
    else:
        average_top = []
        majority_top = []
        final_top = []

    return {
        "expected_label": expected_label,
        "video_path": str(video_path),
        "total_frames": total_frames,
        "sampled_frames": sampled_frames,
        "hand_frames": hand_frames,
        "windows": len(windows),
        "final_top": final_top,
        "average_top": average_top,
        "majority_top": majority_top,
        "window_predictions": windows,
    }


def write_summary_csv(results: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "expected_label",
                "windows",
                "hand_frames",
                "final_1",
                "final_1_conf",
                "average_1",
                "average_1_conf",
                "majority_1",
                "majority_1_ratio",
            ],
        )
        writer.writeheader()
        for result in results:
            final_1 = result["final_top"][0] if result["final_top"] else {}
            average_1 = result["average_top"][0] if result["average_top"] else {}
            majority_1 = result["majority_top"][0] if result["majority_top"] else {}
            writer.writerow(
                {
                    "expected_label": result["expected_label"],
                    "windows": result["windows"],
                    "hand_frames": result["hand_frames"],
                    "final_1": final_1.get("label", ""),
                    "final_1_conf": final_1.get("confidence", ""),
                    "average_1": average_1.get("label", ""),
                    "average_1_conf": average_1.get("confidence", ""),
                    "majority_1": majority_1.get("label", ""),
                    "majority_1_ratio": majority_1.get("ratio", ""),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--clip", action="append", type=parse_clip, required=True, help="LABEL=URL")
    parser.add_argument("--output-dir", default="outputs/video_eval")
    parser.add_argument("--frame-step", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mirror", action="store_true", help="Flip frames horizontally before inference")
    args = parser.parse_args()

    out_dir = ROOT / args.output_dir
    video_dir = out_dir / "videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    model, labels, input_size, sequence_length = load_sequence_model(args.config)
    results: list[dict[str, Any]] = []

    for expected_label, url in args.clip:
        video_path = download_video(expected_label, url, video_dir)
        result = analyze_video(
            video_path=video_path,
            expected_label=expected_label,
            model=model,
            labels=labels,
            input_size=input_size,
            sequence_length=sequence_length,
            frame_step=max(1, args.frame_step),
            top_k=max(1, args.top_k),
            mirror=args.mirror,
        )
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k != "window_predictions"}, ensure_ascii=False, indent=2))

    json_path = out_dir / ("youtube_eval_mirror.json" if args.mirror else "youtube_eval.json")
    csv_path = out_dir / ("youtube_eval_mirror_summary.csv" if args.mirror else "youtube_eval_summary.csv")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_summary_csv(results, csv_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
