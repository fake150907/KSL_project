from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import mediapipe as mp
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from backend import app as backend_app  # noqa: E402


SAMPLE_FPS = 12.0
MIN_SEGMENT_FRAMES = 8
MODEL_TYPE = "cnn_gru"


@dataclass(frozen=True)
class Segment:
    name: str
    expected: str
    start: float
    end: float


@dataclass(frozen=True)
class Scenario:
    name: str
    video: Path
    duration: float
    segments: tuple[Segment, ...]


SCENARIOS = (
    Scenario(
        name="merge1",
        video=ROOT / "web" / "public" / "demo-videos" / "merge1.mp4",
        duration=11.0,
        segments=(
            Segment("상처", "상처", 0.0, 4.4),
            Segment("붕대", "붕대", 4.4, 8.9),
            Segment("원하다", "원하다", 8.9, 11.0),
        ),
    ),
    Scenario(
        name="merge2",
        video=ROOT / "web" / "public" / "demo-videos" / "merge2.mp4",
        duration=10.457,
        segments=(
            Segment("다리", "다리", 0.0, 3.47),
            Segment("골절", "골절", 3.47, 7.97),
            Segment("아프다", "아프다", 7.97, 10.457),
        ),
    ),
    Scenario(
        name="merge3",
        video=ROOT / "web" / "public" / "demo-videos" / "merge3.mp4",
        duration=11.265,
        segments=(
            Segment("소화불량", "소화불량", 0.0, 4.5),
            Segment("어떻게", "어떻게", 4.5, 7.233),
            Segment("치료", "치료", 7.233, 11.265),
        ),
    ),
)


def crop_zoom(frame: np.ndarray, zoom: float) -> np.ndarray:
    if zoom <= 1.0:
        return frame
    height, width = frame.shape[:2]
    crop_w = max(2, int(round(width / zoom)))
    crop_h = max(2, int(round(height / zoom)))
    x0 = max(0, (width - crop_w) // 2)
    y0 = max(0, (height - crop_h) // 2)
    cropped = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def make_transform(flip: bool = False, zoom: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    def transform(frame: np.ndarray) -> np.ndarray:
        out = crop_zoom(frame, zoom)
        if flip:
            out = cv2.flip(out, 1)
        return out

    return transform


SPATIAL_VARIANTS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "orig": make_transform(),
    "hflip": make_transform(flip=True),
    "zoom1.10": make_transform(zoom=1.10),
    "zoom1.20": make_transform(zoom=1.20),
    "zoom1.10_hflip": make_transform(flip=True, zoom=1.10),
    "zoom1.20_hflip": make_transform(flip=True, zoom=1.20),
}


SEGMENT_VARIANTS: dict[str, tuple[float, float]] = {
    "base": (0.0, 0.0),
    "trim_head_0.2": (0.2, 0.0),
    "trim_tail_0.2": (0.0, -0.2),
    "trim_both_0.2": (0.2, -0.2),
    "trim_head_0.4": (0.4, 0.0),
    "trim_tail_0.4": (0.0, -0.4),
    "trim_both_0.4": (0.4, -0.4),
    "extend_head_0.2": (-0.2, 0.0),
    "extend_tail_0.2": (0.0, 0.2),
    "extend_both_0.2": (-0.2, 0.2),
    "shift_left_0.2": (-0.2, -0.2),
    "shift_right_0.2": (0.2, 0.2),
}


def duration_of(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return float(count / fps) if fps else 0.0


def preprocess_for_mediapipe(frame_bgr: np.ndarray) -> np.ndarray:
    image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    scale = min(320 / width, 180 / height, 1.0)
    if scale < 1.0:
        image_rgb = cv2.resize(
            image_rgb,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return image_rgb


def extract_timeline(
    scenario: Scenario,
    transform_name: str,
    transform: Callable[[np.ndarray], np.ndarray],
    holistic: object,
) -> list[tuple[float, np.ndarray | None]]:
    cap = cv2.VideoCapture(str(scenario.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {scenario.video}")

    actual_duration = duration_of(scenario.video) or scenario.duration
    frame_step = 1.0 / SAMPLE_FPS
    times = np.arange(0.0, actual_duration + 1e-6, frame_step)
    timeline: list[tuple[float, np.ndarray | None]] = []
    hand_count = 0

    for current_time in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(current_time * 1000))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = transform(frame)
        image_rgb = preprocess_for_mediapipe(frame)
        results = holistic.process(image_rgb)
        has_hand = bool(results.left_hand_landmarks or results.right_hand_landmarks)
        if has_hand and backend_app.mediapipe_landmarks_to_frame is not None:
            frame_points = backend_app.mediapipe_landmarks_to_frame(results)
            timeline.append((float(current_time), frame_points))
            hand_count += 1
        else:
            timeline.append((float(current_time), None))

    cap.release()
    print(
        f"[timeline] {scenario.name:6s} {transform_name:14s} "
        f"frames={len(timeline):3d} hand_frames={hand_count:3d}",
        flush=True,
    )
    return timeline


def select_segment_frames(
    timeline: list[tuple[float, np.ndarray | None]],
    start: float,
    end: float,
) -> list[np.ndarray]:
    return [frame for current_time, frame in timeline if start <= current_time < end and frame is not None]


def predict_frames(
    frames: list[np.ndarray],
    target_label: str,
    model: torch.nn.Module,
    labels: list[str],
    temperature: float,
) -> dict[str, object]:
    if len(frames) < MIN_SEGMENT_FRAMES:
        return {
            "pred_label": None,
            "pred_conf": 0.0,
            "target_conf": 0.0,
            "target_rank": None,
            "top5": [],
        }

    model_frames = backend_app.smooth_segment_frames(frames)
    tensor = backend_app.frames_to_model_tensor(MODEL_TYPE, model_frames)
    with torch.no_grad():
        logits = model(torch.tensor(tensor[None, ...], dtype=torch.float32))[0].numpy()
    probs = backend_app.softmax(logits / max(float(temperature), 1e-6))
    order = np.argsort(probs)[::-1]
    top_idx = order[:5]
    pred_idx = int(order[0])
    target_idx = labels.index(target_label) if target_label in labels else None
    target_rank = None
    target_conf = 0.0
    if target_idx is not None:
        target_conf = float(probs[target_idx])
        matches = np.where(order == target_idx)[0]
        if matches.size:
            target_rank = int(matches[0]) + 1

    return {
        "pred_label": labels[pred_idx],
        "pred_conf": float(probs[pred_idx]),
        "target_conf": target_conf,
        "target_rank": target_rank,
        "top5": [
            {"label": labels[int(i)], "confidence": float(probs[int(i)])}
            for i in top_idx
        ],
    }


def clamp_segment(start: float, end: float, scenario_duration: float) -> tuple[float, float] | None:
    start = max(0.0, start)
    end = min(float(scenario_duration), end)
    if end - start < 0.4:
        return None
    return start, end


def main() -> int:
    started = time.time()
    output_dir = ROOT / "outputs" / "merge123_variant_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    backend_app.load_models()
    model, labels = backend_app.get_sequence_model_bundle(MODEL_TYPE)
    if model is None or not labels:
        raise RuntimeError("Sequence model is not loaded")

    temperature = float(backend_app.config.get("realtime", {}).get("temperature", 0.7))

    holistic = mp.solutions.holistic.Holistic(
        model_complexity=0,
        smooth_landmarks=False,
        min_detection_confidence=0.2,
        min_tracking_confidence=0.2,
    )

    rows: list[dict[str, object]] = []
    timelines: dict[tuple[str, str], list[tuple[float, np.ndarray | None]]] = {}

    for scenario in SCENARIOS:
        actual_duration = duration_of(scenario.video) or scenario.duration
        for spatial_name, transform in SPATIAL_VARIANTS.items():
            timeline = extract_timeline(scenario, spatial_name, transform, holistic)
            timelines[(scenario.name, spatial_name)] = timeline
            for segment_index, segment in enumerate(scenario.segments, start=1):
                for segment_variant, (start_delta, end_delta) in SEGMENT_VARIANTS.items():
                    adjusted = clamp_segment(
                        segment.start + start_delta,
                        segment.end + end_delta,
                        actual_duration,
                    )
                    if adjusted is None:
                        continue
                    start_sec, end_sec = adjusted
                    frames = select_segment_frames(timeline, start_sec, end_sec)
                    pred = predict_frames(frames, segment.expected, model, labels, temperature)
                    top5 = pred["top5"]
                    rows.append(
                        {
                            "scenario": scenario.name,
                            "segment_index": segment_index,
                            "segment_name": segment.name,
                            "expected": segment.expected,
                            "spatial_variant": spatial_name,
                            "segment_variant": segment_variant,
                            "start_sec": round(start_sec, 3),
                            "end_sec": round(end_sec, 3),
                            "duration_sec": round(end_sec - start_sec, 3),
                            "hand_frames": len(frames),
                            "pred_label": pred["pred_label"],
                            "pred_conf": pred["pred_conf"],
                            "target_conf": pred["target_conf"],
                            "target_rank": pred["target_rank"],
                            "top5": json.dumps(top5, ensure_ascii=False),
                            "top1_hit": pred["pred_label"] == segment.expected,
                        }
                    )

    holistic.close()

    csv_path = output_dir / "merge123_segment_spatial_results.csv"
    json_path = output_dir / "merge123_segment_spatial_results.json"
    summary_path = output_dir / "merge123_segment_spatial_summary.md"

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["scenario"]), int(row["segment_index"]))
        groups.setdefault(key, []).append(row)

    lines = [
        "# merge1~3 segment/spatial offline experiment",
        "",
        f"- sample_fps: {SAMPLE_FPS}",
        f"- segment variants: {', '.join(SEGMENT_VARIANTS)}",
        f"- spatial variants: {', '.join(SPATIAL_VARIANTS)}",
        f"- elapsed_sec: {time.time() - started:.1f}",
        "",
    ]

    for scenario in SCENARIOS:
        lines.append(f"## {scenario.name}")
        for idx, segment in enumerate(scenario.segments, start=1):
            candidates = groups[(scenario.name, idx)]
            best_target = sorted(candidates, key=lambda r: float(r["target_conf"]), reverse=True)[:5]
            best_pred = sorted(candidates, key=lambda r: float(r["pred_conf"]), reverse=True)[:3]
            baseline = [
                r
                for r in candidates
                if r["spatial_variant"] == "orig" and r["segment_variant"] == "base"
            ][0]
            lines.append("")
            lines.append(f"### {idx}. {segment.name} expected={segment.expected}")
            lines.append(
                "- baseline: "
                f"pred={baseline['pred_label']} "
                f"pred_conf={float(baseline['pred_conf']):.3f} "
                f"target_conf={float(baseline['target_conf']):.3f} "
                f"target_rank={baseline['target_rank']} "
                f"hand_frames={baseline['hand_frames']}"
            )
            lines.append("- best by expected target confidence:")
            for row in best_target:
                lines.append(
                    f"  - {row['spatial_variant']} / {row['segment_variant']} "
                    f"[{row['start_sec']}, {row['end_sec']}] "
                    f"pred={row['pred_label']}({float(row['pred_conf']):.3f}) "
                    f"target_conf={float(row['target_conf']):.3f} "
                    f"rank={row['target_rank']} hand={row['hand_frames']}"
                )
            lines.append("- strongest top1 predictions:")
            for row in best_pred:
                lines.append(
                    f"  - {row['spatial_variant']} / {row['segment_variant']} "
                    f"pred={row['pred_label']}({float(row['pred_conf']):.3f}) "
                    f"target_conf={float(row['target_conf']):.3f}"
                )
        lines.append("")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
