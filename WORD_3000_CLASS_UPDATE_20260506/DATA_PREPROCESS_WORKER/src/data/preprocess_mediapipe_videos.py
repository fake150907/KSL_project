"""Build a fixed-length training NPZ from original videos with MediaPipe.

Input:
  - CSV manifest with at least: sample_id,label,word_id,video_path
  - Optional manifest columns: split,start,end,duration
Output:
  - NPZ compatible with src.models.train_lstm
Example:
  python -m src.data.preprocess_mediapipe_videos ^
    --manifest data/mediapipe_video_manifest.csv ^
    --output data/processed/sign_word_mediapipe_subset.npz
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2

_MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", ".matplotlib")).resolve()
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import mediapipe as mp
import numpy as np
import pandas as pd

from src.data.keypoint_utils import mediapipe_landmarks_to_frame, sequence_to_tensor
from src.utils.config import load_config
from src.utils.io import write_json
from src.utils.seed import set_seed


REQUIRED_COLUMNS = {"sample_id", "label", "word_id", "video_path"}
EXPECTED_CLASS_COUNT = 3000


def load_word_id_label_map(path: str | Path) -> dict[str, int]:
    label_map_path = Path(path)
    with label_map_path.open("r", encoding="utf-8-sig") as file:
        label_map = json.load(file)
    if not isinstance(label_map, dict):
        raise ValueError(f"Label map must be a JSON object: {label_map_path}")
    parsed = {str(word_id): int(class_id) for word_id, class_id in label_map.items()}
    if len(parsed) != EXPECTED_CLASS_COUNT:
        raise ValueError(f"Label map must contain {EXPECTED_CLASS_COUNT} WORD IDs, got {len(parsed)}")
    if sorted(parsed.values()) != list(range(EXPECTED_CLASS_COUNT)):
        raise ValueError("Label map class ids must be exactly 0..2999")
    return parsed


def _float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    return float(value)


def _segment_bounds(row: pd.Series, fps: float, total_frames: int) -> tuple[int, int]:
    start = _float_or_none(row.get("start"))
    end = _float_or_none(row.get("end"))
    duration = _float_or_none(row.get("duration"))

    if start is None or end is None:
        return 0, total_frames

    if duration and duration > 0:
        start_frame = int((start / duration) * total_frames)
        end_frame = int((end / duration) * total_frames)
    else:
        safe_fps = fps if fps > 0 else 30.0
        start_frame = int(start * safe_fps)
        end_frame = int(end * safe_fps)

    start_frame = max(0, min(total_frames - 1, start_frame))
    end_frame = max(start_frame + 1, min(total_frames, end_frame))
    return start_frame, end_frame


def extract_video_frames(
    video_path: Path,
    holistic: Any,
    row: pd.Series,
    frame_step: int,
    min_detection_confidence: float,
    landmark_layout: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame, end_frame = _segment_bounds(row, fps, total_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: list[np.ndarray] = []
    read_count = 0
    processed_count = 0
    hand_count = 0
    pose_count = 0
    frame_index = start_frame

    while frame_index < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if read_count % frame_step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            has_pose = results.pose_landmarks is not None
            has_hand = bool(results.left_hand_landmarks or results.right_hand_landmarks)
            if has_pose:
                pose_count += 1
            if has_hand:
                hand_count += 1
            frames.append(mediapipe_landmarks_to_frame(results, layout=landmark_layout))
            processed_count += 1
        read_count += 1
        frame_index += 1

    cap.release()
    return frames, {
        "fps": fps,
        "total_frames": total_frames,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "processed_frames": processed_count,
        "hand_frames": hand_count,
        "pose_frames": pose_count,
        "frame_step": frame_step,
        "min_detection_confidence": min_detection_confidence,
        "landmark_layout": landmark_layout,
    }


def validate_manifest(manifest: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    if manifest.empty:
        raise ValueError("Manifest has no rows.")


def preprocess(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    set_seed(int(config["data"]["random_seed"]))

    manifest_path = Path(args.manifest)
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    validate_manifest(manifest)

    label_to_id = load_word_id_label_map(args.label_map_word_id)
    labels = [f"WORD{i:04d}" for i in range(1, EXPECTED_CLASS_COUNT + 1)]
    sequence_length = int(args.sequence_length or config["data"]["sequence_length"])
    feature_dims = int(config["preprocess"]["feature_dims"])
    normalize = bool(config["preprocess"]["normalize"])
    normalize_mode = str(config.get("preprocess", {}).get("normalize_mode", "bbox"))
    landmark_layout = str(args.landmark_layout or config.get("preprocess", {}).get("landmark_layout", "mediapipe_xyz"))

    tensors: list[np.ndarray] = []
    y: list[int] = []
    splits: list[str] = []
    sample_ids: list[str] = []
    rows_report: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []

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
        for row_index, row in manifest.iterrows():
            video_path = Path(str(row["video_path"]))
            if not video_path.is_absolute():
                video_path = (manifest_path.parent / video_path).resolve()
                if not video_path.exists():
                    video_path = Path(str(row["video_path"])).resolve()

            sample_id = str(row["sample_id"])
            label = str(row["label"])
            word_id = str(row["word_id"])
            if word_id not in label_to_id:
                failed_rows.append(
                    {
                        "row_index": int(row_index),
                        "sample_id": sample_id,
                        "label": label,
                        "word_id": word_id,
                        "video_path": str(video_path),
                        "error": f"Unknown word_id: {word_id}",
                    }
                )
                continue
            try:
                frames, stats = extract_video_frames(
                    video_path=video_path,
                    holistic=holistic,
                    row=row,
                    frame_step=max(1, int(args.frame_step)),
                    min_detection_confidence=float(args.min_detection_confidence),
                    landmark_layout=landmark_layout,
                )
                if not frames:
                    raise RuntimeError("No frames were extracted.")
                if stats["hand_frames"] < int(args.min_hand_frames):
                    raise RuntimeError(
                        f"Too few hand frames: {stats['hand_frames']} < {int(args.min_hand_frames)}"
                    )
                tensor = sequence_to_tensor(
                    frames,
                    sequence_length=sequence_length,
                    feature_dims=feature_dims,
                    normalize=normalize,
                    normalize_mode=normalize_mode,
                )
            except Exception as exc:  # keep batch extraction moving
                failed_rows.append(
                    {
                        "row_index": int(row_index),
                        "sample_id": sample_id,
                        "label": label,
                        "video_path": str(video_path),
                        "error": str(exc),
                    }
                )
                continue

            tensors.append(tensor)
            y.append(label_to_id[word_id])
            splits.append(str(row.get("split", "train") or "train"))
            sample_ids.append(sample_id)
            rows_report.append(
                {
                    "sample_id": sample_id,
                    "label": label,
                    "word_id": word_id,
                    "video_path": str(video_path),
                    "split": splits[-1],
                    **stats,
                }
            )

    if not tensors:
        raise RuntimeError("No usable samples were extracted. Check video paths and MediaPipe detection.")

    max_features = max(t.shape[1] for t in tensors)
    aligned = np.zeros((len(tensors), sequence_length, max_features), dtype=np.float32)
    for idx, tensor in enumerate(tensors):
        aligned[idx, :, : tensor.shape[1]] = tensor

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=aligned,
        y=np.asarray(y, dtype=np.int64),
        splits=np.asarray(splits),
        sample_ids=np.asarray(sample_ids),
        labels=np.asarray(labels),
    )

    meta = {
        "source": "mediapipe_video",
        "manifest": str(manifest_path),
        "output": str(out_path),
        "labels": labels,
        "label_to_id": label_to_id,
        "num_manifest_rows": int(len(manifest)),
        "num_samples": int(len(aligned)),
        "num_failed": int(len(failed_rows)),
        "shape": [int(v) for v in aligned.shape],
        "sequence_length": sequence_length,
        "feature_count": int(aligned.shape[2]),
        "feature_layout": (
            "MediaPipe native xyz slots: pose33 + left_hand21 + right_hand21, each [x,y,z]"
            if landmark_layout == "mediapipe_xyz"
            else "MediaPipe fixed slots: pose25 + left_hand21 + right_hand21, each [x,y,confidence]"
        ),
        "landmark_layout": landmark_layout,
        "normalize": normalize,
        "normalize_mode": normalize_mode,
        "rows": rows_report,
        "failed_rows": failed_rows,
    }
    write_json(out_path.with_suffix(".meta.json"), meta)
    if failed_rows:
        pd.DataFrame(failed_rows).to_csv(out_path.with_suffix(".failed.csv"), index=False, encoding="utf-8-sig")

    return {
        "path": str(out_path),
        "shape": tuple(aligned.shape),
        "labels": labels,
        "failed": len(failed_rows),
        "meta": str(out_path.with_suffix(".meta.json")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--manifest", default="data/mediapipe_video_manifest.csv")
    parser.add_argument("--output", default="data/processed/sign_word_mediapipe_subset.npz")
    parser.add_argument(
        "--label-map-word-id",
        default="team_handover_2026-04-29/WORD_3000_CLASS_UPDATE_20260506/label_map_word_id_3000.json",
        help="WORD ID based class map JSON. Example: WORD0001 -> 0.",
    )
    parser.add_argument("--sequence_length", type=int)
    parser.add_argument("--frame_step", type=int, default=1)
    parser.add_argument(
        "--landmark-layout",
        default=None,
        choices=["mediapipe_fixed", "mediapipe_xyz", "openpose_compat"],
    )
    parser.add_argument("--min_hand_frames", type=int, default=1)
    parser.add_argument("--model_complexity", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--min_detection_confidence", type=float, default=0.3)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.3)
    args = parser.parse_args()
    print(preprocess(args))


if __name__ == "__main__":
    main()

