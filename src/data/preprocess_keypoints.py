"""Role: convert selected keypoint JSON samples into fixed-length tensors.

Input:
  - data/sample_subset_manifest.csv
Output:
  - data/processed/sign_word_subset.npz
Example:
  python -m src.data.preprocess_keypoints --subset_only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.keypoint_utils import extract_frames, sequence_to_tensor
from src.utils.config import apply_cli_overrides, load_config
from src.utils.io import read_json, write_json
from src.utils.seed import set_seed


def dummy_sequence(label_index: int, sample_index: int, sequence_length: int, features: int = 126) -> np.ndarray:
    rng = np.random.default_rng(label_index * 1000 + sample_index)
    base = rng.normal(loc=label_index * 0.15, scale=0.04, size=(sequence_length, features))
    phase = np.linspace(0, np.pi * 2, sequence_length)
    base[:, 0] += np.sin(phase + label_index)
    base[:, 1] += np.cos(phase + sample_index)
    return base.astype(np.float32)


def load_tensor(row: pd.Series, label_index: int, config: dict, row_index: int) -> np.ndarray:
    seq_len = int(config["data"]["sequence_length"])
    if bool(row.get("is_dummy", False)) or not str(row.get("keypoint_path", "")):
        return dummy_sequence(label_index, row_index, seq_len)
    keypoint_path = Path(str(row["keypoint_path"]))
    if keypoint_path.is_dir():
        frames = []
        frame_paths = sorted(keypoint_path.glob("*.json"))
        start = row.get("start")
        end = row.get("end")
        duration = row.get("duration")
        if pd.notna(start) and pd.notna(end) and pd.notna(duration) and float(duration) > 0 and frame_paths:
            total = len(frame_paths)
            start_idx = max(0, min(total - 1, int((float(start) / float(duration)) * total)))
            end_idx = max(start_idx + 1, min(total, int((float(end) / float(duration)) * total)))
            frame_paths = frame_paths[start_idx:end_idx]
        if len(frame_paths) > seq_len:
            indices = np.linspace(0, len(frame_paths) - 1, seq_len).round().astype(int)
            frame_paths = [frame_paths[i] for i in indices]
        for frame_path in frame_paths:
            frame_record = read_json(frame_path)
            frames.extend(extract_frames(frame_record))
    else:
        record = read_json(keypoint_path)
        frames = extract_frames(record)
    return sequence_to_tensor(
        frames,
        sequence_length=seq_len,
        feature_dims=int(config["preprocess"]["feature_dims"]),
        normalize=bool(config["preprocess"]["normalize"]),
    )


def preprocess(config: dict) -> dict[str, object]:
    manifest_path = Path(config["paths"]["subset_manifest"])
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    labels = sorted(manifest["label"].unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    set_seed(int(config["data"]["random_seed"]))

    tensors = []
    y = []
    splits = []
    sample_ids = []
    for idx, row in manifest.iterrows():
        label_idx = label_to_id[row["label"]]
        tensors.append(load_tensor(row, label_idx, config, int(idx)))
        y.append(label_idx)
        splits.append(row.get("split", "train"))
        sample_ids.append(str(row["sample_id"]))

    max_features = max(t.shape[1] for t in tensors)
    aligned = np.zeros((len(tensors), int(config["data"]["sequence_length"]), max_features), dtype=np.float32)
    for idx, tensor in enumerate(tensors):
        aligned[idx, :, : tensor.shape[1]] = tensor

    out_path = Path(config["paths"]["processed_npz"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=aligned,
        y=np.asarray(y, dtype=np.int64),
        splits=np.asarray(splits),
        sample_ids=np.asarray(sample_ids),
        labels=np.asarray(labels),
    )
    write_json(
        out_path.with_suffix(".meta.json"),
        {
            "labels": labels,
            "label_to_id": label_to_id,
            "num_samples": int(len(aligned)),
            "sequence_length": int(aligned.shape[1]),
            "feature_count": int(aligned.shape[2]),
        },
    )
    return {
        "path": str(out_path),
        "shape": tuple(aligned.shape),
        "labels": labels,
        "train_count": int(np.sum(np.asarray(splits) == "train")),
        "validation_count": int(np.sum(np.asarray(splits) == "validation")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--subset_only", action="store_true")
    parser.add_argument("--quick_test", action="store_true", default=None)
    parser.add_argument("--sequence_length", type=int)
    args = parser.parse_args()

    config = apply_cli_overrides(load_config(args.config), args)
    result = preprocess(config)
    print(result)


if __name__ == "__main__":
    main()
