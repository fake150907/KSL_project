"""Validate a shard NPZ for the 3000 WORD-ID training setup.

Required shard NPZ keys:
  X, y, splits, sample_ids, labels

Required X shape:
  (N, 32, 225)

V2 label rule:
  y must be generated from WORD ID, not from Korean label text.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_KEYS = ["X", "y", "splits", "sample_ids", "labels"]
EXPECTED_SEQUENCE_LENGTH = 32
EXPECTED_FEATURE_COUNT = 225
EXPECTED_CLASS_COUNT = 3000
VALID_SPLITS = {"train", "validation", "val", "test"}
WORD_ID_RE = re.compile(r"(WORD\d{4})")


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def load_label_map(path: Path) -> dict[str, int]:
    if not path.exists():
        fail(f"label map not found: {path}")
    with path.open("r", encoding="utf-8-sig") as file:
        data: Any = json.load(file)
    if not isinstance(data, dict):
        fail("label map must be a JSON object")
    label_map: dict[str, int] = {}
    for key, value in data.items():
        word_id = str(key).strip()
        if not WORD_ID_RE.fullmatch(word_id):
            fail(f"invalid WORD ID in label map: {word_id}")
        try:
            label_map[word_id] = int(value)
        except Exception as exc:
            fail(f"invalid class id for {word_id}: {value} ({exc})")
    return label_map


def scalar_list(array: np.ndarray) -> list[str]:
    return [str(value) for value in array.tolist()]


def word_id_from_sample_id(sample_id: str) -> str:
    match = WORD_ID_RE.search(sample_id)
    if not match:
        fail(f"sample_id does not contain WORDxxxx: {sample_id}")
    return match.group(1)


def validate_npz(npz_path: Path, label_map_path: Path) -> None:
    if not npz_path.exists():
        fail(f"NPZ not found: {npz_path}")

    label_map = load_label_map(label_map_path)
    if len(label_map) != EXPECTED_CLASS_COUNT:
        fail(f"label map must contain {EXPECTED_CLASS_COUNT} WORD IDs, got {len(label_map)}")
    if sorted(label_map.values()) != list(range(EXPECTED_CLASS_COUNT)):
        fail("label map class ids must be exactly 0..2999")

    data = np.load(npz_path, allow_pickle=True)
    files = list(data.files)
    print("NPZ internal keys:", files)

    missing = [key for key in REQUIRED_KEYS if key not in files]
    if missing:
        fail(f"missing required keys: {missing}")

    x = data["X"]
    y = data["y"]
    splits = data["splits"]
    sample_ids = data["sample_ids"]
    labels = scalar_list(data["labels"])

    print("X shape:", x.shape, x.dtype)
    print("y shape:", y.shape, y.dtype)
    print("splits shape:", splits.shape, splits.dtype)
    print("sample_ids shape:", sample_ids.shape, sample_ids.dtype)
    print("labels_count:", len(labels))
    print("y_min:", int(np.min(y)) if len(y) else None)
    print("y_max:", int(np.max(y)) if len(y) else None)

    if x.ndim != 3:
        fail(f"X must be 3D, expected (N, 32, 225), got {x.shape}")
    if x.shape[1] != EXPECTED_SEQUENCE_LENGTH:
        fail(f"sequence length must be {EXPECTED_SEQUENCE_LENGTH}, got {x.shape[1]}")
    if x.shape[2] != EXPECTED_FEATURE_COUNT:
        fail(f"feature count must be {EXPECTED_FEATURE_COUNT}, got {x.shape[2]}")

    sample_count = x.shape[0]
    if y.shape != (sample_count,):
        fail(f"y shape must be {(sample_count,)}, got {y.shape}")
    if splits.shape != (sample_count,):
        fail(f"splits shape must be {(sample_count,)}, got {splits.shape}")
    if sample_ids.shape != (sample_count,):
        fail(f"sample_ids shape must be {(sample_count,)}, got {sample_ids.shape}")

    bad_splits = sorted(set(str(value) for value in splits.tolist()) - VALID_SPLITS)
    if bad_splits:
        fail(f"invalid splits values: {bad_splits}. Allowed values: {sorted(VALID_SPLITS)}")

    if len(labels) != EXPECTED_CLASS_COUNT:
        fail(f"labels must contain {EXPECTED_CLASS_COUNT} WORD IDs, got {len(labels)}")
    if labels[0] != "WORD0001" or labels[-1] != "WORD3000":
        fail("labels must be WORD0001..WORD3000 order")

    if not np.issubdtype(y.dtype, np.integer):
        fail(f"y must be integer dtype, got {y.dtype}")
    if len(y) and (int(np.min(y)) < 0 or int(np.max(y)) >= EXPECTED_CLASS_COUNT):
        fail("y values must be in range 0..2999")

    mismatches: list[str] = []
    for sample_id, class_id in zip(sample_ids.tolist(), y.tolist()):
        sample_id_text = str(sample_id)
        word_id = word_id_from_sample_id(sample_id_text)
        expected = label_map[word_id]
        actual = int(class_id)
        if actual != expected:
            mismatches.append(f"{sample_id_text}: y={actual}, expected={expected}")
            if len(mismatches) >= 10:
                break
    if mismatches:
        fail("sample_id WORD ID and y do not match label map. Examples: " + "; ".join(mismatches))

    print("PASS: NPZ is valid for 3000 WORD-ID submission.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_path", help="Path to shard NPZ")
    parser.add_argument(
        "--label-map",
        default=str(Path(__file__).resolve().parents[1] / "label_map_word_id_3000.json"),
        help="Path to label_map_word_id_3000.json",
    )
    args = parser.parse_args()
    validate_npz(Path(args.npz_path), Path(args.label_map))


if __name__ == "__main__":
    main()

