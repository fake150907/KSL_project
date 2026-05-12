"""Role: create a small manifest for fast end-to-end validation.

Input:
  - data/selected_labels_small.json
  - morpheme/keypoint JSON files under data/raw
Output:
  - data/sample_subset_manifest.csv
Example:
  python -m src.data.build_small_subset --max_classes 8 --max_samples_per_class 30
  python -m src.data.build_small_subset --make_dummy
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.data.extract_labels import extract_label, extract_sample_id, iter_json_files
from src.utils.config import apply_cli_overrides, load_config
from src.utils.io import read_json, write_json
from src.utils.sample_id import parse_angle


def build_keypoint_index(paths: list[str]) -> dict[str, str]:
    index: dict[str, str] = {}
    for path in paths:
        stem = Path(path).stem
        index[stem] = path
    return index


def make_dummy_manifest(labels: list[str], max_samples_per_class: int) -> pd.DataFrame:
    rows = []
    for label in labels:
        for idx in range(max_samples_per_class):
            rows.append(
                {
                    "sample_id": f"{label}_{idx:04d}",
                    "label": label,
                    "angle": "",
                    "morpheme_path": "",
                    "keypoint_path": "",
                    "split": "validation" if idx % 5 == 0 else "train",
                    "is_dummy": True,
                }
            )
    return pd.DataFrame(rows)


def build_manifest(config: dict, make_dummy: bool = False) -> pd.DataFrame:
    selected = read_json(config["paths"]["selected_labels_small"])
    labels = selected["labels"][: int(config["data"]["max_classes"])]
    max_per_class = int(config["data"]["max_samples_per_class"])
    if make_dummy:
        return make_dummy_manifest(labels, max_per_class)

    morpheme_paths = iter_json_files(config["data"]["morpheme_globs"])
    keypoint_index = build_keypoint_index(iter_json_files(config["data"]["keypoint_globs"]))
    counts: dict[str, int] = defaultdict(int)
    rows = []
    for path in morpheme_paths:
        record = read_json(path)
        label = extract_label(record, path)
        if label not in labels or counts[label] >= max_per_class:
            continue
        sample_id = extract_sample_id(record if isinstance(record, dict) else {}, path)
        angle = parse_angle(sample_id)
        keypoint_path = keypoint_index.get(sample_id) or keypoint_index.get(Path(path).stem, "")
        if not keypoint_path:
            continue
        counts[label] += 1
        rows.append(
            {
                "sample_id": sample_id,
                "label": label,
                "angle": angle,
                "morpheme_path": path,
                "keypoint_path": keypoint_path,
                "split": "validation" if counts[label] % 5 == 0 else "train",
                "is_dummy": False,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--max_classes", type=int)
    parser.add_argument("--max_samples_per_class", type=int)
    parser.add_argument("--quick_test", action="store_true", default=None)
    parser.add_argument("--make_dummy", action="store_true")
    args = parser.parse_args()

    config = apply_cli_overrides(load_config(args.config), args)
    selected_path = Path(config["paths"]["selected_labels_small"])
    if not selected_path.exists():
        labels = [f"dummy_label_{i + 1}" for i in range(int(config["data"]["max_classes"]))]
        write_json(selected_path, {"labels": labels, "mode": "quick_test"})
    df = build_manifest(config, make_dummy=args.make_dummy)
    out = Path(config["paths"]["subset_manifest"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"manifest_rows={len(df)} -> {out}")
    print(df.groupby("label").size().to_string() if not df.empty else "empty manifest")


if __name__ == "__main__":
    main()
