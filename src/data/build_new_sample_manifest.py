"""Role: build a manifest from data/New_sample sample folders.

Input:
  - data/New_sample/labeling/REAL/WORD/01_real_word_keypoint/<sample_id>/*.json
  - existing AI Hub morpheme JSONs under data/raw/**/*
Output:
  - data/sample_subset_manifest.csv
  - data/selected_labels_small.json
  - data/new_sample_labels.csv
Example:
  python -m src.data.build_new_sample_manifest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.extract_keypoint_subset_from_zip import sample_id_from_morpheme, timing_from_morpheme
from src.data.extract_labels import extract_label, iter_json_files
from src.utils.config import load_config
from src.utils.io import read_json, write_json
from src.utils.sample_id import parse_angle


def build_morpheme_index(config: dict) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in iter_json_files(config["data"]["morpheme_globs"]):
        p = Path(path)
        index[sample_id_from_morpheme(path)] = p
    return index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--sample_root", default="data/New_sample")
    parser.add_argument("--keypoint_rel", default="labeling/REAL/WORD/01_real_word_keypoint")
    parser.add_argument("--raw_rel", default="raw/REAL/WORD/01")
    args = parser.parse_args()

    config = load_config(args.config)
    sample_root = Path(args.sample_root)
    keypoint_root = sample_root / args.keypoint_rel
    raw_root = sample_root / args.raw_rel
    morpheme_index = build_morpheme_index(config)

    rows = []
    for keypoint_dir in sorted(p for p in keypoint_root.iterdir() if p.is_dir()):
        sample_id = keypoint_dir.name
        morpheme_path = morpheme_index.get(sample_id)
        if morpheme_path is None:
            continue
        record = read_json(morpheme_path)
        label = extract_label(record, str(morpheme_path))
        start, end, duration = timing_from_morpheme(record)
        angle = parse_angle(sample_id)
        rows.append(
            {
                "sample_id": sample_id,
                "label": label,
                "angle": angle,
                "morpheme_path": str(morpheme_path),
                "start": start,
                "end": end,
                "duration": duration,
                "split": "validation" if angle == "U" else "train",
                "keypoint_path": str(keypoint_dir),
                "video_path": str(raw_root / f"{sample_id}.mp4"),
                "is_dummy": False,
            }
        )

    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError(f"No New_sample rows matched under {keypoint_root}")

    labels = sorted(manifest["label"].unique().tolist())
    manifest.to_csv(config["paths"]["subset_manifest"], index=False, encoding="utf-8-sig")
    manifest.groupby("label").size().reset_index(name="sample_count").to_csv(
        "data/new_sample_labels.csv", index=False, encoding="utf-8-sig"
    )
    write_json(
        config["paths"]["selected_labels_small"],
        {
            "labels": labels,
            "mode": "new_sample_real_keypoint",
            "sample_count": int(len(manifest)),
            "note": "Built from data/New_sample keypoint folders and existing AI Hub morpheme labels.",
        },
    )
    print(
        {
            "rows": int(len(manifest)),
            "classes": int(len(labels)),
            "manifest": config["paths"]["subset_manifest"],
            "counts": manifest.groupby("label").size().to_dict(),
        }
    )


if __name__ == "__main__":
    main()
