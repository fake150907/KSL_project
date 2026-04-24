"""Role: extract expression labels from AI Hub WORD morpheme JSON files.

Input:
  - data/raw/**/real_word_morpheme/**/*.json or --morpheme_glob
Output:
  - data/label_candidates.csv
  - data/selected_labels_small.json
Example:
  python -m src.data.extract_labels --quick_test --max_classes 8
"""

from __future__ import annotations

import argparse
import glob
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import apply_cli_overrides, load_config
from src.utils.io import read_json, write_json


LABEL_KEYS = (
    "label",
    "word",
    "name",
    "gloss",
    "text",
    "korean",
    "expression",
    "morpheme",
)
ID_KEYS = ("id", "file_id", "video_id", "sentence_id", "word_id", "clip_id", "metadata_id")


def _flatten_strings(obj: Any, key_hint: str = "") -> list[str]:
    values: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            lowered = str(key).lower()
            if lowered in LABEL_KEYS and isinstance(value, str) and value.strip():
                values.append(value.strip())
            elif "morpheme" in lowered and isinstance(value, str) and value.strip():
                values.append(value.strip())
            else:
                values.extend(_flatten_strings(value, lowered))
    elif isinstance(obj, list):
        for item in obj:
            values.extend(_flatten_strings(item, key_hint))
    return values


def extract_label(record: dict[str, Any], fallback: str) -> str:
    data = record.get("data")
    if isinstance(data, list):
        names: list[str] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            attrs = item.get("attributes")
            if isinstance(attrs, list):
                for attr in attrs:
                    if isinstance(attr, dict) and isinstance(attr.get("name"), str) and attr["name"].strip():
                        names.append(attr["name"].strip())
        if names:
            return " ".join(names)

    for key in LABEL_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    strings = _flatten_strings(record)
    if strings:
        return strings[0]
    return Path(fallback).stem


def extract_sample_id(record: dict[str, Any], path: str) -> str:
    for key in ID_KEYS:
        value = record.get(key)
        if isinstance(value, (str, int)):
            return str(value)
    return Path(path).stem


def iter_json_files(patterns: list[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern, recursive=True))
    return sorted(set(paths))


def discover_labels(morpheme_paths: list[str]) -> pd.DataFrame:
    counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    for path in morpheme_paths:
        try:
            record = read_json(path)
            label = extract_label(record, path)
            sample_id = extract_sample_id(record if isinstance(record, dict) else {}, path)
        except Exception:
            label = Path(path).stem
            sample_id = Path(path).stem
        counts[label] += 1
        if len(examples[label]) < 3:
            examples[label].append(sample_id)
    rows = [
        {"label": label, "sample_count": count, "example_ids": "|".join(examples[label])}
        for label, count in counts.most_common()
    ]
    return pd.DataFrame(rows)


def choose_small_labels(df: pd.DataFrame, max_classes: int, min_samples: int) -> list[str]:
    if df.empty:
        return [f"dummy_label_{i + 1}" for i in range(max_classes)]
    filtered = df[df["sample_count"] >= min_samples]
    if filtered.empty:
        filtered = df
    return filtered.head(max_classes)["label"].tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--morpheme_glob", action="append")
    parser.add_argument("--quick_test", action="store_true", default=None)
    parser.add_argument("--max_classes", type=int)
    parser.add_argument("--max_samples_per_class", type=int)
    args = parser.parse_args()

    config = apply_cli_overrides(load_config(args.config), args)
    patterns = args.morpheme_glob or config["data"]["morpheme_globs"]
    paths = iter_json_files(patterns)
    df = discover_labels(paths)

    out_csv = Path(config["paths"]["label_candidates"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    labels = choose_small_labels(
        df,
        max_classes=int(config["data"]["max_classes"]),
        min_samples=int(config["data"]["min_samples_per_class"]),
    )
    write_json(
        config["paths"]["selected_labels_small"],
        {
            "labels": labels,
            "mode": "quick_test",
            "source_file_count": len(paths),
            "note": "Edit this file to pin demo labels before building the subset.",
        },
    )
    print(f"morpheme_files={len(paths)}")
    print(f"label_candidates={len(df)} -> {out_csv}")
    print(f"selected_small={labels}")


if __name__ == "__main__":
    main()
