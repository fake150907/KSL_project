"""Role: extract only selected-label keypoint sample folders from AI Hub zip.

Input:
  - data/selected_labels_small.json
  - AI Hub WORD morpheme JSONs
  - 01_real_word_keypoint.zip
Output:
  - data/raw/selected_keypoints/<sample_id>/*.json
  - data/sample_subset_manifest.csv
Example:
  python -m src.data.extract_keypoint_subset_from_zip --max_samples_per_class 10
  python -m src.data.extract_keypoint_subset_from_zip --zip_path data/raw/.../02_real_word_keypoint.zip --output_dir data/raw/worker_A_selected_keypoints --manifest_output data/worker_A_sample_subset_manifest.csv
"""

from __future__ import annotations

import argparse
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.data.extract_labels import extract_label, iter_json_files
from src.utils.config import apply_cli_overrides, load_config
from src.utils.io import read_json
from src.utils.sample_id import parse_angle, word_angle_key


def sample_id_from_morpheme(path: str) -> str:
    stem = Path(path).stem
    return stem.removesuffix("_morpheme")


def timing_from_morpheme(record: dict) -> tuple[float | None, float | None, float | None]:
    duration = None
    meta = record.get("metaData")
    if isinstance(meta, dict):
        try:
            duration = float(meta.get("duration")) if meta.get("duration") is not None else None
        except Exception:
            duration = None
    starts: list[float] = []
    ends: list[float] = []
    data = record.get("data")
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                if item.get("start") is not None:
                    starts.append(float(item["start"]))
                if item.get("end") is not None:
                    ends.append(float(item["end"]))
            except Exception:
                continue
    return (min(starts) if starts else None, max(ends) if ends else None, duration)


def collect_targets(config: dict, max_samples_per_class: int) -> pd.DataFrame:
    selected = read_json(config["paths"]["selected_labels_small"])["labels"]
    selected_set = set(selected)
    counts: dict[str, int] = defaultdict(int)
    rows = []
    for path in iter_json_files(config["data"]["morpheme_globs"]):
        try:
            record = read_json(path)
        except Exception:
            continue
        label = extract_label(record if isinstance(record, dict) else {}, path)
        if label not in selected_set or counts[label] >= max_samples_per_class:
            continue
        start, end, duration = timing_from_morpheme(record if isinstance(record, dict) else {})
        counts[label] += 1
        sample_id = sample_id_from_morpheme(path)
        angle = parse_angle(sample_id)
        rows.append(
            {
                "sample_id": sample_id,
                "match_key": word_angle_key(sample_id),
                "label": label,
                "angle": angle,
                "morpheme_path": path,
                "start": start,
                "end": end,
                "duration": duration,
                "split": "validation" if angle == "U" or counts[label] % 5 == 0 else "train",
            }
        )
    return pd.DataFrame(rows)


def _find_sample_id_in_zip_path(filename: str, targets: set[str]) -> str | None:
    parts = Path(filename).parts
    if not parts or not filename.endswith(".json"):
        return None
    for part in reversed(parts[:-1]):
        if part in targets:
            return part
    return None


def _sample_id_from_zip_path(filename: str) -> str | None:
    parts = Path(filename).parts
    if not parts or not filename.endswith(".json"):
        return None
    for part in reversed(parts[:-1]):
        if word_angle_key(part):
            return part
    return None


def _find_matching_sample_id(filename: str, target_sample_ids: set[str], target_match_keys: set[str], match_mode: str) -> str | None:
    sample_id = _sample_id_from_zip_path(filename)
    if sample_id is None:
        return None
    if match_mode == "sample_id":
        return sample_id if sample_id in target_sample_ids else None
    key = word_angle_key(sample_id)
    return sample_id if key in target_match_keys else None


def extract_sample_dirs(
    zip_path: Path,
    targets: set[str],
    output_dir: Path,
    *,
    target_match_keys: set[str] | None = None,
    match_mode: str = "sample_id",
) -> set[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: set[str] = set()
    target_match_keys = target_match_keys or set()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            sample_id = _find_matching_sample_id(info.filename, targets, target_match_keys, match_mode)
            if sample_id is None:
                continue
            rel_name = Path(info.filename).name
            target = output_dir / sample_id / rel_name
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                dst.write(src.read())
            extracted.add(sample_id)
    return extracted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--zip_path",
        default="data/raw/aihub_downloads/word_keypoint_01/004.수어영상/1.Training/라벨링데이터/REAL/WORD/01_real_word_keypoint.zip",
    )
    parser.add_argument("--output_dir", default="data/raw/selected_keypoints")
    parser.add_argument("--manifest_output", default=None)
    parser.add_argument("--max_samples_per_class", type=int)
    parser.add_argument(
        "--match_mode",
        choices=["sample_id", "word_angle"],
        default="sample_id",
        help="sample_id requires exact REALxx sample id match; word_angle matches by WORDxxxx + angle across REAL splits.",
    )
    parser.add_argument(
        "--selected_labels_path",
        help="Override config paths.selected_labels_small, e.g. data/selected_labels_small_label239.json",
    )
    args = parser.parse_args()

    config = apply_cli_overrides(load_config(args.config), args)
    if args.selected_labels_path:
        config["paths"]["selected_labels_small"] = args.selected_labels_path
    max_per_class = int(config["data"]["max_samples_per_class"])
    targets = collect_targets(config, max_samples_per_class=max_per_class)
    if targets.empty:
        raise RuntimeError("No target samples found from morpheme labels.")
    extracted = extract_sample_dirs(
        Path(args.zip_path),
        set(targets["sample_id"]),
        Path(args.output_dir),
        target_match_keys=set(targets["match_key"]),
        match_mode=args.match_mode,
    )
    if args.match_mode == "sample_id":
        manifest = targets[targets["sample_id"].isin(extracted)].copy()
        manifest["keypoint_sample_id"] = manifest["sample_id"]
    else:
        extracted_rows = pd.DataFrame(
            [{"keypoint_sample_id": sample_id, "match_key": word_angle_key(sample_id)} for sample_id in sorted(extracted)]
        )
        manifest = extracted_rows.merge(targets, on="match_key", how="inner", suffixes=("", "_morpheme"))
        manifest["sample_id"] = manifest["keypoint_sample_id"]
        # REALxx is ignored only for matching. The WORD+angle pair represents
        # the same signed word segment, so keep morpheme timing for cropping.
    manifest["keypoint_path"] = manifest["keypoint_sample_id"].map(lambda x: str(Path(args.output_dir) / x))
    manifest["is_dummy"] = False
    output_columns = [
        "sample_id",
        "label",
        "angle",
        "morpheme_path",
        "start",
        "end",
        "duration",
        "split",
        "keypoint_path",
        "is_dummy",
    ]
    manifest = manifest[output_columns].drop_duplicates(subset=["sample_id"]).sort_values(["label", "sample_id"])
    manifest_output = args.manifest_output or config["paths"]["subset_manifest"]
    Path(manifest_output).parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_output, index=False, encoding="utf-8-sig")
    print(
        {
            "zip_path": str(Path(args.zip_path)),
            "match_mode": args.match_mode,
            "target_samples": int(len(targets)),
            "extracted_samples": int(len(extracted)),
            "manifest": manifest_output,
            "counts": manifest.groupby("label").size().to_dict() if not manifest.empty else {},
        }
    )


if __name__ == "__main__":
    main()
