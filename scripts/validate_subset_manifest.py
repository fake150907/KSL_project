"""Validate a worker's sample_subset_manifest.csv.

Input:
  - worker result manifest CSV
Output:
  - pass/fail messages for required columns, duplicate sample_id, labels, angles, and paths
Example:
  python scripts/validate_subset_manifest.py --manifest worker_A_result/sample_subset_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_COLUMNS = [
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

ALLOWED_LABELS = {"가다", "감사", "괜찮다", "배고프다", "병원", "아프다", "우유", "자다"}
ALLOWED_ANGLES = {"F", "U", "D", "R", "L"}
ALLOWED_SPLITS = {"train", "validation"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to sample_subset_manifest.csv")
    parser.add_argument(
        "--check_paths",
        action="store_true",
        help="Also check whether each keypoint_path exists on this PC.",
    )
    return parser.parse_args()


def as_float(value: str) -> bool:
    if value == "":
        return True
    try:
        float(value)
        return True
    except ValueError:
        return False


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"[FAIL] Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        rows = list(reader)

    missing = [column for column in REQUIRED_COLUMNS if column not in columns]
    extra = [column for column in columns if column not in REQUIRED_COLUMNS]
    if missing:
        raise SystemExit(f"[FAIL] Missing required columns: {missing}")
    if extra:
        print(f"[WARN] Extra columns will be ignored by the current pipeline: {extra}")
    if not rows:
        raise SystemExit("[FAIL] Manifest has no rows.")

    errors: list[str] = []
    seen: set[str] = set()
    label_counts: dict[str, int] = {}
    angle_counts: dict[str, int] = {}

    for index, row in enumerate(rows, start=2):
        sample_id = row["sample_id"].strip()
        label = row["label"].strip()
        angle = row["angle"].strip()
        split = row["split"].strip()
        keypoint_path = row["keypoint_path"].strip()

        if not sample_id:
            errors.append(f"line {index}: sample_id is empty")
        elif sample_id in seen:
            errors.append(f"line {index}: duplicate sample_id {sample_id}")
        seen.add(sample_id)

        if label not in ALLOWED_LABELS:
            errors.append(f"line {index}: unexpected label {label!r}")
        if angle not in ALLOWED_ANGLES:
            errors.append(f"line {index}: unexpected angle {angle!r}")
        if split not in ALLOWED_SPLITS:
            errors.append(f"line {index}: split must be train or validation, got {split!r}")
        for numeric_column in ("start", "end", "duration"):
            if not as_float(row[numeric_column].strip()):
                errors.append(f"line {index}: {numeric_column} must be numeric or empty")
        if not keypoint_path:
            errors.append(f"line {index}: keypoint_path is empty")
        elif args.check_paths and not Path(keypoint_path).exists():
            errors.append(f"line {index}: keypoint_path does not exist: {keypoint_path}")

        label_counts[label] = label_counts.get(label, 0) + 1
        angle_counts[angle] = angle_counts.get(angle, 0) + 1

    if errors:
        print("[FAIL] Manifest validation failed.")
        for error in errors[:50]:
            print(f"- {error}")
        if len(errors) > 50:
            print(f"- ... and {len(errors) - 50} more errors")
        raise SystemExit(1)

    print("[OK] Manifest validation passed.")
    print(f"rows: {len(rows)}")
    print(f"labels: {label_counts}")
    print(f"angles: {angle_counts}")


if __name__ == "__main__":
    main()
