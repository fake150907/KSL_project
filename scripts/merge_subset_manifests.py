"""Merge worker subset manifests into one unified CSV.

Example:
  python scripts/merge_subset_manifests.py --inputs data/worker_A_sample_subset_manifest.csv data/worker_B_sample_subset_manifest.csv --output data/sample_subset_manifest_merged.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_subset_manifest import REQUIRED_COLUMNS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", default="data/sample_subset_manifest_merged.csv")
    args = parser.parse_args()

    frames: list[pd.DataFrame] = []
    for input_path in args.inputs:
        path = Path(input_path)
        if not path.exists():
            raise SystemExit(f"Missing manifest: {path}")
        df = pd.read_csv(path, encoding="utf-8-sig")
        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise SystemExit(f"Manifest {path} is missing columns: {missing}")
        frames.append(df[REQUIRED_COLUMNS].copy())

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["sample_id"]).sort_values(["label", "sample_id"]).reset_index(drop=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"rows={len(merged)} -> {out}")
    print(merged.groupby('label').size().to_string())


if __name__ == "__main__":
    main()
