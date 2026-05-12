"""Export all morpheme target samples for the currently selected labels.

Input:
  - data/selected_labels_small.json
  - morpheme JSON files under data/raw
Output:
  - data/selected_label_targets.csv
Example:
  python scripts/export_selected_targets.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.extract_keypoint_subset_from_zip import collect_targets  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default="data/selected_label_targets.csv")
    parser.add_argument(
        "--max_samples_per_class",
        type=int,
        default=100000,
        help="Large default so all currently selected-label morpheme targets are exported.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    df = collect_targets(config, max_samples_per_class=args.max_samples_per_class)
    if df.empty:
        raise SystemExit("No selected-label targets found in morpheme JSON files.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["label", "sample_id"]).reset_index(drop=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"rows={len(df)} -> {out}")
    print(df.groupby('label').size().to_string())


if __name__ == "__main__":
    main()
