"""Role: choose demo-friendly lifestyle labels from extracted candidates.

Input:
  - data/label_candidates.csv
Output:
  - data/selected_labels_small.json
  - data/selected_lifestyle_candidates.csv
Example:
  python -m src.data.select_lifestyle_labels --max_classes 8 --min_samples 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.config import load_config
from src.utils.io import write_json


PRIORITY_LABELS = [
    "안녕",
    "감사",
    "고맙다",
    "미안",
    "괜찮다",
    "아프다",
    "병원",
    "약",
    "화장실",
    "도와주다",
    "배고프다",
    "먹다",
    "물",
    "우유",
    "자다",
    "가다",
    "오다",
    "알다",
    "모르다",
    "없다",
    "있다",
    "싫다",
    "좋다",
    "남자",
    "여자",
    "지도",
    "걷다",
    "일어나다",
    "몸",
]


def score_label(label: str) -> int:
    if label in PRIORITY_LABELS:
        return 1000 - PRIORITY_LABELS.index(label)
    for idx, keyword in enumerate(PRIORITY_LABELS):
        if keyword in label:
            return 700 - idx
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--max_classes", type=int, default=8)
    parser.add_argument("--min_samples", type=int, default=5)
    args = parser.parse_args()

    config = load_config(args.config)
    df = pd.read_csv(config["paths"]["label_candidates"])
    df["sample_count"] = df["sample_count"].astype(int)
    df["lifestyle_score"] = df["label"].map(score_label)
    candidates = df[(df["sample_count"] >= args.min_samples) & (df["lifestyle_score"] > 0)].copy()
    if candidates.empty:
        candidates = df[df["sample_count"] >= args.min_samples].copy()
        candidates["lifestyle_score"] = 0
    candidates = candidates.sort_values(["lifestyle_score", "sample_count"], ascending=[False, False])
    selected = candidates.head(args.max_classes)["label"].tolist()

    out_csv = Path("data/selected_lifestyle_candidates.csv")
    candidates.to_csv(out_csv, index=False, encoding="utf-8-sig")
    write_json(
        config["paths"]["selected_labels_small"],
        {
            "labels": selected,
            "mode": "quick_test_real_morpheme",
            "min_samples": args.min_samples,
            "note": "Selected from real AI Hub WORD morpheme labels. Keypoint data still required for real training.",
        },
    )
    print({"selected": selected, "candidate_csv": str(out_csv), "candidate_count": int(len(candidates))})


if __name__ == "__main__":
    main()
