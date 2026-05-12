"""Extract one AI Hub keypoint zip and merge it into the master subset manifest.

Typical flow:
  1. Start with zip 01 and an empty/new master manifest
  2. Add zip 02, merge into the same master manifest
  3. Add zip 03, merge again
  4. Repeat the same way for later zips if more data is downloaded

Example:
  python scripts/add_zip_subset.py --zip_path data/raw/.../02_real_word_keypoint.zip
  python scripts/add_zip_subset.py --zip_path data/raw/.../03_real_word_keypoint.zip --master_manifest data/sample_subset_manifest.csv
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_subset_manifest import REQUIRED_COLUMNS  # noqa: E402


def infer_zip_tag(zip_path: Path) -> str:
    stem = zip_path.stem
    if "_real_word_keypoint" in stem:
        return stem.split("_real_word_keypoint", 1)[0]
    return stem


def run_python(args: list[str]) -> None:
    cmd = [sys.executable, *args]
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def merge_manifests(master_path: Path, new_path: Path) -> pd.DataFrame:
    new_df = pd.read_csv(new_path, encoding="utf-8-sig")
    missing = [column for column in REQUIRED_COLUMNS if column not in new_df.columns]
    if missing:
        raise SystemExit(f"New manifest is missing columns: {missing}")

    if master_path.exists():
        master_df = pd.read_csv(master_path, encoding="utf-8-sig")
        missing_master = [column for column in REQUIRED_COLUMNS if column not in master_df.columns]
        if missing_master:
            raise SystemExit(f"Master manifest is missing columns: {missing_master}")
        merged = pd.concat([master_df[REQUIRED_COLUMNS], new_df[REQUIRED_COLUMNS]], ignore_index=True)
    else:
        merged = new_df[REQUIRED_COLUMNS].copy()

    merged = merged.drop_duplicates(subset=["sample_id"]).sort_values(["label", "sample_id"]).reset_index(drop=True)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", required=True, help="Path to 01/02/03_real_word_keypoint.zip")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument(
        "--match_mode",
        choices=["sample_id", "word_angle"],
        default="sample_id",
        help="sample_id requires exact REALxx sample id match; word_angle matches by WORDxxxx + angle across REAL splits.",
    )
    parser.add_argument(
        "--selected_labels_path",
        help="Override selected labels JSON, e.g. data/selected_labels_small_label239.json",
    )
    parser.add_argument("--output_root", default="data/raw", help="Base directory for extracted subset folders")
    parser.add_argument("--master_manifest", default="data/sample_subset_manifest.csv")
    parser.add_argument(
        "--keep_individual_manifest",
        action="store_true",
        help="Keep the per-zip manifest instead of deleting it after merge.",
    )
    args = parser.parse_args()

    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        raise SystemExit(f"Missing zip file: {zip_path}")

    zip_tag = infer_zip_tag(zip_path)
    output_dir = Path(args.output_root) / f"selected_keypoints_{zip_tag}"
    individual_manifest = Path("data") / f"sample_subset_manifest_{zip_tag}.csv"
    master_manifest = Path(args.master_manifest)

    run_python(
        [
            "-m",
            "src.data.extract_keypoint_subset_from_zip",
            "--config",
            args.config,
            "--zip_path",
            str(zip_path),
            "--output_dir",
            str(output_dir),
            "--manifest_output",
            str(individual_manifest),
            "--match_mode",
            args.match_mode,
        ]
        + (["--selected_labels_path", args.selected_labels_path] if args.selected_labels_path else [])
    )

    run_python(["scripts/validate_subset_manifest.py", "--manifest", str(individual_manifest), "--check_paths"])

    merged = merge_manifests(master_manifest, individual_manifest)
    master_manifest.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(master_manifest, index=False, encoding="utf-8-sig")

    if not args.keep_individual_manifest and individual_manifest.exists():
        individual_manifest.unlink()

    print(
        {
            "zip_tag": zip_tag,
            "zip_path": str(zip_path),
            "output_dir": str(output_dir),
            "master_manifest": str(master_manifest),
            "rows": int(len(merged)),
            "counts": merged.groupby("label").size().to_dict() if not merged.empty else {},
        }
    )


if __name__ == "__main__":
    main()
