"""Merge shard manifests and NPZ files using fixed 3000 WORD-ID labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


EXPECTED_CLASS_COUNT = 3000
EXPECTED_SEQUENCE_LENGTH = 32
EXPECTED_FEATURE_COUNT = 225
REQUIRED_NPZ_KEYS = {"X", "y", "splits", "sample_ids", "labels"}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [dict(row) for row in csv.DictReader(file)]


def load_word_id_label_map(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8-sig") as file:
        label_map = json.load(file)
    if not isinstance(label_map, dict):
        raise SystemExit(f"WORD ID label map must be a JSON object: {path}")
    parsed = {str(word_id): int(class_id) for word_id, class_id in label_map.items()}
    if len(parsed) != EXPECTED_CLASS_COUNT:
        raise SystemExit(f"WORD ID label map must contain {EXPECTED_CLASS_COUNT} entries, got {len(parsed)}")
    if sorted(parsed.values()) != list(range(EXPECTED_CLASS_COUNT)):
        raise SystemExit("WORD ID label map class ids must be exactly 0..2999")
    return parsed


def expected_labels() -> list[str]:
    return [f"WORD{i:04d}" for i in range(1, EXPECTED_CLASS_COUNT + 1)]


def validate_shard_npz(npz_path: Path, data: np.lib.npyio.NpzFile) -> None:
    missing = REQUIRED_NPZ_KEYS - set(data.files)
    if missing:
        raise SystemExit(f"{npz_path} missing NPZ keys: {sorted(missing)}")

    x = data["X"]
    y = data["y"]
    splits = data["splits"]
    sample_ids = data["sample_ids"]
    labels = [str(label) for label in data["labels"].tolist()]

    if x.ndim != 3:
        raise SystemExit(f"{npz_path} X must be 3D, got shape {x.shape}")
    if int(x.shape[1]) != EXPECTED_SEQUENCE_LENGTH:
        raise SystemExit(f"{npz_path} sequence_length must be {EXPECTED_SEQUENCE_LENGTH}, got {x.shape[1]}")
    if int(x.shape[2]) != EXPECTED_FEATURE_COUNT:
        raise SystemExit(f"{npz_path} feature_count must be {EXPECTED_FEATURE_COUNT}, got {x.shape[2]}")
    if len(sample_ids) != len(x) or len(y) != len(x) or len(splits) != len(x):
        raise SystemExit(
            f"{npz_path} length mismatch: X={len(x)}, y={len(y)}, splits={len(splits)}, sample_ids={len(sample_ids)}"
        )
    if len(labels) != EXPECTED_CLASS_COUNT or labels[0] != "WORD0001" or labels[-1] != "WORD3000":
        raise SystemExit(f"{npz_path} labels must be WORD0001..WORD3000, got count={len(labels)}")
    if len(y) and (int(np.min(y)) < 0 or int(np.max(y)) >= EXPECTED_CLASS_COUNT):
        raise SystemExit(f"{npz_path} y values must be in range 0..2999")


def validate_manifest_against_npz(tag: str, rows_by_sample_id: dict[str, dict[str, str]], sample_ids: list[str]) -> None:
    processed_rows = [row for row in rows_by_sample_id.values() if row.get("status") in {"processed", "ok"}]
    if processed_rows and len(processed_rows) != len(sample_ids):
        raise SystemExit(
            f"{tag} processed manifest rows ({len(processed_rows)}) do not match NPZ samples ({len(sample_ids)})"
        )
    missing = [sample_id for sample_id in sample_ids if sample_id not in rows_by_sample_id]
    if missing:
        raise SystemExit(f"{tag} sample_ids missing from manifest: {missing[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", default="team_handover_outputs")
    parser.add_argument("--output-manifest", default="data/processed/mediapipe_master_manifest.csv")
    parser.add_argument("--output-npz", default="data/processed/mediapipe_train_dataset.npz")
    parser.add_argument("--label-map", default="data/processed/label_map.json")
    parser.add_argument(
        "--word-id-label-map",
        default="team_handover_2026-04-29/WORD_3000_CLASS_UPDATE_20260506/label_map_word_id_3000.json",
        help="Input WORD ID class map used to rebuild merged y.",
    )
    parser.add_argument("--merge-report", default="reports/merge_report.md")
    parser.add_argument("--merge-failed", default="reports/merge_failed_samples.csv")
    args = parser.parse_args()

    word_id_label_map = load_word_id_label_map(Path(args.word_id_label_map))
    shard_root = Path(args.shard_root)
    manifest_paths = sorted(shard_root.glob("*/shard_manifest_*.csv"))
    npz_paths = sorted(shard_root.glob("*/mediapipe_npz_*.npz"))

    if not manifest_paths:
        raise SystemExit(f"No shard manifests found under {shard_root}")
    if not npz_paths:
        raise SystemExit(f"No shard NPZ files found under {shard_root}")

    rows: list[dict[str, str]] = []
    manifest_by_tag: dict[str, dict[str, dict[str, str]]] = {}
    for manifest_path in manifest_paths:
        tag = manifest_path.stem.replace("shard_manifest_", "")
        part_rows = read_manifest(manifest_path)
        manifest_by_tag[tag] = {row["sample_id"]: row for row in part_rows}
        for index, row in enumerate(part_rows):
            row = dict(row)
            row.setdefault("npz_index", str(index))
            row.setdefault("status", "pending")
            row["source_shard"] = tag
            rows.append(row)

    all_x: list[np.ndarray] = []
    all_y: list[int] = []
    all_splits: list[str] = []
    all_sample_ids: list[str] = []

    for npz_path in npz_paths:
        tag = npz_path.stem.replace("mediapipe_npz_", "")
        manifest = manifest_by_tag.get(tag)
        if manifest is None:
            raise SystemExit(f"Missing matching manifest for {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        validate_shard_npz(npz_path, data)
        x = data["X"].astype(np.float32)
        sample_ids = [str(item) for item in data["sample_ids"].tolist()]
        splits = [str(item) for item in data["splits"].tolist()]
        validate_manifest_against_npz(tag, manifest, sample_ids)

        for sample_index, sample_id in enumerate(sample_ids):
            row = manifest.get(sample_id)
            if row is None:
                raise SystemExit(f"sample_id missing from shard manifest: {sample_id}")
            word_id = str(row.get("word_id", "")).strip()
            if word_id not in word_id_label_map:
                raise SystemExit(f"{tag} invalid or missing word_id for {sample_id}: {word_id}")
            all_y.append(word_id_label_map[word_id])
            row["npz_index"] = str(sample_index)
            row["status"] = "processed"

        all_x.append(x)
        all_splits.extend(splits)
        all_sample_ids.extend(sample_ids)

    merged_x = np.concatenate(all_x, axis=0)
    labels = expected_labels()

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        X=merged_x,
        y=np.asarray(all_y, dtype=np.int64),
        splits=np.asarray(all_splits),
        sample_ids=np.asarray(all_sample_ids),
        labels=np.asarray(labels),
    )

    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with output_manifest.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    label_map = Path(args.label_map)
    label_map.parent.mkdir(parents=True, exist_ok=True)
    label_map.write_text(
        json.dumps({"labels": labels, "label_to_id": word_id_label_map}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    failed_rows: list[dict[str, str]] = []
    for failed_path in sorted(shard_root.glob("*/failed_samples_*.csv")):
        with failed_path.open("r", encoding="utf-8-sig", newline="") as file:
            for row in csv.DictReader(file):
                row = dict(row)
                if any(value for value in row.values()):
                    row["source_file"] = failed_path.as_posix()
                    failed_rows.append(row)

    merge_failed = Path(args.merge_failed)
    merge_failed.parent.mkdir(parents=True, exist_ok=True)
    failed_fieldnames = sorted({key for row in failed_rows for key in row.keys()}) or [
        "row_index",
        "sample_id",
        "label",
        "word_id",
        "video_path",
        "error",
        "source_file",
    ]
    with merge_failed.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=failed_fieldnames)
        writer.writeheader()
        writer.writerows(failed_rows)

    label_counts = Counter(row.get("label", "") for row in rows if row.get("label"))
    word_id_counts = Counter(row.get("word_id", "") for row in rows if row.get("word_id"))
    shard_counts = Counter(row.get("source_shard", "") for row in rows if row.get("source_shard"))
    report = Path(args.merge_report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# Merge Report",
                "",
                f"shard_root: {shard_root}",
                f"manifest_rows: {len(rows)}",
                f"npz_samples: {int(merged_x.shape[0])}",
                f"npz_shape: {tuple(int(value) for value in merged_x.shape)}",
                f"class_count: {EXPECTED_CLASS_COUNT}",
                f"expected_sequence_length: {EXPECTED_SEQUENCE_LENGTH}",
                f"expected_feature_count: {EXPECTED_FEATURE_COUNT}",
                "expected_layout: mediapipe_xyz",
                "expected_normalization: shoulder-center + shoulder-width scale",
                "label_basis: WORD ID, not Korean label text",
                f"output_manifest: {output_manifest}",
                f"output_npz: {output_npz}",
                f"label_map: {label_map}",
                f"merge_failed_samples: {merge_failed}",
                "",
                "## Shard Counts",
                *[f"- {key}: {value}" for key, value in sorted(shard_counts.items())],
                "",
                "## Label Counts",
                *[f"- {key}: {value}" for key, value in sorted(label_counts.items())],
                "",
                "## WORD ID Counts",
                *[f"- {key}: {value}" for key, value in sorted(word_id_counts.items())],
                "",
                f"failed_rows: {len(failed_rows)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        {
            "merged_samples": int(merged_x.shape[0]),
            "shape": tuple(int(value) for value in merged_x.shape),
            "class_count": EXPECTED_CLASS_COUNT,
            "output_npz": str(output_npz),
            "merge_report": str(report),
        }
    )


if __name__ == "__main__":
    main()

