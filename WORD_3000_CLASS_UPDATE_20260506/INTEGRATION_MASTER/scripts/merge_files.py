"""Merge selected FILE-based manifests and NPZ files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


EXPECTED_CLASS_COUNT = 3000
EXPECTED_SEQUENCE_LENGTH = 32
EXPECTED_FEATURE_COUNT = 225
REQUIRED_NPZ_KEYS = {"X", "y", "splits", "sample_ids", "labels"}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return [dict(row) for row in reader]


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


def sample_id_match_keys(row: dict[str, str]) -> list[str]:
    sample_id = row.get("sample_id", "")
    keys = [sample_id] if sample_id else []
    actor_id = row.get("actor_id", "").strip()
    if sample_id and actor_id.isdigit():
        real_token = f"_REAL{int(actor_id):02d}_"
        keys.append(re.sub(r"_FILE\d+_", real_token, sample_id))
    return list(dict.fromkeys(key for key in keys if key))


def validate_npz(tag: str, npz_path: Path, data: np.lib.npyio.NpzFile) -> None:
    missing = REQUIRED_NPZ_KEYS - set(data.files)
    if missing:
        raise SystemExit(f"{tag} missing NPZ keys: {sorted(missing)}")
    x = data["X"]
    if x.ndim != 3 or int(x.shape[1]) != EXPECTED_SEQUENCE_LENGTH or int(x.shape[2]) != EXPECTED_FEATURE_COUNT:
        raise SystemExit(f"{tag} invalid X shape: {x.shape}")
    labels = [str(label) for label in data["labels"].tolist()]
    if len(labels) != EXPECTED_CLASS_COUNT or labels[0] != "WORD0001" or labels[-1] != "WORD3000":
        raise SystemExit(f"{tag} labels must be WORD0001..WORD3000, got count={len(labels)}")
    y = data["y"]
    if len(y) and (int(np.min(y)) < 0 or int(np.max(y)) >= EXPECTED_CLASS_COUNT):
        raise SystemExit(f"{tag} y values must be in range 0..2999")


def find_tag_folder(tag: str, shard_roots: list[Path]) -> Path:
    matches = [root / tag for root in shard_roots if (root / tag).exists()]
    if not matches:
        raise SystemExit(f"Missing folder for {tag}. Searched: {', '.join(str(root) for root in shard_roots)}")
    if len(matches) > 1:
        raise SystemExit(f"Ambiguous folder for {tag}: {', '.join(str(path) for path in matches)}")
    return matches[0]


def find_manifest(folder: Path, tag: str) -> Path:
    expected = folder / f"shard_manifest_{tag}.csv"
    if expected.exists():
        return expected
    candidates = sorted(folder.glob(f"shard_manifest*{tag}.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        candidates = sorted(folder.glob("shard_manifest*.csv"))
        if len(candidates) == 1:
            return candidates[0]
    if not candidates:
        raise SystemExit(f"Missing manifest for {tag}: {expected}")
    raise SystemExit(f"Ambiguous manifest for {tag}: {', '.join(str(path) for path in candidates)}")


def find_single_file(folder: Path, expected_name: str, fallback_glob: str, description: str, required: bool = True) -> Path | None:
    expected = folder / expected_name
    if expected.exists():
        return expected
    candidates = sorted(folder.glob(fallback_glob))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        if required:
            raise SystemExit(f"Missing {description} for {folder.name}: {expected}")
        return None
    raise SystemExit(f"Ambiguous {description} for {folder.name}: {', '.join(str(path) for path in candidates)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", default="team_handover_outputs")
    parser.add_argument("--shard-roots", nargs="*")
    parser.add_argument("--file-tags", nargs="+", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument(
        "--word-id-label-map",
        default="team_handover_2026-04-29/WORD_3000_CLASS_UPDATE_20260506/label_map_word_id_3000.json",
        help="Input WORD ID class map used to rebuild merged y.",
    )
    parser.add_argument("--merge-report", required=True)
    parser.add_argument("--merge-failed", required=True)
    args = parser.parse_args()
    word_id_label_map = load_word_id_label_map(Path(args.word_id_label_map))

    shard_roots = [Path(root) for root in (args.shard_roots or [args.shard_root])]
    all_rows: list[dict[str, str]] = []
    all_x: list[np.ndarray] = []
    all_word_ids: list[str] = []
    all_splits: list[str] = []
    all_sample_ids: list[str] = []
    failed_rows: list[dict[str, str]] = []
    seen_sample_ids: set[str] = set()

    for tag in args.file_tags:
        folder = find_tag_folder(tag, shard_roots)
        manifest_path = find_manifest(folder, tag)
        npz_path = find_single_file(folder, f"mediapipe_npz_{tag}.npz", "mediapipe_npz_*.npz", "NPZ")
        failed_path = find_single_file(folder, f"failed_samples_{tag}.csv", "*failed*.csv", "failed CSV", required=False)

        rows = read_manifest(manifest_path)
        rows_by_sample_id: dict[str, dict[str, str]] = {}
        for row in rows:
            for key in sample_id_match_keys(row):
                if key in rows_by_sample_id and rows_by_sample_id[key] is not row:
                    raise SystemExit(f"{tag} duplicate manifest match key: {key}")
                rows_by_sample_id[key] = row
        data = np.load(npz_path, allow_pickle=True)
        validate_npz(tag, npz_path, data)

        x = data["X"].astype(np.float32)
        sample_ids = [str(item) for item in data["sample_ids"].tolist()]
        splits = [str(item) for item in data["splits"].tolist()]
        if len(sample_ids) != len(x) or len(splits) != len(x):
            raise SystemExit(f"{tag} NPZ length mismatch.")

        for sample_index, sample_id in enumerate(sample_ids):
            if sample_id in seen_sample_ids:
                raise SystemExit(f"Duplicate sample_id across merged inputs: {sample_id}")
            seen_sample_ids.add(sample_id)
            row = rows_by_sample_id.get(sample_id)
            if row is None:
                raise SystemExit(f"{tag} sample missing from manifest: {sample_id}")
            row = dict(row)
            original_sample_id = row.get("sample_id", "")
            if original_sample_id != sample_id:
                row["original_manifest_sample_id"] = original_sample_id
                row["sample_id"] = sample_id
            row["source_file_tag"] = tag
            row["source_npz_index"] = str(sample_index)
            row["status"] = "processed"
            all_rows.append(row)
            word_id = str(row.get("word_id", "")).strip()
            if word_id not in word_id_label_map:
                raise SystemExit(f"{tag} invalid or missing word_id for {sample_id}: {word_id}")
            all_word_ids.append(word_id)

        all_x.append(x)
        all_splits.extend(splits)
        all_sample_ids.extend(sample_ids)

        if failed_path and failed_path.exists():
            with failed_path.open("r", encoding="utf-8-sig", newline="") as file:
                reader = csv.DictReader(file)
                for failed in reader:
                    if any(failed.values()):
                        failed = dict(failed)
                        failed["source_file_tag"] = tag
                        failed_rows.append(failed)

    labels = [f"WORD{i:04d}" for i in range(1, EXPECTED_CLASS_COUNT + 1)]
    label_to_id = {word_id: word_id_label_map[word_id] for word_id in labels}
    y = np.asarray([word_id_label_map[word_id] for word_id in all_word_ids], dtype=np.int64)
    merged_x = np.concatenate(all_x, axis=0)

    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        X=merged_x,
        y=y,
        splits=np.asarray(all_splits),
        sample_ids=np.asarray(all_sample_ids),
        labels=np.asarray(labels),
    )

    output_manifest = Path(args.output_manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in all_rows for key in row.keys()))
    with output_manifest.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    label_map = Path(args.label_map)
    label_map.write_text(
        json.dumps({"labels": labels, "label_to_id": label_to_id}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    merge_failed = Path(args.merge_failed)
    failed_fieldnames = sorted({key for row in failed_rows for key in row}) or [
        "row_index",
        "sample_id",
        "label",
        "video_path",
        "error",
        "source_file_tag",
    ]
    with merge_failed.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=failed_fieldnames)
        writer.writeheader()
        writer.writerows(failed_rows)

    label_counts = Counter(row.get("label", "") for row in all_rows if row.get("label"))
    word_id_counts = Counter(all_word_ids)
    file_counts = Counter(row["source_file_tag"] for row in all_rows)
    real_counts = Counter(row.get("real_id", "") for row in all_rows)
    report = Path(args.merge_report)
    report.write_text(
        "\n".join(
            [
                "# FILE Merge Report",
                "",
                f"file_tags: {', '.join(args.file_tags)}",
                f"manifest_rows: {len(all_rows)}",
                f"npz_samples: {int(merged_x.shape[0])}",
                f"npz_shape: {tuple(int(value) for value in merged_x.shape)}",
                "extraction_basis: FILE zip number, not REAL signer number",
                "layout: mediapipe_xyz",
                "feature_count: 225",
                "sequence_length: 32",
                f"output_manifest: {output_manifest}",
                f"output_npz: {output_npz}",
                f"label_map: {label_map}",
                f"merge_failed_samples: {merge_failed}",
                "",
                "## FILE Counts",
                *[f"- {key}: {value}" for key, value in sorted(file_counts.items())],
                "",
                "## Label Counts",
                *[f"- {key}: {value}" for key, value in sorted(label_counts.items())],
                "",
                "## WORD ID Counts",
                *[f"- {key}: {value}" for key, value in sorted(word_id_counts.items())],
                "",
                "## REAL Metadata Counts",
                *[f"- {key}: {value}" for key, value in sorted(real_counts.items())],
                "",
                f"failed_rows: {len(failed_rows)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        {
            "manifest_rows": len(all_rows),
            "npz_samples": int(merged_x.shape[0]),
            "shape": tuple(int(value) for value in merged_x.shape),
            "labels": labels,
            "output_manifest": str(output_manifest),
            "output_npz": str(output_npz),
            "merge_report": str(report),
            "failed_rows": len(failed_rows),
        }
    )


if __name__ == "__main__":
    main()

