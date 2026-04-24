"""Extract AI Hub tar downloads safely on Windows.

Role:
  Python tar extractor that handles Korean paths better than some Windows tar
  invocations and then merges *.zip.partN files.

Input:
  data/raw/aihub_downloads/word_morpheme/download.tar
Output:
  extracted/merged zip files under the same folder
Example:
  python scripts/extract_aihub_tar.py data/raw/aihub_downloads/word_morpheme/download.tar
"""

from __future__ import annotations

import argparse
import re
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path


PART_RE = re.compile(r"^(?P<base>.+)\.part(?P<num>\d+)$")


def safe_extract(tar_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:*") as tar:
        for member in tar.getmembers():
            target = (output_dir / member.name).resolve()
            if not str(target).startswith(str(output_dir.resolve())):
                raise RuntimeError(f"Blocked unsafe tar member: {member.name}")
        tar.extractall(output_dir)


def merge_parts(root: Path) -> list[Path]:
    groups: dict[Path, list[tuple[int, Path]]] = defaultdict(list)
    for path in root.rglob("*.part*"):
        match = PART_RE.match(path.name)
        if not match:
            continue
        base_path = path.with_name(match.group("base"))
        groups[base_path].append((int(match.group("num")), path))

    merged: list[Path] = []
    for base_path, parts in groups.items():
        parts = sorted(parts, key=lambda item: item[0])
        base_path.parent.mkdir(parents=True, exist_ok=True)
        with base_path.open("wb") as out:
            for _, part_path in parts:
                with part_path.open("rb") as src:
                    shutil.copyfileobj(src, out)
        merged.append(base_path)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tar_path")
    parser.add_argument("--output_dir")
    parser.add_argument("--merge_parts", action="store_true", default=True)
    args = parser.parse_args()

    tar_path = Path(args.tar_path)
    output_dir = Path(args.output_dir) if args.output_dir else tar_path.parent
    safe_extract(tar_path, output_dir)
    merged = merge_parts(output_dir)
    print({"extracted_to": str(output_dir), "merged": [str(p) for p in merged]})


if __name__ == "__main__":
    main()
