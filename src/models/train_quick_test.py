"""Role: one-command lightweight validation pipeline.

Output:
  - selected labels, manifest, processed tensors, baseline checkpoint, quick inference
Example:
  python -m src.models.train_quick_test --make_dummy --max_classes 5 --max_samples_per_class 10 --epochs 1
"""

from __future__ import annotations

import argparse

from src.data.build_small_subset import build_manifest
from src.data.extract_labels import choose_small_labels, discover_labels, iter_json_files
from src.data.preprocess_keypoints import preprocess
from src.inference.infer import predict_baseline
from src.models.train_baseline import train_baseline
from src.utils.config import apply_cli_overrides, load_config
from src.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--quick_test", action="store_true", default=True)
    parser.add_argument("--max_classes", type=int)
    parser.add_argument("--max_samples_per_class", type=int)
    parser.add_argument("--sequence_length", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--make_dummy", action="store_true")
    args = parser.parse_args()

    config = apply_cli_overrides(load_config(args.config), args)
    paths = iter_json_files(config["data"]["morpheme_globs"])
    df = discover_labels(paths)
    labels = choose_small_labels(df, int(config["data"]["max_classes"]), int(config["data"]["min_samples_per_class"]))
    write_json(config["paths"]["selected_labels_small"], {"labels": labels, "mode": "quick_test", "source_file_count": len(paths)})

    manifest = build_manifest(config, make_dummy=args.make_dummy or len(paths) == 0)
    manifest.to_csv(config["paths"]["subset_manifest"], index=False, encoding="utf-8-sig")
    prep = preprocess(config)
    metrics = train_baseline(config)

    import numpy as np

    data = np.load(config["paths"]["processed_npz"], allow_pickle=True)
    label, confidence = predict_baseline(metrics["checkpoint"], data["X"][0])
    print({"preprocess": prep, "baseline": metrics, "quick_prediction": {"label": label, "confidence": confidence}})


if __name__ == "__main__":
    main()
