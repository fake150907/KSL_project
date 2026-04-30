"""Role: train/evaluate a quick sklearn baseline.

Input: data/processed/sign_word_subset.npz
Output:
  - outputs/checkpoints/baseline.joblib
  - outputs/reports/baseline_metrics.json
  - outputs/confusion_matrix.png
Example:
  python -m src.models.train_baseline --quick_test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report

from src.data.dataset import load_npz
from src.models.model_baseline import flatten_sequences, make_baseline
from src.utils.config import apply_cli_overrides, load_config
from src.utils.io import write_json


def train_baseline(config: dict) -> dict[str, object]:
    data = load_npz(config["paths"]["processed_npz"])
    train_mask = data.splits == "train"
    val_mask = data.splits != "train"
    if not np.any(val_mask):
        val_mask = train_mask

    model = make_baseline(config["train"]["baseline_model"], int(config["data"]["random_seed"]))
    model.fit(flatten_sequences(data.X[train_mask]), data.y[train_mask])
    pred = model.predict(flatten_sequences(data.X[val_mask]))
    acc = float(accuracy_score(data.y[val_mask], pred))

    checkpoint = Path(config["paths"]["checkpoints_dir"]) / "baseline.joblib"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": data.labels}, checkpoint)

    labels_present = sorted(set(data.y[val_mask].tolist()) | set(pred.tolist()))
    report = classification_report(
        data.y[val_mask],
        pred,
        labels=labels_present,
        target_names=[data.labels[i] for i in labels_present],
        zero_division=0,
        output_dict=True,
    )
    metrics = {"accuracy": acc, "labels": data.labels, "classification_report": report}
    try:
        import pandas as pd

        manifest = pd.read_csv(config["paths"]["subset_manifest"], encoding="utf-8-sig")
        val_indices = np.where(val_mask)[0]
        if "angle" in manifest.columns and len(manifest) == len(data.y):
            angle_rows = []
            for angle in sorted(manifest.loc[val_indices, "angle"].fillna("").unique().tolist()):
                mask = manifest.loc[val_indices, "angle"].fillna("").to_numpy() == angle
                angle_rows.append(
                    {
                        "angle": angle,
                        "samples": int(np.sum(mask)),
                        "accuracy": float(accuracy_score(data.y[val_mask][mask], pred[mask])) if np.any(mask) else 0.0,
                    }
                )
            metrics["angle_metrics"] = angle_rows
    except Exception as exc:
        metrics["angle_metrics_error"] = str(exc)
    write_json(Path(config["paths"]["reports_dir"]) / "baseline_metrics.json", metrics)

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(
        data.y[val_mask],
        pred,
        display_labels=[data.labels[i] for i in labels_present],
        labels=labels_present,
        xticks_rotation=45,
        ax=ax,
        colorbar=False,
    )
    fig.tight_layout()
    fig.savefig(Path(config["paths"]["outputs_dir"]) / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    return {"accuracy": acc, "checkpoint": str(checkpoint), "validation_samples": int(np.sum(val_mask))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--quick_test", action="store_true", default=None)
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    print(train_baseline(config))


if __name__ == "__main__":
    main()
