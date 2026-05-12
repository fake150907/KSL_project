"""Role: run single-sample inference with a trained baseline or GRU model.

Input: processed NPZ sample or keypoint JSON
Output: predicted expression and confidence
Example:
  python -m src.inference.infer --sample_index 0 --model baseline
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from src.data.dataset import load_npz
from src.models.model_baseline import flatten_sequences
from src.utils.config import load_config


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


def predict_baseline(checkpoint: str, X: np.ndarray) -> tuple[str, float]:
    bundle = joblib.load(checkpoint)
    model = bundle["model"]
    labels = bundle["labels"]
    flat = flatten_sequences(X[None, ...])
    pred = int(model.predict(flat)[0])
    if hasattr(model, "predict_proba"):
        confidence = float(np.max(model.predict_proba(flat)[0]))
    else:
        confidence = 1.0
    return labels[pred], confidence


def predict_sequence(checkpoint: str, X: np.ndarray) -> tuple[str, float]:
    from src.models.model_sequence import build_sequence_model

    bundle = torch.load(checkpoint, map_location="cpu")
    train_config = bundle.get("config", {}).get("train", {})
    model = build_sequence_model(
        input_size=bundle["input_size"],
        num_classes=len(bundle["labels"]),
        hidden_size=int(train_config.get("hidden_size", 64)),
        num_layers=int(train_config.get("num_layers", 1)),
        dropout=float(train_config.get("dropout", 0.1)),
        rnn_type=str(train_config.get("rnn_type", "gru")),
        model_type=str(bundle.get("model_type", train_config.get("model_type", "rnn"))),
        conv_channels=int(train_config.get("conv_channels", 128)),
        num_heads=int(train_config.get("num_heads", 4)),
        sequence_length=int(bundle.get("config", {}).get("data", {}).get("sequence_length", X.shape[0])),
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X[None, ...], dtype=torch.float32))[0].numpy()
    probs = softmax(logits)
    pred = int(np.argmax(probs))
    return bundle["labels"][pred], float(probs[pred])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--model", choices=["baseline", "sequence"], default="baseline")
    parser.add_argument("--sample_index", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    data = load_npz(config["paths"]["processed_npz"])
    idx = min(args.sample_index, len(data.X) - 1)
    if args.model == "baseline":
        ckpt = str(Path(config["paths"]["checkpoints_dir"]) / "baseline.joblib")
        label, confidence = predict_baseline(ckpt, data.X[idx])
    else:
        ckpt = str(Path(config["paths"]["checkpoints_dir"]) / "sequence_model.pt")
        label, confidence = predict_sequence(ckpt, data.X[idx])
    true_label = data.labels[int(data.y[idx])]
    print({"sample_index": idx, "true": true_label, "predicted": label, "confidence": round(confidence, 4)})

    if args.model == "baseline":
        bundle = joblib.load(ckpt)
        pred = bundle["model"].predict(flatten_sequences(data.X))
        print({"processed_accuracy": round(float(accuracy_score(data.y, pred)), 4)})


if __name__ == "__main__":
    main()
