"""Role: train/evaluate sequence models for the MVP.

Input: data/processed/sign_word_subset.npz
Output:
  - outputs/checkpoints/sequence_model.pt
  - outputs/reports/sequence_metrics.json
Example:
  python -m src.models.train_sequence --epochs 2 --quick_test
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from sklearn.metrics import accuracy_score, classification_report
from torch import nn

from src.data.dataset import make_loaders
from src.models.model_sequence import build_sequence_model
from src.utils.config import apply_cli_overrides, load_config
from src.utils.io import write_json
from src.utils.seed import set_seed


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())
            y_true.extend(y.tolist())
    return y_true, y_pred


def train_sequence(config: dict) -> dict[str, object]:
    set_seed(int(config["data"]["random_seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, data = make_loaders(config["paths"]["processed_npz"], int(config["train"]["batch_size"]))
    model = build_sequence_model(
        input_size=int(data.X.shape[2]),
        num_classes=len(data.labels),
        hidden_size=int(config["train"]["hidden_size"]),
        num_layers=int(config["train"]["num_layers"]),
        dropout=float(config["train"]["dropout"]),
        rnn_type=str(config["train"].get("rnn_type", "gru")),
        model_type=str(config["train"].get("model_type", "rnn")),
        conv_channels=int(config["train"].get("conv_channels", 128)),
        num_heads=int(config["train"].get("num_heads", 4)),
        sequence_length=int(config["data"].get("sequence_length", data.X.shape[1])),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(config["train"]["learning_rate"]))
    loss_fn = nn.CrossEntropyLoss()

    history = []
    best_acc = -1.0
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    for epoch in range(int(config["train"]["epochs"])):
        model.train()
        total = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(y)
        y_true, y_pred = evaluate(model, val_loader, device)
        acc = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
        history.append({"epoch": epoch + 1, "loss": total / max(1, len(train_loader.dataset)), "val_accuracy": acc})
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())

    checkpoint = Path(config["paths"]["checkpoints_dir"]) / "sequence_model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": data.labels,
            "input_size": int(data.X.shape[2]),
            "config": config,
            "model_type": str(config["train"].get("model_type", "rnn")),
            "best_epoch": best_epoch,
            "best_accuracy": best_acc,
        },
        checkpoint,
    )
    y_true, y_pred = evaluate(model, val_loader, device)
    labels_present = sorted(set(y_true) | set(y_pred))
    metrics = {
        "history": history,
        "device": str(device),
        "model_type": str(config["train"].get("model_type", "rnn")),
        "best_epoch": best_epoch,
        "best_accuracy": best_acc,
        "accuracy": float(accuracy_score(y_true, y_pred)) if y_true else 0.0,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels_present,
            target_names=[data.labels[i] for i in labels_present],
            zero_division=0,
            output_dict=True,
        )
        if y_true
        else {},
    }
    write_json(Path(config["paths"]["reports_dir"]) / "sequence_metrics.json", metrics)
    return {
        "checkpoint": str(checkpoint),
        "history": history,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_accuracy": best_acc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--quick_test", action="store_true", default=None)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--rnn_type", choices=["gru", "lstm"])
    parser.add_argument("--model_type", choices=["rnn", "cnn_gru", "bigru_attention", "transformer"])
    parser.add_argument("--conv_channels", type=int)
    parser.add_argument("--num_heads", type=int)
    args = parser.parse_args()
    config = apply_cli_overrides(load_config(args.config), args)
    print(train_sequence(config))


if __name__ == "__main__":
    main()
