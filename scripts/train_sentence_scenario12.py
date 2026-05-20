"""Fine-tune the 2000-class sentence CNNGRUAttn model down to the 12 demo scenario classes.

Backbone weights are warm-started from outputs/checkpoints/sentence_stage2_v2.pt.
The 2000-way FC head is sliced to 12 rows (one per target SEN id) and then
fine-tuned end-to-end on the SEN-restricted TRAIN/VAL/TEAM NPZs.

Outputs:
  outputs/checkpoints/sentence_scenario12.pt          (model_state + model_config + labels)
  model_results/label_map_sen_scenario12.json        (SEN -> idx 0..11)
  model_results/idx_to_label_sen_scenario12.json     (idx -> {label, sen_id})
  outputs/reports/sentence_scenario12_training.md    (per-class metrics + curves)
  outputs/reports/sentence_scenario12_curve.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.cnngru_attn import CNNGRUAttn

# ---------------------------------------------------------------------------
# Paths and target classes
# ---------------------------------------------------------------------------
TRAIN_NPZ = Path(r"C:\github\ai-project-01\team_handover_outputs\00.DATASETS\TRAIN_SENTENCE\mediapipe_npz_SENTENCE_FILE01_32.npz")
VAL_NPZ   = Path(r"C:\github\ai-project-01\team_handover_outputs\00.DATASETS\VALIDATION_SENTENCE\mediapipe_npz_VALIDATION_SENTENCE.npz")
TEAM_NPZ  = Path(r"C:\github\ai-project-01\team_handover_outputs\00.DATASETS\TEAM_VIDEO\mediapipe_npz_TEAM_SENTENCE.npz")

SRC_CKPT  = REPO_ROOT / "outputs" / "checkpoints" / "sentence_stage2_v2.pt"
OUT_CKPT  = REPO_ROOT / "outputs" / "checkpoints" / "sentence_scenario12.pt"
LBL_MAP   = REPO_ROOT / "model_results" / "label_map_sen_scenario12.json"
IDX_MAP   = REPO_ROOT / "model_results" / "idx_to_label_sen_scenario12.json"
REPORT_MD = REPO_ROOT / "outputs" / "reports" / "sentence_scenario12_training.md"
CURVE_PNG = REPO_ROOT / "outputs" / "reports" / "sentence_scenario12_curve.png"

# Ordered as the user specified — this defines the new 12-class index space (0..11).
TARGET_LABELS: list[str] = [
    "SEN0109", "SEN0110",
    "SEN0169", "SEN0170",
    "SEN0175", "SEN0176",
    "SEN0278", "SEN0279",
    "SEN0322", "SEN0354", "SEN0355", "SEN1817",
]

# Demo-priority classes the user called out — surfaced in the per-class report.
DEMO_PRIORITY = {"SEN0322", "SEN1817", "SEN0169", "SEN0175", "SEN0279"}

# Human-readable labels (mirrors backend/inference/scenario_lookup.json).
DISPLAY_TEXT = {
    "SEN0109": "가능한가요?",
    "SEN0110": "가능한가요?",
    "SEN0169": "신분증이 여기 있어요",
    "SEN0170": "신분증이 여기 있어요",
    "SEN0175": "여기 있어요",
    "SEN0176": "여기 있어요",
    "SEN0278": "지하철에서 잃어버렸어요",
    "SEN0279": "지하철에서 잃어버렸어요",
    "SEN0322": "잃어버렸어요",
    "SEN0354": "안녕하세요",
    "SEN0355": "감사합니다",
    "SEN1817": "카드를 잃어버렸어요",
}

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
EPOCHS         = 60
BATCH_SIZE     = 32
LR_BACKBONE    = 5e-5
LR_HEAD        = 5e-4
WEIGHT_DECAY   = 1e-4
LABEL_SMOOTH   = 0.05
GRAD_CLIP      = 1.0
SEED           = 42
EARLY_PATIENCE = 15  # epochs without val_acc improvement → stop


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class KeypointDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32, copy=False))
        self.y = torch.from_numpy(y.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def filter_npz_to_targets(
    npz_path: Path,
    target_old_idx: list[int],
    old_to_new: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"]
    y = z["y"]
    keep_mask = np.isin(y, target_old_idx)
    X_f = X[keep_mask]
    y_old = y[keep_mask]
    y_new = np.array([old_to_new[int(v)] for v in y_old], dtype=np.int64)
    return X_f, y_new


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
def build_finetune_model(src_ckpt: Path, target_old_idx: list[int]) -> CNNGRUAttn:
    bundle = torch.load(src_ckpt, map_location="cpu", weights_only=False)
    src_state = bundle["model_state"]
    src_cfg = dict(bundle["model_config"])

    # New model: same backbone config, but with num_classes=12.
    new_cfg = dict(src_cfg)
    new_cfg["num_classes"] = len(target_old_idx)
    model = CNNGRUAttn(**new_cfg)

    # Slice FC weights/bias for the 12 target rows in the new (0..11) order.
    src_fc_w = src_state["fc.weight"]   # (2000, 512)
    src_fc_b = src_state["fc.bias"]     # (2000,)
    slice_w = src_fc_w[target_old_idx, :].clone()
    slice_b = src_fc_b[target_old_idx].clone()

    # Drop the old FC keys from the state dict and load the rest strictly.
    backbone_state = {k: v for k, v in src_state.items() if not k.startswith("fc.")}
    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    # Expect only fc.* in missing (those we are about to set manually).
    missing = [m for m in missing if not m.startswith("fc.")]
    if missing:
        raise RuntimeError(f"Unexpected missing backbone keys: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected ckpt keys: {unexpected}")

    with torch.no_grad():
        model.fc.weight.copy_(slice_w)
        model.fc.bias.copy_(slice_b)

    return model, new_cfg, bundle.get("metrics", {}), bundle.get("epoch", -1)


def param_groups(model: nn.Module) -> list[dict]:
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("fc."):
            head_params.append(p)
        else:
            backbone_params.append(p)
    return [
        {"params": backbone_params, "lr": LR_BACKBONE, "name": "backbone"},
        {"params": head_params,     "lr": LR_HEAD,     "name": "head"},
    ]


# ---------------------------------------------------------------------------
# Train / Eval loops
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_truth = []
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(X)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if GRAD_CLIP:
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            total_loss += float(loss) * y.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum())
            total += int(y.size(0))
            all_preds.append(preds.cpu().numpy())
            all_truth.append(y.cpu().numpy())
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    preds = np.concatenate(all_preds) if all_preds else np.empty(0, dtype=np.int64)
    truth = np.concatenate(all_truth) if all_truth else np.empty(0, dtype=np.int64)
    return avg_loss, acc, preds, truth


def per_class_report(truth: np.ndarray, preds: np.ndarray, labels: list[str]) -> list[dict]:
    rows = []
    for new_idx, lbl in enumerate(labels):
        mask = truth == new_idx
        n = int(mask.sum())
        c = int((preds[mask] == new_idx).sum()) if n else 0
        rows.append({
            "label": lbl,
            "display": DISPLAY_TEXT.get(lbl, lbl),
            "support": n,
            "correct": c,
            "acc": (c / n) if n else 0.0,
            "priority": lbl in DEMO_PRIORITY,
        })
    return rows


def top_confusions(truth: np.ndarray, preds: np.ndarray, labels: list[str], k: int = 5) -> list[str]:
    wrong = truth != preds
    if not wrong.any():
        return []
    pairs: dict[tuple[int, int], int] = {}
    for t, p in zip(truth[wrong], preds[wrong]):
        pairs[(int(t), int(p))] = pairs.get((int(t), int(p)), 0) + 1
    ranked = sorted(pairs.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return [f"{labels[t]} → {labels[p]} : {n}" for (t, p), n in ranked]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  cuda_dev: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    # --- Resolve target old/new indices from the source label_map ---------
    with (REPO_ROOT / "model_results" / "label_map_sen_id_2000.json").open("r", encoding="utf-8") as f:
        sen_label_to_old = json.load(f)
    target_old_idx = [sen_label_to_old[lbl] for lbl in TARGET_LABELS]
    old_to_new = {old: new for new, old in enumerate(target_old_idx)}
    print(f"target old indices: {target_old_idx}")
    print(f"new label order   : {TARGET_LABELS}")

    # --- Load NPZ subsets -------------------------------------------------
    Xtr, ytr = filter_npz_to_targets(TRAIN_NPZ, target_old_idx, old_to_new)
    Xva, yva = filter_npz_to_targets(VAL_NPZ,   target_old_idx, old_to_new)
    Xte, yte = filter_npz_to_targets(TEAM_NPZ,  target_old_idx, old_to_new)
    print(f"train: X{Xtr.shape}  y{ytr.shape}  per-class={np.bincount(ytr, minlength=12).tolist()}")
    print(f"val  : X{Xva.shape}  y{yva.shape}  per-class={np.bincount(yva, minlength=12).tolist()}")
    print(f"team : X{Xte.shape}  y{yte.shape}  per-class={np.bincount(yte, minlength=12).tolist()}")

    train_loader = DataLoader(KeypointDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(KeypointDataset(Xva, yva), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    team_loader  = DataLoader(KeypointDataset(Xte, yte), batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # --- Build fine-tune model -------------------------------------------
    model, new_cfg, src_metrics, src_epoch = build_finetune_model(SRC_CKPT, target_old_idx)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: CNNGRUAttn(num_classes=12)  params={n_params:,}")
    print(f"source ckpt epoch={src_epoch}  metrics={src_metrics}")

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(param_groups(model), weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = []
    best_val_acc = -1.0
    best_state = None
    best_epoch = -1
    epochs_since_best = 0

    print("=" * 70)
    print(f"start fine-tune  epochs={EPOCHS}  batch={BATCH_SIZE}  "
          f"lr(bb)={LR_BACKBONE}  lr(head)={LR_HEAD}")
    print("=" * 70)
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc, _, _ = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step()

        improved = va_acc > best_val_acc
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc,
        })
        marker = " ← best" if improved else ""
        print(f"epoch {epoch:3d}/{EPOCHS} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}{marker}")

        if improved:
            best_val_acc = va_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= EARLY_PATIENCE:
                print(f"early stop: no val improvement for {EARLY_PATIENCE} epochs")
                break

    elapsed = time.time() - t0
    print(f"done in {elapsed:.1f}s  best epoch {best_epoch}  val_acc {best_val_acc:.4f}")

    # --- Restore best weights and evaluate -------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)

    va_loss, va_acc, va_preds, va_truth = run_epoch(model, val_loader,  criterion, optimizer, device, train=False)
    te_loss, te_acc, te_preds, te_truth = run_epoch(model, team_loader, criterion, optimizer, device, train=False)
    val_rows  = per_class_report(va_truth, va_preds, TARGET_LABELS)
    team_rows = per_class_report(te_truth, te_preds, TARGET_LABELS)
    val_confusions  = top_confusions(va_truth, va_preds, TARGET_LABELS)
    team_confusions = top_confusions(te_truth, te_preds, TARGET_LABELS)

    print("\nVAL  per-class:")
    for r in val_rows:
        flag = " *" if r["priority"] else "  "
        print(f"  {flag}{r['label']}  {r['display']:<22}  {r['correct']}/{r['support']}  ({r['acc']*100:5.1f}%)")
    print("\nTEAM (demo) per-class:")
    for r in team_rows:
        flag = " *" if r["priority"] else "  "
        print(f"  {flag}{r['label']}  {r['display']:<22}  {r['correct']}/{r['support']}  ({r['acc']*100:5.1f}%)")
    print(f"\nVAL  overall acc: {va_acc*100:.2f}%  loss: {va_loss:.4f}")
    print(f"TEAM overall acc: {te_acc*100:.2f}%  loss: {te_loss:.4f}")

    # --- Save artifacts ---------------------------------------------------
    OUT_CKPT.parent.mkdir(parents=True, exist_ok=True)
    LBL_MAP.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state": model.state_dict(),
        "model_config": new_cfg,
        "epoch": best_epoch,
        "metrics": {
            "val_loss": va_loss, "val_top1": va_acc,
            "team_loss": te_loss, "team_top1": te_acc,
        },
        "labels": TARGET_LABELS,
        "source_ckpt": str(SRC_CKPT),
        "source_old_indices": target_old_idx,
    }, OUT_CKPT)
    print(f"\nsaved ckpt → {OUT_CKPT}")

    label_to_idx = {lbl: i for i, lbl in enumerate(TARGET_LABELS)}
    with LBL_MAP.open("w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, ensure_ascii=False, indent=2)
    print(f"saved label_map → {LBL_MAP}")

    idx_to_label = {
        str(i): {
            "label": lbl,
            "sen_id": lbl,
            "display": DISPLAY_TEXT.get(lbl, lbl),
            "source_old_idx": target_old_idx[i],
        }
        for i, lbl in enumerate(TARGET_LABELS)
    }
    with IDX_MAP.open("w", encoding="utf-8") as f:
        json.dump(idx_to_label, f, ensure_ascii=False, indent=2)
    print(f"saved idx_to_label → {IDX_MAP}")

    # Curve plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        epochs_ax = [h["epoch"] for h in history]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
        ax1.plot(epochs_ax, [h["train_loss"] for h in history], label="train")
        ax1.plot(epochs_ax, [h["val_loss"]   for h in history], label="val")
        ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend(); ax1.grid(True)
        ax2.plot(epochs_ax, [h["train_acc"] for h in history], label="train")
        ax2.plot(epochs_ax, [h["val_acc"]   for h in history], label="val")
        ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend(); ax2.grid(True)
        plt.suptitle("sentence_scenario12 fine-tune (12-class)")
        plt.tight_layout(); plt.savefig(CURVE_PNG, dpi=130); plt.close()
        print(f"saved curve → {CURVE_PNG}")
    except Exception as exc:
        print(f"curve plot skipped: {exc}")

    # Markdown report
    lines = [
        "# sentence_scenario12 fine-tune report",
        "",
        f"- source ckpt: `{SRC_CKPT.relative_to(REPO_ROOT)}` (epoch {src_epoch})",
        f"- output ckpt: `{OUT_CKPT.relative_to(REPO_ROOT)}`",
        f"- label map  : `{LBL_MAP.relative_to(REPO_ROOT)}` (12 classes)",
        f"- best epoch : {best_epoch} / {EPOCHS}",
        f"- elapsed    : {elapsed:.1f}s",
        f"- device     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
        "",
        "## Overall metrics",
        "",
        "| split | loss | top1 |",
        "|---|---|---|",
        f"| val (held-in) | {va_loss:.4f} | {va_acc*100:.2f}% |",
        f"| team_video (demo proxy) | {te_loss:.4f} | {te_acc*100:.2f}% |",
        "",
        "## Per-class accuracy",
        "",
        "`*` = user-flagged demo priority class.",
        "",
        "| | label | text | val | team (demo) |",
        "|---|---|---|---|---|",
    ]
    for vr, tr in zip(val_rows, team_rows):
        flag = "*" if vr["priority"] else ""
        lines.append(
            f"| {flag} | {vr['label']} | {vr['display']} | "
            f"{vr['correct']}/{vr['support']} ({vr['acc']*100:.1f}%) | "
            f"{tr['correct']}/{tr['support']} ({tr['acc']*100:.1f}%) |"
        )
    lines += ["", "## Top confusions (val)", ""]
    lines += [f"- {line}" for line in val_confusions] or ["- (none)"]
    lines += ["", "## Top confusions (team)", ""]
    lines += [f"- {line}" for line in team_confusions] or ["- (none)"]
    lines += [
        "",
        "## Curves",
        f"![]({CURVE_PNG.name})",
        "",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved report → {REPORT_MD}")


if __name__ == "__main__":
    main()
