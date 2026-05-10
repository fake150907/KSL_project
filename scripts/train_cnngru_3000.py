"""
train_cnngru_3000.py  ── 수정버전 (Conv1d 적용)

수정 내용:
  - Conv2d → Conv1d 로 변경 (gru_input_size 14400 → 64으로 축소)
  - forward 흐름 정리: (batch,32,225) → Conv1d → GRU → FC

MediaPipe NPZ로 CNN-GRU 모델 학습 (3000 WORD ID 기준)
입력: data/processed/mediapipe_npz_FILE01_FILE12_ALL.npz
출력: outputs/checkpoints/best_cnngru_3000.pt
      outputs/training_report_cnngru_3000.md
      outputs/loss_accuracy_curve_cnngru.png
      outputs/confusion_matrix_cnngru.png

실행:
  conda activate myenv310
  cd C:\python_study\project_2team
  python train_cnngru_3000.py
"""

import os
import json
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# =====================
# 경로 설정
# =====================
NPZ_PATH     = r"C:\python_study\project_2team\data\processed\mediapipe_npz_FILE01_FILE12_ALL.npz"
IDX_TO_LABEL = r"C:\python_study\project_2team\WORD_3000_CLASS_UPDATE_20260506\idx_to_label_3000.json"
OUTPUT_DIR   = r"C:\python_study\KSL_project_minseok\outputs"
CKPT_DIR     = os.path.join(OUTPUT_DIR, "checkpoints")
REPORT_PATH  = os.path.join(OUTPUT_DIR, "training_report_cnngru_3000.md")
CURVE_PATH   = os.path.join(OUTPUT_DIR, "loss_accuracy_curve_cnngru.png")
CM_PATH      = os.path.join(OUTPUT_DIR, "confusion_matrix_cnngru.png")

# =====================
# 하이퍼파라미터
# =====================
MODEL_NAME      = "CNN-GRU"
SEQUENCE_LENGTH = 32
FEATURE_DIM     = 225
NUM_CLASSES     = 3000
BATCH_SIZE      = 32
EPOCHS          = 300
LEARNING_RATE   = 3e-4
VAL_RATIO       = 0.2
RANDOM_SEED     = 42

# CNN 파라미터
CNN_OUT_CHANNELS = 64
CNN_KERNEL_SIZE  = 3

# GRU 파라미터
GRU_HIDDEN_SIZE = 128
GRU_NUM_LAYERS  = 2
GRU_DROPOUT     = 0.3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
class SignDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ──────────────────────────────────────────────
# CNN-GRU 모델 (수정버전)
# ──────────────────────────────────────────────
class CnnGruModel(nn.Module):
    """
    입력: (batch, seq_len=32, feature=225)

    [수정 전] Conv2d → gru_input_size = 64 * 225 = 14,400  ← 너무 커서 수렴 안 됨
    [수정 후] Conv1d → gru_input_size = 64                  ← 정상 크기

    흐름:
      x: (batch, 32, 225)
      permute → (batch, 225, 32)   Conv1d는 (batch, channel, length) 형식
      Conv1d  → (batch, 64, 32)
      permute → (batch, 32, 64)    GRU는 (batch, seq, feature) 형식
      GRU     → (batch, 32, 256)   양방향이라 hidden*2=256
      last    → (batch, 256)
      FC      → (batch, 3000)
    """
    def __init__(self):
        super().__init__()

        # ── Conv1d: 시간축 방향 특징 추출 ──
        self.cnn = nn.Sequential(
            nn.Conv1d(FEATURE_DIM, CNN_OUT_CHANNELS,
                      kernel_size=CNN_KERNEL_SIZE,
                      padding=CNN_KERNEL_SIZE // 2),   # padding=1 → 길이 유지
            nn.BatchNorm1d(CNN_OUT_CHANNELS),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # GRU input_size = CNN_OUT_CHANNELS = 64
        self.gru = nn.GRU(
            input_size=CNN_OUT_CHANNELS,
            hidden_size=GRU_HIDDEN_SIZE,
            num_layers=GRU_NUM_LAYERS,
            batch_first=True,
            dropout=GRU_DROPOUT if GRU_NUM_LAYERS > 1 else 0,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(GRU_HIDDEN_SIZE * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        # x: (batch, 32, 225)
        x = x.permute(0, 2, 1)       # → (batch, 225, 32)
        x = self.cnn(x)               # → (batch, 64, 32)
        x = x.permute(0, 2, 1)       # → (batch, 32, 64)
        x, _ = self.gru(x)            # → (batch, 32, 256)
        x = x[:, -1, :]              # → (batch, 256)  마지막 타임스텝
        return self.classifier(x)     # → (batch, 3000)


# ──────────────────────────────────────────────
# 학습/평가 함수
# ──────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, epoch, epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch:3d}/{epochs} [Train]", leave=False, ncols=100)
    for X_batch, y_batch in pbar:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (output.argmax(dim=1) == y_batch).sum().item()
        total += len(y_batch)
        pbar.set_postfix(loss=f"{total_loss/len(loader):.4f}", acc=f"{correct/total:.4f}")
    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, criterion, epoch=0, epochs=0, mode="Val"):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_preds, all_labels = [], []
    desc = f"Epoch {epoch:3d}/{epochs} [ {mode} ]" if epoch > 0 else f"[{mode}]"
    pbar = tqdm(loader, desc=desc, leave=False, ncols=100)
    with torch.no_grad():
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            output = model(X_batch)
            loss = criterion(output, y_batch)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == y_batch).sum().item()
            total += len(y_batch)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
            pbar.set_postfix(loss=f"{total_loss/len(loader):.4f}", acc=f"{correct/total:.4f}")
    return total_loss / len(loader), correct / total, all_preds, all_labels


# ──────────────────────────────────────────────
# 그래프 저장 함수
# ──────────────────────────────────────────────
def save_curve(history, path):
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, [h["train_loss"] for h in history], label="Train Loss", color="steelblue")
    ax1.plot(epochs, [h["val_loss"]   for h in history], label="Val Loss",   color="tomato")
    ax1.set_title("Loss Curve")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs, [h["train_acc"] for h in history], label="Train Acc", color="steelblue")
    ax2.plot(epochs, [h["val_acc"]   for h in history], label="Val Acc",   color="tomato")
    ax2.set_title("Accuracy Curve")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.suptitle(f"{MODEL_NAME} - Loss & Accuracy (과적합 확인)", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  학습 곡선 저장 → {path}")


def save_confusion_matrix(cm, labels, path, top_n=30):
    class_counts = cm.sum(axis=1)
    top_idx = np.argsort(class_counts)[-top_n:][::-1]
    cm_top = cm[np.ix_(top_idx, top_idx)]
    top_labels = [labels[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm_top, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(top_labels)))
    ax.set_yticks(range(len(top_labels)))
    ax.set_xticklabels(top_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(top_labels, fontsize=7)
    ax.set_title(f"Confusion Matrix (상위 {top_n}개 클래스)", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  혼동행렬 저장 → {path}")


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    start_time = datetime.datetime.now()
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("=" * 60)
    print(f"{MODEL_NAME} 학습 시작 (3000 WORD ID 기준) [수정버전 Conv1d]")
    print(f"Device: {DEVICE}")
    print(f"시작 시각: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. 데이터 로드
    print("\n[1단계] 데이터 로드...")
    data = np.load(NPZ_PATH, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)
    labels_arr = [str(l) for l in data["labels"].tolist()]
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  labels 수: {len(labels_arr)}")

    with open(IDX_TO_LABEL, "r", encoding="utf-8-sig") as f:
        idx_to_label = json.load(f)

    # 2. train/val 분리
    print(f"\n[2단계] train/val 분리 (val={VAL_RATIO})...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_RATIO, random_state=RANDOM_SEED, stratify=y
    )
    print(f"  train: {len(X_train)}개 / val: {len(X_val)}개")

    train_loader = DataLoader(SignDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(SignDataset(X_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 3. 모델 초기화
    print("\n[3단계] 모델 초기화...")
    model     = CnnGruModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  총 파라미터: {total_params:,}개")
    print(f"  ※ GRU input_size = {CNN_OUT_CHANNELS} (수정 전 14,400 → 수정 후 {CNN_OUT_CHANNELS})")

    # 체크포인트 이어받기
    RESUME_CKPT = os.path.join(CKPT_DIR, "best_cnngru_3000.pt")
    START_EPOCH = 1
    best_val_acc_init = 0.0

    if os.path.exists(RESUME_CKPT):
        ckpt_resume = torch.load(RESUME_CKPT, map_location=DEVICE)
        model.load_state_dict(ckpt_resume["model_state_dict"])
        START_EPOCH = ckpt_resume["epoch"] + 1
        best_val_acc_init = ckpt_resume["val_acc"]
        print(f"  ✅ 체크포인트 로드 완료 → epoch {ckpt_resume['epoch']} (val_acc: {best_val_acc_init:.4f}) 부터 이어서")
    else:
        print(f"  ※ 체크포인트 없음 → epoch 1 부터 시작")

    # 4. 학습
    print(f"\n[4단계] 학습 ({EPOCHS} epochs)...")
    best_val_acc  = best_val_acc_init
    best_val_loss = float("inf")
    best_epoch    = START_EPOCH - 1
    best_ckpt     = os.path.join(CKPT_DIR, "best_cnngru_3000.pt")
    history       = []

    for epoch in range(START_EPOCH, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, epoch, EPOCHS)
        val_loss, val_acc, val_preds, val_labels = eval_epoch(model, val_loader, criterion, epoch, EPOCHS)
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
              f"val loss: {val_loss:.4f} acc: {val_acc:.4f}"
              + (" ← best" if val_acc > best_val_acc else ""))

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
                "num_classes": NUM_CLASSES,
                "feature_dim": FEATURE_DIM,
                "sequence_length": SEQUENCE_LENGTH,
                "layout": "mediapipe_xyz",
            }, best_ckpt)

    print(f"\n  best val_acc: {best_val_acc:.4f} (epoch {best_epoch})")

    # 5. 최종 평가
    print("\n[5단계] 최종 평가...")
    ckpt = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    _, final_acc, final_preds, final_labels_list = eval_epoch(model, val_loader, criterion, mode="Final")

    label_names = [idx_to_label.get(str(i), {}).get("label", f"WORD{i+1:04d}") for i in range(NUM_CLASSES)]
    present_classes = sorted(set(final_labels_list))
    present_names   = [label_names[i] for i in present_classes]

    report_str = classification_report(
        final_labels_list, final_preds,
        labels=present_classes,
        target_names=present_names,
        zero_division=0
    )
    cm = confusion_matrix(final_labels_list, final_preds, labels=present_classes)

    print(f"\n  Test Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")

    # 6. 그래프 저장
    print("\n[6단계] 그래프 저장...")
    save_curve(history, CURVE_PATH)
    save_confusion_matrix(cm, present_names, CM_PATH, top_n=min(30, len(present_classes)))

    # 7. 리포트 저장
    end_time = datetime.datetime.now()
    elapsed  = (end_time - start_time).total_seconds()

    report_lines = [
        "# 학습 결과 리포트",
        "",
        f"| 항목 | 값 |",
        f"|---|---|",
        f"| 학습모델명 | {MODEL_NAME} |",
        f"| Test Accuracy | {final_acc:.4f} ({final_acc*100:.2f}%) |",
        f"| 최종 Val Loss | {best_val_loss:.4f} |",
        f"| 최종 학습 단어 수 | {NUM_CLASSES}개 |",
        f"| 총 학습 Epoch | {EPOCHS} (best: {best_epoch}) |",
        f"| 총 파라미터 | {total_params:,}개 |",
        f"| 소요 시간 | {elapsed:.1f}초 ({elapsed/3600:.1f}시간) |",
        f"| Device | {DEVICE} |",
        "",
        "## 학습 곡선",
        f"![Loss & Accuracy]({CURVE_PATH})",
        "",
        "## 혼동행렬",
        f"![Confusion Matrix]({CM_PATH})",
        "",
        "## Classification Report",
        "```",
        report_str,
        "```",
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\n  리포트 저장 → {REPORT_PATH}")

    print("\n" + "=" * 60)
    print(f"학습 완료!")
    print(f"  모델명: {MODEL_NAME}")
    print(f"  Test Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
    print(f"  최종 Val Loss: {best_val_loss:.4f}")
    print(f"  학습 단어 수: {NUM_CLASSES}개")
    print(f"  총 Epoch: {EPOCHS} (best: {best_epoch})")
    print("=" * 60)


if __name__ == "__main__":
    main()
