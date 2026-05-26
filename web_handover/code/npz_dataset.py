"""NPZ-backed dataset for KSL keypoint sequences.

NPZ schema (read-only reference, see project README):
- X:          (N, 32, 225) float32 — body-normalized keypoint sequences
- y:          (N,)         int64   — class index in 3000/2000-class head
- splits:     (N,)         str     — train/validation/team tag
- sample_ids: (N,)         str     — e.g. NIA_SL_WORD0001_REAL01_D, WORD0579_REALZ01_part01_REALZ01
- labels:     (C,)         str     — label_id table (WORD0001 .. or SEN0001 ..)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


_SIGNER_RE = re.compile(r"(REALZ\d+|REAL\d+)")


def extract_signer(sample_id: str) -> str:
    m = _SIGNER_RE.search(sample_id)
    if m is None:
        raise ValueError(f"signer id not found in sample_id: {sample_id}")
    return m.group(1)


class NPZKeypointDataset(Dataset):
    """Wraps an NPZ file. Loads X into memory (float32, ~225*32*4 = 28.8 KB/sample)."""

    def __init__(
        self,
        npz_path: str | Path,
        indices: np.ndarray | None = None,
        transform=None,
    ):
        self.npz_path = Path(npz_path)
        data = np.load(self.npz_path, allow_pickle=True)
        self.X = data["X"].astype(np.float32, copy=False)
        self.y = data["y"].astype(np.int64, copy=False)
        self.sample_ids = np.asarray(data["sample_ids"])
        self.labels = np.asarray(data["labels"])

        self.indices = (
            np.arange(len(self.X), dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.transform = transform

    @property
    def num_classes(self) -> int:
        return int(self.labels.shape[0])

    def signer_of(self, dataset_index: int) -> str:
        return extract_signer(str(self.sample_ids[dataset_index]))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x = self.X[idx]
        y = int(self.y[idx])
        sample_id = str(self.sample_ids[idx])
        if self.transform is not None:
            x = self.transform(x, sample_id=sample_id, label_id=str(self.labels[y]))
        x_tensor = torch.from_numpy(np.ascontiguousarray(x))
        return x_tensor, y, idx


def split_indices_by_signer(
    sample_ids: Iterable[str],
    holdout_signers: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (keep_indices, holdout_indices)."""
    sample_ids = np.asarray(list(sample_ids))
    holdout_set = set(holdout_signers)
    signers = np.asarray([extract_signer(str(s)) for s in sample_ids])
    holdout_mask = np.isin(signers, list(holdout_set))
    keep_idx = np.where(~holdout_mask)[0]
    holdout_idx = np.where(holdout_mask)[0]
    return keep_idx, holdout_idx


class NPZKeypointDatasetV2(Dataset):
    """SENTENCE v2 NPZ schema dataset.

    사양서: claude_advisor/SENTENCE_재전처리_사양서_v2.md §5.2.2, §7

    v1 NPZKeypointDataset과의 차이:
      - X shape (N, 128, 300) — T_max=128, mediapipe_xyzc 75×4
      - valid_lengths (N,) int32 — 실제 frame 수, mask 생성용
      - __getitem__이 (x, y, valid_length, idx) 반환

    기존 v1 NPZKeypointDataset은 그대로 보존 (v1/WORD 학습 호환).
    """

    def __init__(
        self,
        npz_path: str | Path,
        indices: np.ndarray | None = None,
        transform=None,
    ):
        self.npz_path = Path(npz_path)
        data = np.load(self.npz_path, allow_pickle=True)
        self.X = data["X"].astype(np.float32, copy=False)
        self.y = data["y"].astype(np.int64, copy=False)
        if "valid_lengths" not in data.files:
            raise ValueError(
                f"NPZ missing 'valid_lengths' — v2 schema 아닐 가능성. file={self.npz_path}"
            )
        self.valid_lengths = data["valid_lengths"].astype(np.int64, copy=False)
        self.sample_ids = np.asarray(data["sample_ids"])
        self.labels = np.asarray(data["labels"])

        # v2 schema assert
        if self.X.ndim != 3:
            raise ValueError(f"X must be 3D, got shape {self.X.shape}")
        n, t_max, feat = self.X.shape
        if feat != 300:
            raise ValueError(
                f"v2 expects feature_count=300 (xyzc), got {feat}. file={self.npz_path}"
            )
        if len(self.valid_lengths) != n:
            raise ValueError(
                f"valid_lengths length mismatch: {len(self.valid_lengths)} vs {n}"
            )

        self.t_max = int(t_max)
        self.indices = (
            np.arange(n, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        self.transform = transform

    @property
    def num_classes(self) -> int:
        return int(self.labels.shape[0])

    def signer_of(self, dataset_index: int) -> str:
        return extract_signer(str(self.sample_ids[dataset_index]))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        x = self.X[idx]
        y = int(self.y[idx])
        vl = int(self.valid_lengths[idx])
        sample_id = str(self.sample_ids[idx])
        if self.transform is not None:
            x = self.transform(x, sample_id=sample_id, label_id=str(self.labels[y]))
        x_tensor = torch.from_numpy(np.ascontiguousarray(x))
        return x_tensor, y, vl, idx


def stratified_internal_val_split(
    y: np.ndarray,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-class stratified split for Stage 1 internal-val (early-stop trigger).

    Note on rare classes: classes with <2 samples go entirely to train (no val sample).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    train_idx_parts: list[np.ndarray] = []
    val_idx_parts: list[np.ndarray] = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_val = int(round(len(cls_idx) * val_ratio))
        if len(cls_idx) >= 2 and n_val == 0:
            n_val = 1
        val_idx_parts.append(cls_idx[:n_val])
        train_idx_parts.append(cls_idx[n_val:])
    train_idx = np.concatenate(train_idx_parts) if train_idx_parts else np.empty((0,), dtype=np.int64)
    val_idx = np.concatenate(val_idx_parts) if val_idx_parts else np.empty((0,), dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx.astype(np.int64), val_idx.astype(np.int64)
