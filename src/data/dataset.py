"""Role: PyTorch dataset wrappers for preprocessed NPZ tensors.

Input: data/processed/sign_word_subset.npz
Output: SignDataset and train/validation DataLoaders
Example:
  loaders = make_loaders("data/processed/sign_word_subset.npz", batch_size=16)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # baseline inference can still use load_npz without torch
    torch = None
    DataLoader = None

    class Dataset:  # type: ignore[override]
        pass


class SignDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if torch is None:
            raise ImportError("torch is required to build SignDataset.")
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


@dataclass
class LoadedData:
    X: np.ndarray
    y: np.ndarray
    splits: np.ndarray
    labels: list[str]


def load_npz(path: str) -> LoadedData:
    data = np.load(path, allow_pickle=True)
    return LoadedData(
        X=data["X"],
        y=data["y"],
        splits=data["splits"],
        labels=[str(x) for x in data["labels"].tolist()],
    )


def make_loaders(path: str, batch_size: int = 16) -> tuple[DataLoader, DataLoader, LoadedData]:
    if torch is None or DataLoader is None:
        raise ImportError("torch is required to create DataLoaders.")
    data = load_npz(path)
    train_mask = data.splits == "train"
    val_mask = data.splits != "train"
    if not np.any(val_mask):
        val_mask = ~train_mask
    train_loader = DataLoader(SignDataset(data.X[train_mask], data.y[train_mask]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SignDataset(data.X[val_mask], data.y[val_mask]), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, data
