"""CNN-GRU with attention pooling for keypoint sequence classification.

Reconstructed to match the FEARNA v7 checkpoint format expected by
handover_web_realz03_20260520 (word_stage2.pt / sentence_scenario12.pt).
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class _Conv1dStack(nn.Module):
    """Stack of Conv1d → BatchNorm1d → ReLU → Dropout blocks."""

    def __init__(
        self,
        input_dim: int,
        channels: Iterable[int],
        kernel_size: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_ch = input_dim
        for out_ch in channels:
            layers.append(
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
            )
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            in_ch = out_ch
        self.net = nn.Sequential(*layers)
        self.out_channels = in_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) → (B, F, T) for Conv1d
        x = x.transpose(1, 2)
        x = self.net(x)
        # → (B, T, C) for downstream RNN
        return x.transpose(1, 2)


class _AttentionPool(nn.Module):
    """Additive attention pooling over the time dimension."""

    def __init__(self, feature_dim: int, attn_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(feature_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        a = torch.tanh(self.proj(x))      # (B, T, attn_dim)
        s = self.score(a)                  # (B, T, 1)
        w = torch.softmax(s, dim=1)        # over time
        pooled = (x * w).sum(dim=1)        # (B, F)
        return pooled


class CNNGRUAttn(nn.Module):
    """Conv1d front-end + BiGRU + attention pool + FC classifier."""

    def __init__(
        self,
        input_dim: int = 225,
        cnn_channels: Iterable[int] = (128, 256),
        cnn_kernel: int = 5,
        cnn_dropout: float = 0.2,
        gru_hidden: int = 256,
        gru_layers: int = 2,
        gru_bidirectional: bool = True,
        gru_dropout: float = 0.3,
        pooling: str = "attention",
        fc_dropout: float = 0.5,
        num_classes: int = 12,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.cnn = _Conv1dStack(
            input_dim=self.input_dim,
            channels=list(cnn_channels),
            kernel_size=int(cnn_kernel),
            dropout=float(cnn_dropout),
        )
        rnn_input = self.cnn.out_channels
        self.gru = nn.GRU(
            input_size=rnn_input,
            hidden_size=int(gru_hidden),
            num_layers=int(gru_layers),
            batch_first=True,
            dropout=float(gru_dropout) if int(gru_layers) > 1 else 0.0,
            bidirectional=bool(gru_bidirectional),
        )
        feature_dim = int(gru_hidden) * (2 if gru_bidirectional else 1)
        if str(pooling).lower() == "attention":
            self.pool = _AttentionPool(feature_dim=feature_dim, attn_dim=128)
            self._pool_kind = "attention"
        elif str(pooling).lower() == "mean":
            self.pool = None
            self._pool_kind = "mean"
        else:
            self.pool = None
            self._pool_kind = "last"
        self.dropout = nn.Dropout(p=float(fc_dropout))
        self.fc = nn.Linear(feature_dim, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        x = self.cnn(x)
        x, _ = self.gru(x)
        if self._pool_kind == "attention":
            pooled = self.pool(x)
        elif self._pool_kind == "mean":
            pooled = x.mean(dim=1)
        else:
            pooled = x[:, -1, :]
        pooled = self.dropout(pooled)
        return self.fc(pooled)
