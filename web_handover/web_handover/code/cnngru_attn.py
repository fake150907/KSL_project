"""CNN1d + BiGRU + Single-vector Attention Pooling classifier (v7 §C.3).

Input:  (B, T=32, F=225) — body-normalized keypoint sequence.
Output: (B, num_classes) logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalCNN(nn.Module):
    def __init__(self, in_dim: int, channels: list[int], kernel: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        pad = kernel // 2
        for c in channels:
            layers += [
                nn.Conv1d(prev, c, kernel_size=kernel, padding=pad),
                nn.BatchNorm1d(c),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = c
        self.net = nn.Sequential(*layers)
        self.out_dim = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T) for Conv1d -> (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        x = self.net(x)
        x = x.transpose(1, 2)
        return x


class AttentionPooling(nn.Module):
    """Single-vector additive attention over T."""

    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        self.score = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, D) -> weighted sum over T -> (B, D)
        e = self.score(torch.tanh(self.proj(x))).squeeze(-1)  # (B, T)
        if mask is not None:
            e = e.masked_fill(~mask, float("-inf"))
        alpha = F.softmax(e, dim=1).unsqueeze(-1)             # (B, T, 1)
        return (x * alpha).sum(dim=1)                          # (B, D)


class CNNGRUAttn(nn.Module):
    def __init__(
        self,
        input_dim: int = 225,
        cnn_channels: list[int] | tuple[int, ...] = (128, 256),
        cnn_kernel: int = 5,
        cnn_dropout: float = 0.1,
        gru_hidden: int = 256,
        gru_layers: int = 2,
        gru_bidirectional: bool = True,
        gru_dropout: float = 0.2,
        pooling: str = "attention",
        fc_dropout: float = 0.3,
        num_classes: int = 3000,
    ):
        super().__init__()
        self.cnn = TemporalCNN(input_dim, list(cnn_channels), cnn_kernel, cnn_dropout)
        self.gru = nn.GRU(
            input_size=self.cnn.out_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            bidirectional=gru_bidirectional,
        )
        gru_out = gru_hidden * (2 if gru_bidirectional else 1)
        self.pool_mode = pooling
        if pooling == "attention":
            self.pool = AttentionPooling(gru_out)
        elif pooling == "last":
            self.pool = None
        else:
            raise ValueError(f"unknown pooling: {pooling}")
        self.dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(gru_out, num_classes)

        # store config for ckpt round-trip
        self.config = dict(
            input_dim=input_dim,
            cnn_channels=list(cnn_channels),
            cnn_kernel=cnn_kernel,
            cnn_dropout=cnn_dropout,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            gru_bidirectional=gru_bidirectional,
            gru_dropout=gru_dropout,
            pooling=pooling,
            fc_dropout=fc_dropout,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.cnn(x)            # (B, T, C)
        h, _ = self.gru(h)         # (B, T, 2H)
        if self.pool_mode == "attention":
            pooled = self.pool(h)
        else:
            pooled = h[:, -1, :]
        pooled = self.dropout(pooled)
        return self.fc(pooled)


class CNNGRUAttnV2(nn.Module):
    """v2 variant — pack_padded_sequence GRU + AttentionPooling mask.

    사양서: SENTENCE_재전처리_사양서_v2.md §5.2

    v1 CNNGRUAttn과 차이:
      - input_dim default = 300 (mediapipe_xyzc 75 × 4)
      - forward(x, valid_lengths) 시 valid_lengths를 사용해서:
          1) GRU는 pack_padded_sequence/pad_packed 적용해 valid 영역까지만 hidden update
          2) AttentionPooling은 mask로 padding 영역 weight = 0
      - valid_lengths가 None이면 기존 동작 (mask 없음, full T 통과)
    """

    def __init__(
        self,
        input_dim: int = 300,
        cnn_channels: list[int] | tuple[int, ...] = (128, 256),
        cnn_kernel: int = 5,
        cnn_dropout: float = 0.1,
        gru_hidden: int = 256,
        gru_layers: int = 2,
        gru_bidirectional: bool = True,
        gru_dropout: float = 0.2,
        pooling: str = "attention",
        fc_dropout: float = 0.3,
        num_classes: int = 2000,
    ):
        super().__init__()
        self.cnn = TemporalCNN(input_dim, list(cnn_channels), cnn_kernel, cnn_dropout)
        self.gru = nn.GRU(
            input_size=self.cnn.out_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            bidirectional=gru_bidirectional,
        )
        gru_out = gru_hidden * (2 if gru_bidirectional else 1)
        self.pool_mode = pooling
        if pooling == "attention":
            self.pool = AttentionPooling(gru_out)
        elif pooling == "last":
            self.pool = None
        else:
            raise ValueError(f"unknown pooling: {pooling}")
        self.dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(gru_out, num_classes)

        self.config = dict(
            input_dim=input_dim,
            cnn_channels=list(cnn_channels),
            cnn_kernel=cnn_kernel,
            cnn_dropout=cnn_dropout,
            gru_hidden=gru_hidden,
            gru_layers=gru_layers,
            gru_bidirectional=gru_bidirectional,
            gru_dropout=gru_dropout,
            pooling=pooling,
            fc_dropout=fc_dropout,
            num_classes=num_classes,
            variant="v2",
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, input_dim) float
            valid_lengths: (B,) long, 1..T. None이면 mask 없이 full T 처리.
        Returns:
            logits: (B, num_classes)
        """
        h = self.cnn(x)            # (B, T, C)
        b, t_max, _ = h.shape

        if valid_lengths is not None:
            # GRU: pack_padded_sequence (enforce_sorted=False라 sort 불필요)
            # CPU long 텐서 필요 — pack_padded는 lengths를 CPU 텐서로 받음
            lengths_cpu = valid_lengths.detach().to("cpu", dtype=torch.long).clamp(min=1, max=t_max)
            packed = nn.utils.rnn.pack_padded_sequence(
                h, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.gru(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=t_max
            )

            # AttentionPooling mask: True = valid, False = padding
            mask = (
                torch.arange(t_max, device=valid_lengths.device).unsqueeze(0)
                < valid_lengths.unsqueeze(1)
            )
        else:
            h, _ = self.gru(h)
            mask = None

        if self.pool_mode == "attention":
            pooled = self.pool(h, mask=mask)
        else:
            # "last" pooling: valid_lengths가 있으면 valid 끝 index에서 추출
            if valid_lengths is not None:
                idx = (valid_lengths.clamp(min=1) - 1).view(b, 1, 1).expand(b, 1, h.size(-1))
                pooled = h.gather(1, idx).squeeze(1)
            else:
                pooled = h[:, -1, :]

        pooled = self.dropout(pooled)
        return self.fc(pooled)
