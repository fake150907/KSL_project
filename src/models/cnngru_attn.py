from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock1d(nn.Module):
    def __init__(
        self,
        input_dim: int,
        channels: list[int],
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_channels = input_dim
        padding = kernel_size // 2
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class AttentionPool(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = self.score(torch.tanh(self.proj(x)))
        weights = torch.softmax(scores, dim=1)
        return torch.sum(x * weights, dim=1)


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
    ) -> None:
        super().__init__()
        self.input_size = input_dim
        self.input_dim = input_dim
        self.pooling = pooling
        channels = list(cnn_channels)
        if not channels:
            channels = [input_dim]

        self.cnn = ConvBlock1d(input_dim, channels, kernel_size=cnn_kernel, dropout=cnn_dropout)
        gru_output_dim = gru_hidden * (2 if gru_bidirectional else 1)
        self.gru = nn.GRU(
            input_size=channels[-1],
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            bidirectional=gru_bidirectional,
        )
        self.pool = AttentionPool(gru_output_dim, hidden_dim=128)
        self.dropout = nn.Dropout(fc_dropout)
        self.fc = nn.Linear(gru_output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        output, _ = self.gru(features)
        if self.pooling == "mean":
            pooled = output.mean(dim=1)
        elif self.pooling == "last":
            pooled = output[:, -1, :]
        else:
            pooled = self.pool(output)
        return self.fc(self.dropout(pooled))
