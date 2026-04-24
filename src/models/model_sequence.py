"""Role: sequence classifiers for final MVP training."""

from __future__ import annotations

import torch
from torch import nn


class SignSequenceClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        rnn_type: str = "gru",
    ):
        super().__init__()
        self.input_size = input_size
        rnn_cls = nn.LSTM if rnn_type.lower() == "lstm" else nn.GRU
        self.rnn_type = rnn_type.lower()
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, hidden = self.rnn(x)
        if self.rnn_type == "lstm":
            hidden = hidden[0]
        last = hidden[-1]
        return self.classifier(last)


class CnnGruSequenceClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
        conv_channels: int = 128,
    ):
        super().__init__()
        self.input_size = input_size
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_size, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
        )
        self.rnn = nn.GRU(
            input_size=conv_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x.transpose(1, 2)).transpose(1, 2)
        _, hidden = self.rnn(features)
        return self.classifier(hidden[-1])


class BiGruAttentionSequenceClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        context_size = hidden_size * 2
        self.attention = nn.Sequential(
            nn.Linear(context_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(context_size),
            nn.Dropout(dropout),
            nn.Linear(context_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        weights = torch.softmax(self.attention(output), dim=1)
        context = torch.sum(output * weights, dim=1)
        return self.classifier(context)


class TransformerSequenceClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_heads: int = 4,
        max_length: int = 128,
    ):
        super().__init__()
        self.input_size = input_size
        self.input_projection = nn.Linear(input_size, hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.position = nn.Parameter(torch.zeros(1, max_length + 1, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.input_projection(x)
        cls = self.cls_token.expand(projected.size(0), -1, -1)
        tokens = torch.cat([cls, projected], dim=1)
        tokens = tokens + self.position[:, : tokens.size(1), :]
        encoded = self.encoder(tokens)
        return self.classifier(encoded[:, 0])


def build_sequence_model(
    input_size: int,
    num_classes: int,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.1,
    rnn_type: str = "gru",
    model_type: str = "rnn",
    conv_channels: int = 128,
    num_heads: int = 4,
    sequence_length: int = 32,
) -> nn.Module:
    if model_type == "cnn_gru":
        return CnnGruSequenceClassifier(
            input_size=input_size,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            conv_channels=conv_channels,
        )
    if model_type == "bigru_attention":
        return BiGruAttentionSequenceClassifier(
            input_size=input_size,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        )
    if model_type == "transformer":
        return TransformerSequenceClassifier(
            input_size=input_size,
            num_classes=num_classes,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_heads=num_heads,
            max_length=sequence_length,
        )
    return SignSequenceClassifier(
        input_size=input_size,
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        rnn_type=rnn_type,
    )
