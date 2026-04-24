"""Role: load YAML config and merge simple CLI overrides.

Input: config/default.yaml
Output: nested dict
Example: from src.utils.config import load_config
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config/default.yaml") -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def apply_cli_overrides(config: dict[str, Any], args: Any) -> dict[str, Any]:
    updates: dict[str, Any] = {"data": {}, "train": {}}
    for name in ("quick_test", "max_classes", "max_samples_per_class", "sequence_length"):
        value = getattr(args, name, None)
        if value is not None:
            updates["data"][name] = value
    for name in (
        "epochs",
        "batch_size",
        "learning_rate",
        "hidden_size",
        "num_layers",
        "dropout",
        "rnn_type",
        "model_type",
        "conv_channels",
        "num_heads",
    ):
        value = getattr(args, name, None)
        if value is not None:
            updates["train"][name] = value
    return deep_update(config, updates)
