"""Role: sklearn baseline classifiers for fixed-length sign tensors.

Input: X shaped [samples, frames, features]
Output: fitted sklearn model
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def flatten_sequences(X):
    return X.reshape(X.shape[0], -1)


def make_baseline(name: str = "random_forest", random_seed: int = 42):
    if name == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(128,), max_iter=80, random_state=random_seed)),
            ]
        )
    return RandomForestClassifier(n_estimators=80, max_depth=12, random_state=random_seed, class_weight="balanced")
