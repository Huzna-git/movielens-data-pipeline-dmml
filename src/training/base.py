"""The recommender interface every model implements, plus the training context."""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.store import FeatureStore


@dataclass
class TrainingContext:
    """Everything a model may need besides the training interactions."""
    items: pd.DataFrame
    item_genres: pd.DataFrame
    store: FeatureStore
    cfg: dict
    seed: int = 42

    @property
    def item_ids(self) -> np.ndarray:
        return np.sort(self.items["item_id"].unique())


class Recommender(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, train: pd.DataFrame, ctx: TrainingContext) -> "Recommender": ...

    @abstractmethod
    def recommend(self, user_id: int, k: int, exclude: set[int]) -> list[int]:
        """Top-k item ids for a user, never containing ``exclude`` (seen items)."""

    def predict_rating(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray | None:
        """Predicted explicit ratings, or None when the model does not predict ratings."""
        return None

    def params(self) -> dict:
        return {}

    def train_metrics(self) -> dict:
        return {}

    # ---- persistence --------------------------------------------------------
    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)
        return path

    @staticmethod
    def load(path: Path) -> "Recommender":
        with open(path, "rb") as fh:
            return pickle.load(fh)

    # ---- helpers ------------------------------------------------------------
    @staticmethod
    def _top_k(scores: np.ndarray, item_ids: np.ndarray, k: int, exclude: set[int]) -> list[int]:
        s = scores.copy()
        if exclude:
            mask = np.isin(item_ids, list(exclude))
            s[mask] = -np.inf
        k = min(k, int(np.isfinite(s).sum()))
        if k <= 0:
            return []
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top], kind="stable")]
        return item_ids[top].tolist()
