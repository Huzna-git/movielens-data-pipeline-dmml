"""Baseline: a non-personalized popularity recommender.

Every user gets the same ranking (minus the items they have already seen).
Score = log1p(n_ratings) * mean_rating, i.e. popular AND liked.  As a rating
predictor it answers the item's mean rating (global mean for unseen items),
which is the classic "item average" baseline for RMSE/MAE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Recommender, TrainingContext


class PopularityRecommender(Recommender):
    name = "popularity"

    def __init__(self, min_ratings: int = 1):
        self.min_ratings = min_ratings

    def fit(self, train: pd.DataFrame, ctx: TrainingContext) -> "PopularityRecommender":
        g = train.groupby("item_id")["rating"].agg(["size", "mean"])
        g = g[g["size"] >= self.min_ratings]
        self.item_ids = ctx.item_ids
        self.global_mean = float(train["rating"].mean())
        idx = pd.Index(self.item_ids)
        n = g["size"].reindex(idx).fillna(0).to_numpy(dtype=float)
        mean = g["mean"].reindex(idx).to_numpy(dtype=float)
        self.item_mean = np.where(np.isnan(mean), self.global_mean, mean)
        self.scores = np.log1p(n) * np.nan_to_num(mean, nan=0.0)
        self._n_items_scored = int((n > 0).sum())
        return self

    def recommend(self, user_id: int, k: int, exclude: set[int]) -> list[int]:
        return self._top_k(self.scores, self.item_ids, k, exclude)

    def predict_rating(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        pos = pd.Index(self.item_ids).get_indexer(item_ids)
        out = np.full(len(item_ids), self.global_mean)
        ok = pos >= 0
        out[ok] = self.item_mean[pos[ok]]
        return out

    def params(self) -> dict:
        return {"min_ratings": self.min_ratings, "score": "log1p(n_ratings) * mean_rating"}

    def train_metrics(self) -> dict:
        return {"items_scored": self._n_items_scored, "global_mean_rating": round(self.global_mean, 4)}
