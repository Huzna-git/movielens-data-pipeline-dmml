"""Personalized model 2: content-based filtering on item genres.

The item side is the genre one-hot vector from the feature store; the user
side is the ``user_pref_*`` genre profile (share of the user's liked ratings
per genre), also from the feature store - i.e. the model consumes catalogued
features rather than recomputing them.

Ranking:  score(u, i) = cosine(profile_u, genres_i), unseen items only.
Rating:   a genre-kNN estimate over the user's own history:
          r_hat(u, i) = sum_j sim(i, j) r_uj / sum_j sim(i, j)  (j in train(u)),
          falling back to the user's mean when no rated item shares a genre.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import Recommender, TrainingContext


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, n, out=np.zeros_like(m), where=n > 0)


class ContentBasedRecommender(Recommender):
    name = "content_based_genres"

    def __init__(self, like_threshold: float = 4.0):
        self.like_threshold = like_threshold

    def fit(self, train: pd.DataFrame, ctx: TrainingContext) -> "ContentBasedRecommender":
        self.item_ids, G = ctx.store.item_genre_matrix()
        self.G = _normalize(G)
        self.user_ids, Uprof = ctx.store.user_genre_profile()
        self.U = _normalize(Uprof)
        uf = ctx.store.read("user").set_index("user_id")
        self.user_mean = uf["user_mean_rating"].to_dict()
        self.global_mean = float(train["rating"].mean())
        # per-user history (item positions + ratings) for the kNN rating estimate
        ipos = pd.Index(self.item_ids).get_indexer(train["item_id"])
        self.history: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for u, grp in train.assign(_pos=ipos).groupby("user_id"):
            self.history[int(u)] = (grp["_pos"].to_numpy(), grp["rating"].to_numpy(dtype=float))
        self._n_users_with_profile = int((np.linalg.norm(self.U, axis=1) > 0).sum())
        return self

    def recommend(self, user_id: int, k: int, exclude: set[int]) -> list[int]:
        pos = pd.Index(self.user_ids).get_indexer([user_id])[0]
        if pos < 0 or not np.any(self.U[pos]):
            return []  # no profile (cold-start or nothing liked): the content model has no opinion
        scores = self.G @ self.U[pos]
        return self._top_k(scores, self.item_ids, k, exclude)

    def predict_rating(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        ipos = pd.Index(self.item_ids).get_indexer(item_ids)
        out = np.empty(len(user_ids))
        for n, (u, ip) in enumerate(zip(user_ids, ipos)):
            fallback = self.user_mean.get(int(u), np.nan)
            fallback = self.global_mean if np.isnan(fallback) else fallback
            hist = self.history.get(int(u))
            if hist is None or ip < 0:
                out[n] = fallback
                continue
            hpos, hr = hist
            sims = self.G[hpos] @ self.G[ip]
            out[n] = (sims @ hr) / sims.sum() if sims.sum() > 0 else fallback
        return np.clip(out, 1.0, 5.0)

    def params(self) -> dict:
        return {"item_representation": "genre one-hot (feature store)", "user_representation": "user_pref_* genre profile (feature store)",
                "like_threshold": self.like_threshold, "similarity": "cosine"}

    def train_metrics(self) -> dict:
        return {"users_with_profile": self._n_users_with_profile, "n_genres": int(self.G.shape[1])}
