"""Personalized model 4: a hybrid of collaborative, content and popularity signals.

    score(u, i) = w_cf * z(implicit-ALS score) + w_genre * z(genre affinity) + w_popularity * z(popularity)

* collaborative - the implicit-ALS model's preference score (``implicit_als.py``);
* content       - cosine between the user's ``user_pref_*`` genre profile and the
                  item's genre vector, both read from the feature store;
* popularity    - the catalogued ``item_popularity_score`` (popular AND liked).

Each signal is z-scored over the catalogue before it is weighted, so a weight
means the same thing whatever the raw scale of the signal.  A user the
collaborative model has never seen still gets a list: their genre profile if
they have one, the popularity prior otherwise - the hybrid has no empty list.

The blend weights are tuned in ``train.py`` on a validation split carved from
TRAIN by the same time rule (features for that split are rebuilt in memory from
the fit part, so neither validation nor test leaks into them).  Like the
implicit model it ranks; it predicts no explicit rating, so no RMSE/MAE.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..transformation.transform import GENRE_COLUMNS
from .base import Recommender, TrainingContext
from .implicit_als import ImplicitALS

PREF_COLUMNS = [f"user_pref_{c[len('genre_'):]}" for c in GENRE_COLUMNS]


def _normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, n, out=np.zeros_like(m), where=n > 0)


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-9)


class HybridRecommender(Recommender):
    name = "hybrid"

    def __init__(self, w_cf: float = 1.0, w_genre: float = 0.25, w_popularity: float = 0.5,
                 cf_params: dict | None = None, seed: int = 42):
        self.w_cf, self.w_genre, self.w_popularity = w_cf, w_genre, w_popularity
        self.cf_params = dict(cf_params or {})
        self.seed = seed

    def fit(self, train: pd.DataFrame, ctx: TrainingContext, base: ImplicitALS | None = None,
            user_features: pd.DataFrame | None = None, item_features: pd.DataFrame | None = None) -> "HybridRecommender":
        """``base`` reuses an already-trained implicit ALS (same train split);
        ``user_features`` / ``item_features`` replace the feature-store tables (tuning only)."""
        self.base = base if base is not None else ImplicitALS(**self.cf_params, seed=self.seed).fit(train, ctx)
        self.item_ids = self.base.item_ids
        uf = user_features if user_features is not None else ctx.store.read("user")
        itf = item_features if item_features is not None else ctx.store.read("item")
        itf = itf.set_index("item_id").reindex(self.item_ids)
        self.G = _normalize(itf[GENRE_COLUMNS].fillna(0).to_numpy(dtype=float))
        self.pop_z = _z(itf["item_popularity_score"].fillna(0.0).to_numpy(dtype=float))
        self.U = _normalize(uf[PREF_COLUMNS].to_numpy(dtype=float))
        self._profile_pos = {int(u): n for n, u in enumerate(uf["user_id"].to_numpy())}
        self._cf_pos = {int(u): n for n, u in enumerate(self.base.user_ids)}
        self._n_users_with_profile = int((np.linalg.norm(self.U, axis=1) > 0).sum())
        self._n_interactions = int(len(train))
        return self

    def set_weights(self, w_cf: float | None = None, w_genre: float | None = None,
                    w_popularity: float | None = None) -> "HybridRecommender":
        if w_cf is not None:
            self.w_cf = w_cf
        if w_genre is not None:
            self.w_genre = w_genre
        if w_popularity is not None:
            self.w_popularity = w_popularity
        return self

    def components(self, user_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The three z-scored signals for one user (zeros where the user is unknown to a signal)."""
        zeros = np.zeros(len(self.item_ids))
        cp = self._cf_pos.get(int(user_id))
        cf = _z(self.base.Y @ self.base.X[cp]) if cp is not None else zeros
        pp = self._profile_pos.get(int(user_id))
        genre = _z(self.G @ self.U[pp]) if pp is not None and np.any(self.U[pp]) else zeros
        return cf, genre, self.pop_z

    def recommend(self, user_id: int, k: int, exclude: set[int]) -> list[int]:
        cf, genre, pop = self.components(user_id)
        return self._top_k(self.w_cf * cf + self.w_genre * genre + self.w_popularity * pop, self.item_ids, k, exclude)

    def params(self) -> dict:
        return {"w_cf": self.w_cf, "w_genre": self.w_genre, "w_popularity": self.w_popularity,
                **{f"cf_{k}": v for k, v in self.base.params().items() if k != "solver"},
                "blend": "z(implicit ALS) + z(genre affinity) + z(popularity)"}

    def train_metrics(self) -> dict:
        return {"interactions_fit": self._n_interactions, "users_with_profile": self._n_users_with_profile}
