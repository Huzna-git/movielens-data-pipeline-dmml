"""Personalized model 2: implicit-feedback ALS (Hu, Koren & Volinsky 2008).

Top-K recommendation is a ranking problem, and a rating is also a signal
that the user *chose* to watch.  This model treats every train interaction
as a positive observation with confidence ``1 + alpha`` and every missing
pair as a weak zero, and factorizes the resulting preference matrix with
alternating least squares (one small ``f x f`` solve per user / per item,
the YtY trick keeps each epoch linear in the number of interactions).  It
does not predict explicit ratings, so it reports no RMSE/MAE.

``tune()`` runs a small grid on a validation split carved from the train set
by the same time-based rule, so the chosen hyper-parameters never see test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .base import Recommender, TrainingContext


class ImplicitALS(Recommender):
    name = "implicit_als"

    def __init__(self, n_factors: int = 40, n_epochs: int = 12, regularization: float = 0.1, alpha: float = 5.0, seed: int = 42):
        self.n_factors, self.n_epochs, self.reg, self.alpha, self.seed = n_factors, n_epochs, regularization, alpha, seed

    def fit(self, train: pd.DataFrame, ctx: TrainingContext) -> "ImplicitALS":
        rng = np.random.default_rng(self.seed)
        self.user_ids = np.sort(train["user_id"].unique())
        self.item_ids = ctx.item_ids
        u = pd.Index(self.user_ids).get_indexer(train["user_id"])
        i = pd.Index(self.item_ids).get_indexer(train["item_id"])
        C = sp.csr_matrix((np.full(len(u), float(self.alpha)), (u, i)), shape=(len(self.user_ids), len(self.item_ids)))
        Ct = C.T.tocsr()
        X = rng.normal(0, 0.01, (C.shape[0], self.n_factors))
        Y = rng.normal(0, 0.01, (C.shape[1], self.n_factors))
        for _ in range(self.n_epochs):
            self._step(C, X, Y)
            self._step(Ct, Y, X)
        self.X, self.Y = X, Y
        self._n_interactions = int(C.nnz)
        return self

    def _step(self, C: sp.csr_matrix, X: np.ndarray, Y: np.ndarray) -> None:
        """Solve every row of X given Y:  (YtY + Yt (C_u - I) Y + reg I) x = Yt C_u p_u."""
        YtY = Y.T @ Y
        I = np.eye(self.n_factors) * self.reg
        for row in range(C.shape[0]):
            s, e = C.indptr[row], C.indptr[row + 1]
            if s == e:
                X[row] = 0.0
                continue
            cols, conf = C.indices[s:e], C.data[s:e]
            Yr = Y[cols]
            A = YtY + (Yr.T * conf) @ Yr + I
            b = (Yr.T * (1.0 + conf)).sum(axis=1)
            X[row] = np.linalg.solve(A, b)

    def recommend(self, user_id: int, k: int, exclude: set[int]) -> list[int]:
        pos = pd.Index(self.user_ids).get_indexer([user_id])[0]
        if pos < 0:
            return []  # cold-start user: nothing learned for them
        return self._top_k(self.Y @ self.X[pos], self.item_ids, k, exclude)

    def params(self) -> dict:
        return {"n_factors": self.n_factors, "n_epochs": self.n_epochs, "regularization": self.reg, "alpha": self.alpha,
                "seed": self.seed, "solver": "implicit ALS (confidence-weighted, YtY trick)"}

    def train_metrics(self) -> dict:
        return {"interactions_fit": self._n_interactions}
