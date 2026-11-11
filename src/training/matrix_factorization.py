"""Personalized model 1: biased EXPLICIT-rating matrix factorization trained with ALS.

    r_hat(u, i) = mu + b_u + b_i + p_u . q_i

Alternating least squares with the bias folded into the factor vector: a
user's unknowns [b_u, p_u] are solved against the fixed item side [1, q_i]
in one ridge regression per user, then the item unknowns [b_i, q_i] against
the fixed user side [1, p_u].  Each epoch is two passes of small
(n_factors+1)-dimensional solves, so training 100k ratings takes seconds in
plain NumPy and is fully deterministic given the seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .base import Recommender, TrainingContext


class ALSMatrixFactorization(Recommender):
    name = "als_matrix_factorization"

    def __init__(self, n_factors: int = 20, n_epochs: int = 12, regularization: float = 0.1, bias_regularization: float = 1.0,
                 popularity_prior: float = 0.25, seed: int = 42):
        self.n_factors, self.n_epochs, self.reg, self.seed = n_factors, n_epochs, regularization, seed
        self.bias_reg = bias_regularization
        self.popularity_prior = popularity_prior
        self.epoch_rmse: list[float] = []

    # ---- training -----------------------------------------------------------
    def fit(self, train: pd.DataFrame, ctx: TrainingContext, callback=None) -> "ALSMatrixFactorization":
        rng = np.random.default_rng(self.seed)
        self.user_ids = np.sort(train["user_id"].unique())
        self.item_ids = ctx.item_ids
        uidx = pd.Index(self.user_ids).get_indexer(train["user_id"])
        iidx = pd.Index(self.item_ids).get_indexer(train["item_id"])
        r = train["rating"].to_numpy(dtype=float)
        self.mu = float(r.mean())
        n_u, n_i, f = len(self.user_ids), len(self.item_ids), self.n_factors
        R = sp.csr_matrix((r - self.mu, (uidx, iidx)), shape=(n_u, n_i))
        Rt = R.tocsc().T.tocsr()
        # factor matrices with the bias as column 0
        P = np.hstack([np.zeros((n_u, 1)), rng.normal(0, 0.05, (n_u, f))])
        Q = np.hstack([np.zeros((n_i, 1)), rng.normal(0, 0.05, (n_i, f))])
        eye = np.eye(f + 1)
        eye[0, 0] = self.bias_reg  # a weakly regularized bias lets an item rated 5 by two people top every list
        self.log_counts = np.log1p(np.bincount(iidx, minlength=n_i).astype(float))
        self.epoch_rmse = []
        for epoch in range(1, self.n_epochs + 1):
            self._solve_side(R, P, Q, eye)    # users given items
            self._solve_side(Rt, Q, P, eye)   # items given users
            pred = self.mu + P[uidx, 0] + Q[iidx, 0] + np.einsum("ij,ij->i", P[uidx, 1:], Q[iidx, 1:])
            rmse = float(np.sqrt(np.mean((pred - r) ** 2)))
            self.epoch_rmse.append(rmse)
            if callback:
                callback(epoch, rmse)
        self.P, self.Q = P, Q
        return self

    def _solve_side(self, R: sp.csr_matrix, X: np.ndarray, Y: np.ndarray, eye: np.ndarray) -> None:
        """Solve every row of X given Y:  min ||r - (x_b + y_b + x.y)||^2 + reg*n*||x||^2."""
        for row in range(R.shape[0]):
            start, end = R.indptr[row], R.indptr[row + 1]
            if start == end:
                X[row] = 0.0
                continue
            cols, vals = R.indices[start:end], R.data[start:end]
            Yr = Y[cols]
            A = np.hstack([np.ones((len(cols), 1)), Yr[:, 1:]])   # [1, y_factors]
            target = vals - Yr[:, 0]                              # remove the other side's bias
            n = len(cols)
            X[row] = np.linalg.solve(A.T @ A + self.reg * n * eye, A.T @ target)

    # ---- inference ----------------------------------------------------------
    def _scores(self, user_id: int) -> np.ndarray | None:
        pos = pd.Index(self.user_ids).get_indexer([user_id])[0]
        if pos < 0:
            return None
        p = self.P[pos]
        return self.mu + p[0] + self.Q[:, 0] + self.Q[:, 1:] @ p[1:]

    def recommend(self, user_id: int, k: int, exclude: set[int]) -> list[int]:
        """Rank by predicted rating plus a popularity prior (``popularity_prior`` x
        log1p(train ratings)): a predicted 4.6 backed by 300 ratings should beat a
        4.7 backed by two.  The prior changes the ORDER only; ``predict_rating``
        stays the pure model estimate the RMSE is measured on."""
        s = self._scores(user_id)
        if s is None:  # cold-start user: fall back to item bias ranking (the model's own popularity)
            s = self.mu + self.Q[:, 0]
        return self._top_k(s + self.popularity_prior * self.log_counts, self.item_ids, k, exclude)

    def predict_rating(self, user_ids: np.ndarray, item_ids: np.ndarray) -> np.ndarray:
        up = pd.Index(self.user_ids).get_indexer(user_ids)
        ip = pd.Index(self.item_ids).get_indexer(item_ids)
        pred = np.full(len(user_ids), self.mu)
        ok_u, ok_i = up >= 0, ip >= 0
        pred[ok_u] += self.P[up[ok_u], 0]
        pred[ok_i] += self.Q[ip[ok_i], 0]
        both = ok_u & ok_i
        pred[both] += np.einsum("ij,ij->i", self.P[up[both], 1:], self.Q[ip[both], 1:])
        return np.clip(pred, 1.0, 5.0)

    def params(self) -> dict:
        return {"n_factors": self.n_factors, "n_epochs": self.n_epochs, "regularization": self.reg,
                "bias_regularization": self.bias_reg, "popularity_prior": self.popularity_prior, "seed": self.seed,
                "solver": "ALS (biased, ridge per row)"}

    def train_metrics(self) -> dict:
        return {"train_rmse": round(self.epoch_rmse[-1], 4), "global_mean_rating": round(self.mu, 4)}
