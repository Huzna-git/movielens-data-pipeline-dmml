"""Ranking and rating metrics.  Pure functions, unit-tested in tests/test_metrics.py."""
from __future__ import annotations

import numpy as np


def precision_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    top = recommended[:k]
    return (sum(1 for i in top if i in relevant) / k) if k else 0.0


def recall_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for i in recommended[:k] if i in relevant) / len(relevant)


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Binary-gain NDCG: DCG over the hits / DCG of the ideal list (min(|rel|, k) hits on top)."""
    if not relevant:
        return 0.0
    dcg = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(recommended[:k]) if item in relevant)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    return float(dcg / idcg) if idcg else 0.0


def coverage(all_recommendations: list[list[int]], catalog_size: int) -> float:
    """Share of the catalogue that appears in at least one user's top-K list."""
    seen = {i for recs in all_recommendations for i in recs}
    return len(seen) / catalog_size if catalog_size else 0.0


def intra_list_diversity(recommended: list[int], vectors: dict[int, np.ndarray]) -> float:
    """1 - mean pairwise cosine similarity of the items in one list (genre vectors)."""
    vs = [vectors[i] for i in recommended if i in vectors]
    if len(vs) < 2:
        return 0.0
    m = np.vstack(vs)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    m = np.divide(m, n, out=np.zeros_like(m), where=n > 0)
    sim = m @ m.T
    iu = np.triu_indices(len(vs), k=1)
    return float(1.0 - sim[iu].mean())


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))
