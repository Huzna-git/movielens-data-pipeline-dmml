import numpy as np
import pytest

from src.evaluation import metrics as M


def test_precision_recall_ndcg_known_values():
    recs, rel = [1, 2, 3, 4, 5], {1, 3, 9}
    assert M.precision_at_k(recs, rel, 5) == pytest.approx(2 / 5)
    assert M.recall_at_k(recs, rel, 5) == pytest.approx(2 / 3)
    dcg = 1 / np.log2(2) + 1 / np.log2(4)
    idcg = 1 / np.log2(2) + 1 / np.log2(3) + 1 / np.log2(4)
    assert M.ndcg_at_k(recs, rel, 5) == pytest.approx(dcg / idcg)
    assert M.ndcg_at_k([1, 3, 9], rel, 5) == pytest.approx(1.0)
    assert M.ndcg_at_k(recs, set(), 5) == 0.0


def test_coverage_and_diversity():
    assert M.coverage([[1, 2], [2, 3]], 10) == pytest.approx(0.3)
    vec = {1: np.array([1.0, 0.0]), 2: np.array([1.0, 0.0]), 3: np.array([0.0, 1.0])}
    assert M.intra_list_diversity([1, 2], vec) == pytest.approx(0.0)
    assert M.intra_list_diversity([1, 3], vec) == pytest.approx(1.0)
    assert M.intra_list_diversity([1], vec) == 0.0


def test_rmse_mae():
    assert M.rmse([1, 2, 3], [1, 2, 5]) == pytest.approx(np.sqrt(4 / 3))
    assert M.mae([1, 2, 3], [1, 2, 5]) == pytest.approx(2 / 3)
