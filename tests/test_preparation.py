"""The time-based split never puts a user's later interaction into train."""
import numpy as np
import pandas as pd

from src.preparation.prepare import time_split


def _interactions(n_users=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_users + 1):
        n = int(rng.integers(1, 40))
        ts = np.sort(rng.integers(880_000_000, 890_000_000, size=n))
        for j, t in enumerate(ts):
            rows.append((u, j + 1, float(rng.integers(1, 6)), int(t)))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def test_per_user_split_is_chronological_and_sized():
    x = _interactions()
    train, test, info = time_split(x, "per_user_time", 0.2, 5)
    assert len(train) + len(test) == len(x)
    last_train = train.groupby("user_id")["timestamp"].max()
    first_test = test.groupby("user_id")["timestamp"].min()
    both = last_train.index.intersection(first_test.index)
    assert (last_train[both] <= first_test[both]).all()
    sizes = x.groupby("user_id").size()
    small = sizes[sizes < 5].index
    assert not test["user_id"].isin(small).any()           # small histories stay in train
    big = sizes[sizes >= 5]
    expected = np.ceil(big * 0.2).astype(int)
    got = test.groupby("user_id").size().reindex(big.index).fillna(0).astype(int)
    assert (got == expected).all()


def test_global_split_uses_one_cutoff():
    x = _interactions()
    train, test, info = time_split(x, "global_time", 0.25, 1)
    assert train["timestamp"].max() <= info["cutoff_timestamp"] < test["timestamp"].min()
