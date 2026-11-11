"""A documented synthetic dataset, for running the pipeline without a download.

Generation (seeded, so it is reproducible):

1. Items get a release year (1980-2020) and 1-3 of the 18 MovieLens genres.
2. Users get an age, gender, occupation and a hidden preference vector over the
   genres (Dirichlet).  Activity follows a long tail (Zipf-like).
3. An interaction's rating is ``clip(round(2.5 + 2.5 * affinity + noise), 1, 5)``
   where affinity is the dot product of the user's preference vector and the
   item's genre indicator (normalized).  Timestamps are spread over one year,
   each user's interactions in time order.
4. ``noise_rate`` of the rows are then DELIBERATELY corrupted so the validation
   stage has something to catch: ratings out of range or non-numeric,
   timestamps outside the window, unknown user/item ids, missing values and
   exact duplicate rows.  The corruption is logged in the ingestion manifest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .movielens import GENRES

OCCUPATIONS = ["student", "engineer", "teacher", "artist", "doctor", "writer", "other"]


def generate(n_users: int, n_items: int, n_interactions: int, noise_rate: float, seed: int):
    rng = np.random.default_rng(seed)
    genres = GENRES[1:]  # drop "unknown"
    g = len(genres)

    # --- items ---------------------------------------------------------------
    item_genre = np.zeros((n_items, g))
    for i in range(n_items):
        k = rng.integers(1, 4)
        item_genre[i, rng.choice(g, size=k, replace=False)] = 1
    items = pd.DataFrame({
        "item_id": np.arange(1, n_items + 1),
        "title": [f"Synthetic Movie {i} ({1980 + int(rng.integers(0, 41))})" for i in range(1, n_items + 1)],
        "release_date": [f"01-Jan-{1980 + int(rng.integers(0, 41))}" for _ in range(n_items)],
        "imdb_url": "",
        "genres": ["|".join(genres[j] for j in np.flatnonzero(row)) for row in item_genre],
    })

    # --- users ---------------------------------------------------------------
    user_pref = rng.dirichlet(np.full(g, 0.3), size=n_users)
    users = pd.DataFrame({
        "user_id": np.arange(1, n_users + 1),
        "age": rng.integers(18, 70, size=n_users),
        "gender": rng.choice(["M", "F"], size=n_users),
        "occupation": rng.choice(OCCUPATIONS, size=n_users),
        "zip_code": [f"{int(z):05d}" for z in rng.integers(10000, 99999, size=n_users)],
    })

    # --- interactions --------------------------------------------------------
    # Long-tailed activity: each user's history length is log-normal, scaled so
    # the total is about n_interactions; items are drawn WITHOUT replacement per
    # user with probability proportional to a log-normal popularity.
    item_pop = rng.lognormal(0.0, 1.2, size=n_items)
    item_pop = item_pop / item_pop.sum()
    hist_len = rng.lognormal(0.0, 0.9, size=n_users)
    hist_len = np.clip(np.round(hist_len / hist_len.sum() * n_interactions), 3, n_items // 2).astype(int)
    t0 = pd.Timestamp("2023-01-01", tz="UTC").timestamp()
    u_col, i_col, r_col, t_col = [], [], [], []
    for u in range(n_users):
        chosen = rng.choice(n_items, size=hist_len[u], replace=False, p=item_pop)
        affinity = (user_pref[u] * item_genre[chosen]).sum(axis=1) / item_genre[chosen].sum(axis=1)
        affinity = affinity / (user_pref[u].max() + 1e-9)
        rating = np.clip(np.round(2.5 + 2.5 * affinity + rng.normal(0, 0.7, len(chosen))), 1, 5)
        ts = np.sort(t0 + rng.uniform(0, 365 * 86400, size=len(chosen)))
        u_col.append(np.full(len(chosen), u + 1)); i_col.append(chosen + 1); r_col.append(rating); t_col.append(ts.astype(int))
    inter = pd.DataFrame({"user_id": np.concatenate(u_col), "item_id": np.concatenate(i_col),
                          "rating": np.concatenate(r_col), "timestamp": np.concatenate(t_col)})
    inter = inter.sort_values("timestamp").reset_index(drop=True)

    # --- deliberate corruption ----------------------------------------------
    inter = inter.astype({"user_id": object, "item_id": object, "rating": object, "timestamp": object})
    n_bad = int(len(inter) * noise_rate)
    bad = rng.choice(len(inter), size=n_bad, replace=False)
    kinds = rng.choice(["rating_range", "rating_type", "timestamp", "unknown_user", "unknown_item", "missing"],
                       size=n_bad)
    for row, kind in zip(bad, kinds):
        if kind == "rating_range":
            inter.at[row, "rating"] = float(rng.choice([0, 6, 9]))
        elif kind == "rating_type":
            inter.at[row, "rating"] = "five"
        elif kind == "timestamp":
            inter.at[row, "timestamp"] = int(rng.choice([0, 4102444800]))  # 1970 / 2100
        elif kind == "unknown_user":
            inter.at[row, "user_id"] = n_users + 1000
        elif kind == "unknown_item":
            inter.at[row, "item_id"] = n_items + 1000
        elif kind == "missing":
            inter.at[row, "rating"] = None
    dup = inter.sample(n=max(1, n_bad // 5), random_state=seed)
    inter = pd.concat([inter, dup], ignore_index=True)
    corruption = {str(k): int((kinds == k).sum()) for k in np.unique(kinds)}
    corruption["duplicate_row"] = len(dup)
    return {"users": users, "items": items, "interactions": inter}, corruption
