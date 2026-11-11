"""Stage 4 - Data transformation.

Derived datasets, computed from the TRAIN split only (so nothing from the test
period leaks into them):

* ``user_activity_summary.parquet``  one row per user
* ``item_popularity.parquet``        one row per item, ranked
* ``item_genres.parquet``            item x genre one-hot table
* ``user_item_matrix.npz``           sparse users x items rating matrix
  (+ ``matrix_index.json`` mapping row/col positions to ids)
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ..common.config import load_config, path_of
from ..common.io import read_parquet, write_json, write_parquet
from ..common.logging import get_logger
from ..common.stage import StageResult
from ..ingestion.movielens import GENRES


def genre_slug(g: str) -> str:
    return "genre_" + re.sub(r"[^a-z0-9]+", "_", g.lower()).strip("_")


GENRE_COLUMNS = [genre_slug(g) for g in GENRES]


def user_activity_summary(train: pd.DataFrame, like_threshold: float) -> pd.DataFrame:
    g = train.groupby("user_id")
    out = pd.DataFrame({
        "n_interactions": g.size(),
        "n_items": g["item_id"].nunique(),
        "mean_rating": g["rating"].mean(),
        "std_rating": g["rating"].std().fillna(0.0),
        "n_liked": g["rating"].apply(lambda s: int((s >= like_threshold).sum())),
        "first_ts": g["timestamp"].min(),
        "last_ts": g["timestamp"].max(),
    })
    out["like_rate"] = out["n_liked"] / out["n_interactions"]
    out["active_span_days"] = (out["last_ts"] - out["first_ts"]) / 86400.0
    return out.reset_index()


def item_popularity(train: pd.DataFrame, items: pd.DataFrame, like_threshold: float) -> pd.DataFrame:
    g = train.groupby("item_id")
    pop = pd.DataFrame({
        "n_ratings": g.size(),
        "n_users": g["user_id"].nunique(),
        "mean_rating": g["rating"].mean(),
        "std_rating": g["rating"].std().fillna(0.0),
        "n_liked": g["rating"].apply(lambda s: int((s >= like_threshold).sum())),
        "first_ts": g["timestamp"].min(),
        "last_ts": g["timestamp"].max(),
    })
    pop = items[["item_id", "title"]].merge(pop, left_on="item_id", right_index=True, how="left")
    pop[["n_ratings", "n_users", "n_liked"]] = pop[["n_ratings", "n_users", "n_liked"]].fillna(0).astype("int64")
    pop["like_rate"] = (pop["n_liked"] / pop["n_ratings"].replace(0, np.nan)).fillna(0.0)
    # "popular AND liked": the log dampens the raw count so a film rated 3.0 by
    # everyone does not outrank one rated 4.5 by half as many.
    pop["popularity_score"] = np.log1p(pop["n_ratings"]) * pop["mean_rating"].fillna(0.0)
    pop["popularity_rank"] = pop["popularity_score"].rank(ascending=False, method="first").astype("int64")
    return pop.sort_values("popularity_rank").reset_index(drop=True)


def item_genre_table(items: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in items[["item_id", "genres"]].iterrows():
        on = {genre_slug(g) for g in str(r["genres"]).split("|") if g}
        rows.append([r["item_id"]] + [int(c in on) for c in GENRE_COLUMNS])
    out = pd.DataFrame(rows, columns=["item_id"] + GENRE_COLUMNS)
    out["n_genres"] = out[GENRE_COLUMNS].sum(axis=1)
    return out


def interaction_matrix(train: pd.DataFrame, users: pd.DataFrame, items: pd.DataFrame):
    user_ids = np.sort(users["user_id"].unique())
    item_ids = np.sort(items["item_id"].unique())
    u_index = {u: i for i, u in enumerate(user_ids)}
    i_index = {it: j for j, it in enumerate(item_ids)}
    rows = train["user_id"].map(u_index).to_numpy()
    cols = train["item_id"].map(i_index).to_numpy()
    m = sp.csr_matrix((train["rating"].to_numpy(), (rows, cols)), shape=(len(user_ids), len(item_ids)))
    index = {"user_ids": user_ids.tolist(), "item_ids": item_ids.tolist()}
    return m, index


def run_transformation(cfg: dict | None = None) -> StageResult:
    cfg = cfg or load_config()
    log = get_logger("transformation", cfg["paths"]["logs"])
    res = StageResult("transformation")
    proc = path_of(cfg, "processed")
    like = cfg["features"]["rating_like_threshold"]
    train = read_parquet(proc / "train.parquet")
    users = read_parquet(proc / "users.parquet")
    items = read_parquet(proc / "items.parquet")

    ua = user_activity_summary(train, like)
    res.outputs["user_activity_summary"] = str(write_parquet(ua, proc / "user_activity_summary.parquet"))
    log.info("user_activity_summary: %d users; median interactions/user = %.0f; mean rating = %.2f",
             len(ua), ua["n_interactions"].median(), ua["mean_rating"].mean())

    pop = item_popularity(train, items, like)
    res.outputs["item_popularity"] = str(write_parquet(pop, proc / "item_popularity.parquet"))
    top = pop.head(5)
    log.info("item_popularity: %d items; %d never rated in train; top-5: %s",
             len(pop), int((pop["n_ratings"] == 0).sum()), " | ".join(f"{t} ({n})" for t, n in zip(top["title"], top["n_ratings"])))

    ig = item_genre_table(items)
    res.outputs["item_genres"] = str(write_parquet(ig, proc / "item_genres.parquet"))
    log.info("item_genres: %d items x %d genres", len(ig), len(GENRE_COLUMNS))

    m, index = interaction_matrix(train, users, items)
    sp.save_npz(proc / "user_item_matrix.npz", m)
    res.outputs["user_item_matrix"] = str(proc / "user_item_matrix.npz")
    res.outputs["matrix_index"] = str(write_json(index, proc / "matrix_index.json"))
    density = m.nnz / (m.shape[0] * m.shape[1])
    log.info("user_item_matrix: shape %s, %d non-zeros, density %.4f (sparsity %.2f%%)",
             m.shape, m.nnz, density, 100 * (1 - density))
    res.metrics = {"users": int(m.shape[0]), "items": int(m.shape[1]), "nnz": int(m.nnz), "density": round(density, 5)}
    log.info("transformation complete")
    return res.done()


if __name__ == "__main__":
    run_transformation()
