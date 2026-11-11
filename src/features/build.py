"""Stage 5 - Feature management: compute the catalogued features and store them.

The three ``compute_*`` functions are pure (frames in, frame out) and are the
same code the prediction path would call for a new day's data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..common.config import load_config, path_of
from ..common.io import read_parquet
from ..common.logging import get_logger
from ..common.stage import StageResult
from ..transformation.transform import GENRE_COLUMNS, item_genre_table, item_popularity, user_activity_summary
from .definitions import FEATURE_NAMES, write_catalog
from .store import FeatureStore

PREF_COLUMNS = [f"user_pref_{c[len('genre_'):]}" for c in GENRE_COLUMNS]


def compute_user_features(train: pd.DataFrame, users: pd.DataFrame, item_genres: pd.DataFrame,
                          reference_ts: int, like_threshold: float) -> pd.DataFrame:
    act = user_activity_summary(train, like_threshold).set_index("user_id")
    liked = train[train["rating"] >= like_threshold].merge(item_genres, on="item_id", how="left")
    genre_counts = liked.groupby("user_id")[GENRE_COLUMNS].sum()
    genre_mean = (liked.melt(id_vars=["user_id", "rating"], value_vars=GENRE_COLUMNS, var_name="genre", value_name="on")
                  .query("on == 1").groupby(["user_id", "genre"])["rating"].mean().unstack(fill_value=0.0))
    genre_mean = genre_mean.reindex(columns=GENRE_COLUMNS, fill_value=0.0)
    # preferred genre: most liked ratings; tie-break on the higher mean rating
    score = genre_counts + genre_mean.reindex(genre_counts.index).fillna(0.0) / 10.0
    preferred = score.idxmax(axis=1).str[len("genre_"):].where(genre_counts.sum(axis=1) > 0, "none")
    profile = genre_counts.div(genre_counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    profile.columns = PREF_COLUMNS

    out = users[["user_id", "age", "gender", "occupation"]].copy()
    out = out.merge(act, left_on="user_id", right_index=True, how="left")
    out["user_n_interactions"] = out.pop("n_interactions").fillna(0).astype("int64")
    out["user_n_liked"] = out.pop("n_liked").fillna(0).astype("int64")
    out["user_like_rate"] = out.pop("like_rate").fillna(0.0)
    out["user_mean_rating"] = out.pop("mean_rating")
    out["user_rating_std"] = out.pop("std_rating").fillna(0.0)
    out["user_days_since_last_interaction"] = (reference_ts - out.pop("last_ts")) / 86400.0
    out["user_active_span_days"] = out.pop("active_span_days").fillna(0.0)
    out = out.drop(columns=["n_items", "first_ts"])
    out["user_preferred_genre"] = out["user_id"].map(preferred).fillna("none")
    out = out.merge(profile, left_on="user_id", right_index=True, how="left")
    out[PREF_COLUMNS] = out[PREF_COLUMNS].fillna(0.0)
    return out[["user_id"] + FEATURE_NAMES["user"]]


def compute_item_features(train: pd.DataFrame, items: pd.DataFrame, item_genres: pd.DataFrame,
                          reference_ts: int, like_threshold: float) -> pd.DataFrame:
    pop = item_popularity(train, items, like_threshold)
    out = pop[["item_id", "n_ratings", "mean_rating", "std_rating", "like_rate", "popularity_score", "popularity_rank", "first_ts"]]
    out = out.rename(columns={"n_ratings": "item_n_ratings", "mean_rating": "item_mean_rating", "std_rating": "item_rating_std",
                              "like_rate": "item_like_rate", "popularity_score": "item_popularity_score",
                              "popularity_rank": "item_popularity_rank"})
    out["item_days_since_first_rating"] = (reference_ts - out.pop("first_ts")) / 86400.0
    meta = items[["item_id", "release_year", "release_year_missing"]].copy()
    median_year = int(meta["release_year"].dropna().median())
    meta["item_release_year"] = meta.pop("release_year").fillna(median_year).astype("int64")
    meta["item_release_year_missing"] = meta.pop("release_year_missing").astype(bool)
    out = out.merge(meta, on="item_id", how="left")
    ig = item_genres.rename(columns={"n_genres": "item_n_genres"})
    out = out.merge(ig[["item_id", "item_n_genres"] + GENRE_COLUMNS], on="item_id", how="left")
    out[GENRE_COLUMNS] = out[GENRE_COLUMNS].astype(bool)
    return out.sort_values("item_id")[["item_id"] + FEATURE_NAMES["item"]].reset_index(drop=True)


def compute_interaction_features(train: pd.DataFrame, user_f: pd.DataFrame, item_f: pd.DataFrame) -> pd.DataFrame:
    x = train[["user_id", "item_id", "timestamp"]].sort_values(["user_id", "timestamp"]).copy()
    prev = x.groupby("user_id")["timestamp"].shift()
    x["ui_days_since_prev_interaction"] = ((x["timestamp"] - prev) / 86400.0).fillna(0.0)
    uf = user_f.set_index("user_id")
    itf = item_f.set_index("item_id")
    U = uf.loc[x["user_id"], PREF_COLUMNS].to_numpy(dtype=float)
    G = itf.loc[x["item_id"], GENRE_COLUMNS].to_numpy(dtype=float)
    n_genres = np.maximum(G.sum(axis=1), 1.0)
    x["ui_genre_affinity"] = (U * G).sum(axis=1) / n_genres
    x["ui_user_bias_minus_item_bias"] = (uf.loc[x["user_id"], "user_mean_rating"].to_numpy()
                                         - itf.loc[x["item_id"], "item_mean_rating"].to_numpy())
    n_items = len(itf)
    x["ui_item_popularity_pct"] = 1.0 - (itf.loc[x["item_id"], "item_popularity_rank"].to_numpy() - 1) / max(n_items - 1, 1)
    return x[["user_id", "item_id"] + FEATURE_NAMES["interaction"]].reset_index(drop=True)


def run_feature_engineering(cfg: dict | None = None) -> StageResult:
    cfg = cfg or load_config()
    log = get_logger("features", cfg["paths"]["logs"])
    res = StageResult("features")
    proc = path_of(cfg, "processed")
    like = cfg["features"]["rating_like_threshold"]
    train = read_parquet(proc / "train.parquet")
    users, items = read_parquet(proc / "users.parquet"), read_parquet(proc / "items.parquet")
    item_genres = read_parquet(proc / "item_genres.parquet") if (proc / "item_genres.parquet").exists() else item_genre_table(items)
    reference_ts = int(train["timestamp"].max())
    log.info("reference time = %s (latest train timestamp); like threshold = %.1f",
             pd.to_datetime(reference_ts, unit="s", utc=True), like)

    store = FeatureStore(cfg["paths"]["features"])
    meta = {"reference_ts": reference_ts, "like_threshold": like, "train_rows": int(len(train))}
    uf = compute_user_features(train, users, item_genres, reference_ts, like)
    res.outputs["user_features"] = str(store.write("user", uf, meta))
    log.info("user features:        %5d rows x %2d features -> %s", len(uf), len(FEATURE_NAMES["user"]), res.outputs["user_features"])
    itf = compute_item_features(train, items, item_genres, reference_ts, like)
    res.outputs["item_features"] = str(store.write("item", itf, meta))
    log.info("item features:        %5d rows x %2d features -> %s", len(itf), len(FEATURE_NAMES["item"]), res.outputs["item_features"])
    xf = compute_interaction_features(train, uf, itf)
    res.outputs["interaction_features"] = str(store.write("interaction", xf, meta))
    log.info("interaction features: %5d rows x %2d features -> %s", len(xf), len(FEATURE_NAMES["interaction"]), res.outputs["interaction_features"])
    res.outputs["catalog"] = str(write_catalog(path_of(cfg, "reports") / "feature_catalog.md"))
    log.info("feature catalog written: %s (%d features)", res.outputs["catalog"], sum(len(v) for v in FEATURE_NAMES.values()))
    log.info("sample user features:\n%s", uf[["user_id", "user_n_interactions", "user_mean_rating", "user_days_since_last_interaction", "user_preferred_genre"]].head(3).to_string(index=False))
    res.metrics = {"user_features": len(FEATURE_NAMES["user"]), "item_features": len(FEATURE_NAMES["item"]),
                   "interaction_features": len(FEATURE_NAMES["interaction"]), "users": len(uf), "items": len(itf), "pairs": len(xf)}
    log.info("feature engineering complete")
    return res.done()


if __name__ == "__main__":
    run_feature_engineering()
