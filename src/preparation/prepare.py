"""Stage 3 - Data preparation.

Cleans and standardizes the validated tables (formats, missing values, residual
duplicates), then creates the train/test datasets with a TIME-BASED split.
Outputs land in ``data/processed``: users, items, interactions, train, test
(parquet) and ``split_manifest.json``.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..common.config import load_config, path_of
from ..common.io import read_parquet, write_json, write_parquet
from ..common.logging import get_logger
from ..common.stage import StageResult


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    u = users.copy()
    u["gender"] = u["gender"].str.upper().str.strip()
    u["occupation"] = u["occupation"].str.lower().str.strip()
    u["zip_code"] = u["zip_code"].astype("string").str.strip().fillna("unknown")
    u["age"] = u["age"].astype("int64")
    u["user_id"] = u["user_id"].astype("int64")
    return u.drop_duplicates("user_id").sort_values("user_id").reset_index(drop=True)


def clean_items(items: pd.DataFrame) -> pd.DataFrame:
    it = items.copy()
    it["item_id"] = it["item_id"].astype("int64")
    it["title"] = it["title"].str.strip()
    it["release_date"] = pd.to_datetime(it["release_date"], format="%d-%b-%Y", errors="coerce")
    it["release_year"] = it["release_date"].dt.year.astype("Int64")
    it["release_year_missing"] = it["release_year"].isna()
    it["genres"] = it["genres"].fillna("unknown").str.strip().replace("", "unknown")
    it["imdb_url"] = it["imdb_url"].fillna("")
    return it.drop_duplicates("item_id").sort_values("item_id").reset_index(drop=True)


def clean_interactions(inter: pd.DataFrame) -> pd.DataFrame:
    x = inter.copy()
    x["user_id"] = x["user_id"].astype("int64")
    x["item_id"] = x["item_id"].astype("int64")
    x["rating"] = x["rating"].astype("float64")
    x["timestamp"] = x["timestamp"].astype("int64")
    x["ts"] = pd.to_datetime(x["timestamp"], unit="s", utc=True)
    x = x.drop_duplicates(["user_id", "item_id"], keep="first")
    return x.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)


def time_split(inter: pd.DataFrame, strategy: str, test_fraction: float, min_user_interactions: int):
    """Split interactions by time, never by random sampling.

    per_user_time: each user's history is ordered by time; the last
        ceil(test_fraction * n) interactions become test, the rest train.  A
        user with fewer than ``min_user_interactions`` stays entirely in train.
    global_time:  one cut-off at the (1 - test_fraction) quantile of timestamps.
    """
    x = inter.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    if strategy == "per_user_time":
        n = x.groupby("user_id")["item_id"].transform("size")
        rank = x.groupby("user_id").cumcount()  # 0..n-1 in time order
        n_test = np.where(n >= min_user_interactions, np.ceil(n * test_fraction).astype(int), 0)
        is_test = rank >= (n - n_test)
        info = {"strategy": strategy, "test_fraction": test_fraction, "min_user_interactions": min_user_interactions}
    elif strategy == "global_time":
        cutoff = x["timestamp"].quantile(1 - test_fraction)
        is_test = x["timestamp"] > cutoff
        info = {"strategy": strategy, "cutoff_timestamp": int(cutoff),
                "cutoff_utc": pd.to_datetime(int(cutoff), unit="s", utc=True).isoformat()}
    else:
        raise ValueError(f"unknown split strategy {strategy}")
    train, test = x[~is_test].reset_index(drop=True), x[is_test].reset_index(drop=True)
    return train, test, info


def run_preparation(cfg: dict | None = None) -> StageResult:
    cfg = cfg or load_config()
    log = get_logger("preparation", cfg["paths"]["logs"])
    res = StageResult("preparation")
    proc = path_of(cfg, "processed")

    users = clean_users(read_parquet(proc / "users_validated.parquet"))
    items = clean_items(read_parquet(proc / "items_validated.parquet"))
    inter = clean_interactions(read_parquet(proc / "interactions_validated.parquet"))
    log.info("cleaned: users %d | items %d | interactions %d", len(users), len(items), len(inter))
    log.info("items with no release date: %d (kept, release_year flagged missing)", int(items["release_year_missing"].sum()))
    log.info("interaction window: %s .. %s", inter["ts"].min().date(), inter["ts"].max().date())

    s = cfg["split"]
    train, test, info = time_split(inter, s["strategy"], s["test_fraction"], s["min_user_interactions"])
    train_users, train_items = set(train["user_id"]), set(train["item_id"])
    info.update({
        "train_rows": len(train), "test_rows": len(test),
        "train_users": len(train_users), "test_users": int(test["user_id"].nunique()),
        "train_items": len(train_items), "test_items": int(test["item_id"].nunique()),
        "test_users_unseen_in_train": int((~test["user_id"].isin(train_users)).sum()),
        "test_rows_with_item_unseen_in_train": int((~test["item_id"].isin(train_items)).sum()),
        "train_time_range": [train["ts"].min().isoformat(), train["ts"].max().isoformat()],
        "test_time_range": [test["ts"].min().isoformat(), test["ts"].max().isoformat()],
    })
    log.info("split %s: train %d rows / test %d rows (%.1f%%); %d test rows name an item unseen in train",
             s["strategy"], len(train), len(test), 100 * len(test) / len(inter), info["test_rows_with_item_unseen_in_train"])

    for name, df in {"users": users, "items": items, "interactions": inter, "train": train, "test": test}.items():
        res.outputs[name] = str(write_parquet(df, proc / f"{name}.parquet"))
    res.outputs["split_manifest"] = str(write_json(info, proc / "split_manifest.json"))
    res.metrics = {"users": len(users), "items": len(items), "interactions": len(inter),
                   "train_rows": len(train), "test_rows": len(test)}
    log.info("preparation complete")
    return res.done()


if __name__ == "__main__":
    run_preparation()
