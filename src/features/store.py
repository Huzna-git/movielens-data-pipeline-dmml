"""A small file-backed feature store.

Feature tables are parquet files keyed by entity id under ``data/features``
with a ``registry.json`` beside them (what was built, when, from which
reference time, with which columns).  Training reads whole tables; prediction
asks for the rows of specific ids through the same ``get_*`` methods, so one
code path serves both.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..common.io import utc_now
from .definitions import FEATURE_NAMES

KEYS = {"user": ["user_id"], "item": ["item_id"], "interaction": ["user_id", "item_id"]}


class FeatureStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, pd.DataFrame] = {}

    # ---- writing ------------------------------------------------------------
    def write(self, entity: str, df: pd.DataFrame, meta: dict | None = None) -> Path:
        key = KEYS[entity]
        undeclared = [c for c in df.columns if c not in key and c not in FEATURE_NAMES[entity]]
        if undeclared:
            raise ValueError(f"{entity}: columns not declared in the feature catalog: {undeclared}")
        missing = [c for c in FEATURE_NAMES[entity] if c not in df.columns]
        if missing:
            raise ValueError(f"{entity}: catalog features not computed: {missing}")
        path = self.root / f"{entity}_features.parquet"
        df[key + FEATURE_NAMES[entity]].to_parquet(path, index=False)
        reg = self.registry()
        reg[entity] = {"path": path.name, "rows": int(len(df)), "columns": FEATURE_NAMES[entity],
                       "built_at": utc_now(), **(meta or {})}
        (self.root / "registry.json").write_text(json.dumps(reg, indent=2, default=str), encoding="utf-8")
        self._cache.pop(entity, None)
        return path

    def registry(self) -> dict:
        p = self.root / "registry.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    # ---- reading ------------------------------------------------------------
    def read(self, entity: str) -> pd.DataFrame:
        if entity not in self._cache:
            self._cache[entity] = pd.read_parquet(self.root / f"{entity}_features.parquet")
        return self._cache[entity]

    def get_user_features(self, user_ids) -> pd.DataFrame:
        df = self.read("user").set_index("user_id")
        return df.reindex(np.asarray(user_ids)).reset_index()

    def get_item_features(self, item_ids) -> pd.DataFrame:
        df = self.read("item").set_index("item_id")
        return df.reindex(np.asarray(item_ids)).reset_index()

    def get_interaction_features(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """``pairs`` carries user_id and item_id; unknown pairs come back as NaN."""
        return pairs[["user_id", "item_id"]].merge(self.read("interaction"), on=["user_id", "item_id"], how="left")

    def user_genre_profile(self) -> tuple[np.ndarray, np.ndarray]:
        """(user_ids, matrix) of the user_pref_* vectors - the content model's user side."""
        df = self.read("user")
        cols = [c for c in df.columns if c.startswith("user_pref_")]
        return df["user_id"].to_numpy(), df[cols].to_numpy(dtype=float)

    def item_genre_matrix(self) -> tuple[np.ndarray, np.ndarray]:
        df = self.read("item")
        cols = [c for c in df.columns if c.startswith("genre_")]
        return df["item_id"].to_numpy(), df[cols].to_numpy(dtype=float)
