"""Documented fault injection for the ingestion stage.

MovieLens 100K is already clean: validation would pass every check and
``data/errors`` would stay empty, which proves nothing about the validation
layer.  When ``dataset.fault_injection.enabled`` is true, a small, counted set
of corrupt records is APPENDED to the three raw tables - never written over a
real row - so that

* every class of error the validator knows has rows to catch,
* the rejected rows land in ``data/errors/<table>_errors.csv`` with a reason,
* the clean output is exactly the original dataset (an appended copy is the
  "second and later" duplicate, so the real row is the one that is kept).

What was injected is recorded in ``data/raw/ingestion_manifest.json``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def inject_faults(tables: dict[str, pd.DataFrame], fi: dict, seed: int) -> tuple[dict[str, pd.DataFrame], dict]:
    """Return (tables with corrupt rows appended, {fault name: rows injected})."""
    rng = np.random.default_rng(seed)
    users, items, inter = (tables[n].astype(object) for n in ("users", "items", "interactions"))
    n, nu, ni = int(fi["interaction_rows"]), int(fi["user_rows"]), int(fi["item_rows"])
    summary: dict[str, int] = {}

    def sample(df: pd.DataFrame, k: int) -> pd.DataFrame:
        return df.iloc[rng.choice(len(df), size=min(k, len(df)), replace=False)].copy()

    def add(bucket: list, name: str, frame: pd.DataFrame) -> None:
        bucket.append(frame)
        summary[name] = len(frame)

    bad: list[pd.DataFrame] = []
    s = sample(inter, n); s["rating"] = None; add(bad, "interactions.missing_rating", s)
    s = sample(inter, n); s["rating"] = rng.choice(["0", "6", "9", "-1"], len(s)); add(bad, "interactions.rating_out_of_range", s)
    s = sample(inter, n); s["rating"] = rng.choice(["five", "n/a", "4.x"], len(s)); add(bad, "interactions.rating_wrong_type", s)
    s = sample(inter, n); s["timestamp"] = rng.choice(["0", "4102444800"], len(s)); add(bad, "interactions.timestamp_out_of_range", s)
    s = sample(inter, n); s["timestamp"] = rng.choice(["abc", "1997-13-45"], len(s)); add(bad, "interactions.timestamp_wrong_type", s)
    s = sample(inter, n); add(bad, "interactions.exact_duplicate", s)
    s = sample(inter, n); s["user_id"] = (900000 + np.arange(len(s))).astype(str); add(bad, "interactions.unknown_user", s)
    s = sample(inter, n); s["item_id"] = (800000 + np.arange(len(s))).astype(str); add(bad, "interactions.unknown_item", s)
    s = sample(inter, n); s["user_id"] = None; add(bad, "interactions.missing_user_id", s)
    inter = pd.concat([inter, *bad], ignore_index=True)

    u_bad: list[pd.DataFrame] = []
    s = sample(users, nu); add(u_bad, "users.exact_duplicate", s)
    s = sample(users, nu); s["user_id"] = (700000 + np.arange(len(s))).astype(str)
    s["age"] = rng.choice(["-3", "200", "abc"], len(s)); add(u_bad, "users.invalid_age", s)
    s = sample(users, nu); s["user_id"] = (710000 + np.arange(len(s))).astype(str)
    s["gender"] = rng.choice(["Q", "X"], len(s)); add(u_bad, "users.invalid_gender", s)
    users = pd.concat([users, *u_bad], ignore_index=True)

    i_bad: list[pd.DataFrame] = []
    s = sample(items, ni); add(i_bad, "items.exact_duplicate", s)
    s = sample(items, ni); s["item_id"] = (600000 + np.arange(len(s))).astype(str)
    s["title"] = None; add(i_bad, "items.missing_title", s)
    s = sample(items, ni); s["item_id"] = (610000 + np.arange(len(s))).astype(str)
    s["title"] = s["title"].astype(str) + " [bad date]"      # its own title, so no real row gains a duplicate-title warning
    s["release_date"] = "31-Feb-1995"; add(i_bad, "items.unparseable_date", s)
    items = pd.concat([items, *i_bad], ignore_index=True)

    return {"users": users, "items": items, "interactions": inter}, summary
