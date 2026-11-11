"""The feature catalog: one ``FeatureSpec`` per feature, in one place.

``write_catalog`` renders it as the Markdown table the project ships in
``reports/feature_catalog.md``.  ``build.py`` computes exactly these features
and the store refuses to write a column that is not declared here, so the
catalog cannot drift from the code.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

from ..transformation.transform import GENRE_COLUMNS


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    entity: str        # user | item | interaction
    dtype: str
    description: str
    source: str        # which table(s) it is computed from
    window: str = "train split"   # the data it is computed over (leakage guard)


USER_FEATURES = [
    FeatureSpec("age", "user", "int", "Age reported in the user profile", "users", "profile"),
    FeatureSpec("gender", "user", "category", "Gender code (M/F) from the user profile", "users", "profile"),
    FeatureSpec("occupation", "user", "category", "Occupation from the user profile", "users", "profile"),
    FeatureSpec("user_n_interactions", "user", "int", "Number of ratings the user gave (activity count)", "interactions"),
    FeatureSpec("user_n_liked", "user", "int", "Number of ratings >= like threshold (4.0)", "interactions"),
    FeatureSpec("user_like_rate", "user", "float", "Share of the user's ratings that are 'likes'", "interactions"),
    FeatureSpec("user_mean_rating", "user", "float", "User's average rating (rating bias)", "interactions"),
    FeatureSpec("user_rating_std", "user", "float", "Std-dev of the user's ratings (0 when a single rating)", "interactions"),
    FeatureSpec("user_days_since_last_interaction", "user", "float", "Days from the user's last rating to the reference time (recency)", "interactions"),
    FeatureSpec("user_active_span_days", "user", "float", "Days between the user's first and last rating", "interactions"),
    FeatureSpec("user_preferred_genre", "user", "category", "Genre with the most liked ratings (ties: higher mean rating)", "interactions + items"),
    *[FeatureSpec(f"user_pref_{c[len('genre_'):]}", "user", "float",
                  f"Share of the user's liked ratings in genre '{c[len('genre_'):]}' (genre profile vector)",
                  "interactions + items") for c in GENRE_COLUMNS],
]

ITEM_FEATURES = [
    FeatureSpec("item_n_ratings", "item", "int", "Number of ratings the item received (popularity count)", "interactions"),
    FeatureSpec("item_mean_rating", "item", "float", "Item's average rating (NaN when never rated in train)", "interactions"),
    FeatureSpec("item_rating_std", "item", "float", "Std-dev of the item's ratings", "interactions"),
    FeatureSpec("item_like_rate", "item", "float", "Share of the item's ratings that are 'likes'", "interactions"),
    FeatureSpec("item_popularity_score", "item", "float", "log1p(n_ratings) * mean_rating - popular AND liked", "interactions"),
    FeatureSpec("item_popularity_rank", "item", "int", "Rank of the popularity score (1 = most popular)", "interactions"),
    FeatureSpec("item_days_since_first_rating", "item", "float", "Days from the item's first rating to the reference time (catalog age)", "interactions"),
    FeatureSpec("item_release_year", "item", "int", "Release year; missing years are filled with the catalogue median", "items", "catalogue"),
    FeatureSpec("item_release_year_missing", "item", "bool", "True when the release year was absent in the source", "items", "catalogue"),
    FeatureSpec("item_n_genres", "item", "int", "How many genres the item carries", "items", "catalogue"),
    *[FeatureSpec(c, "item", "bool", f"Item is tagged with genre '{c[len('genre_'):]}'", "items", "catalogue") for c in GENRE_COLUMNS],
]

INTERACTION_FEATURES = [
    FeatureSpec("ui_genre_affinity", "interaction", "float", "Mean of the user's genre-profile shares over the item's genres (0 = never liked any of them)", "user + item features"),
    FeatureSpec("ui_days_since_prev_interaction", "interaction", "float", "Days since the same user's previous rating (0 for the first)", "interactions"),
    FeatureSpec("ui_user_bias_minus_item_bias", "interaction", "float", "user_mean_rating - item_mean_rating (who is 'harder', the user or the item)", "user + item features"),
    FeatureSpec("ui_item_popularity_pct", "interaction", "float", "Percentile of the item's popularity rank (1.0 = most popular)", "item features"),
]

ALL_FEATURES = USER_FEATURES + ITEM_FEATURES + INTERACTION_FEATURES
FEATURE_NAMES = {"user": [f.name for f in USER_FEATURES], "item": [f.name for f in ITEM_FEATURES],
                 "interaction": [f.name for f in INTERACTION_FEATURES]}


def write_catalog(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Feature catalog", "",
             "Every feature is computed by `src/features/build.py` from the **train split only** "
             "(the reference time for recency features is the latest train timestamp), stored by "
             "`src/features/store.py` under `data/features/`, and read back by the same store at "
             "training and at prediction time.", ""]
    for entity, feats in (("User", USER_FEATURES), ("Item", ITEM_FEATURES), ("Interaction", INTERACTION_FEATURES)):
        key = {"User": "user_id", "Item": "item_id", "Interaction": "(user_id, item_id)"}[entity]
        lines += [f"## {entity} features (key: `{key}`, {len(feats)} features)", "",
                  "| Feature | Type | Description | Source | Computed over |", "|---|---|---|---|---|"]
        lines += [f"| `{f.name}` | {f.dtype} | {f.description} | {f.source} | {f.window} |" for f in feats]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def catalog_records() -> list[dict]:
    return [asdict(f) for f in ALL_FEATURES]
