"""Features are computed from train only, the store round-trips them, and every
model implements the interface (no seen items in a list, k respected)."""
import numpy as np
import pandas as pd

from src.features.build import compute_interaction_features, compute_item_features, compute_user_features
from src.features.definitions import FEATURE_NAMES
from src.features.store import FeatureStore
from src.training.base import TrainingContext
from src.training.content_based import ContentBasedRecommender
from src.training.implicit_als import ImplicitALS
from src.training.matrix_factorization import ALSMatrixFactorization
from src.training.popularity import PopularityRecommender
from src.transformation.transform import item_genre_table


def _world(seed=1, n_users=25, n_items=40):
    rng = np.random.default_rng(seed)
    users = pd.DataFrame({"user_id": range(1, n_users + 1), "age": rng.integers(18, 60, n_users),
                          "gender": rng.choice(["M", "F"], n_users), "occupation": "x"})
    genres = ["Action", "Comedy", "Drama", "Sci-Fi"]
    items = pd.DataFrame({"item_id": range(1, n_items + 1), "title": [f"m{i}" for i in range(n_items)],
                          "genres": ["|".join(rng.choice(genres, size=rng.integers(1, 3), replace=False)) for _ in range(n_items)],
                          "release_year": pd.array(rng.integers(1980, 2000, n_items), dtype="Int64"),
                          "release_year_missing": False})
    items.loc[0, ["release_year", "release_year_missing"]] = [pd.NA, True]
    rows = set()
    while len(rows) < 400:
        rows.add((int(rng.integers(1, n_users + 1)), int(rng.integers(1, n_items + 1))))
    train = pd.DataFrame(sorted(rows), columns=["user_id", "item_id"])
    train["rating"] = rng.integers(1, 6, len(train)).astype(float)
    train["timestamp"] = rng.integers(880_000_000, 890_000_000, len(train))
    return users, items, train


def test_features_match_catalog_and_round_trip(tmp_path):
    users, items, train = _world()
    ig = item_genre_table(items)
    ref = int(train["timestamp"].max())
    uf = compute_user_features(train, users, ig, ref, 4.0)
    itf = compute_item_features(train, items, ig, ref, 4.0)
    xf = compute_interaction_features(train, uf, itf)
    assert list(uf.columns) == ["user_id"] + FEATURE_NAMES["user"]
    assert list(itf.columns) == ["item_id"] + FEATURE_NAMES["item"]
    assert list(xf.columns) == ["user_id", "item_id"] + FEATURE_NAMES["interaction"]
    assert (uf["user_days_since_last_interaction"] >= 0).all()      # reference time is the train max
    assert itf["item_release_year"].notna().all()                   # missing year filled
    assert bool(itf.loc[itf["item_id"] == 1, "item_release_year_missing"].iloc[0])
    store = FeatureStore(tmp_path)
    store.write("user", uf); store.write("item", itf); store.write("interaction", xf)
    back = store.get_user_features([3, 1])
    assert back["user_id"].tolist() == [3, 1]
    assert back.loc[0, "user_n_interactions"] == uf.set_index("user_id").loc[3, "user_n_interactions"]
    assert store.get_interaction_features(train.head(5))["ui_genre_affinity"].notna().all()
    assert set(store.registry()) == {"user", "item", "interaction"}
    # an undeclared column is refused: the catalog cannot drift from the data
    try:
        store.write("user", uf.assign(bogus=1))
        raise AssertionError("undeclared feature accepted")
    except ValueError:
        pass


def test_models_respect_the_interface(tmp_path):
    users, items, train = _world()
    ig = item_genre_table(items)
    ref = int(train["timestamp"].max())
    store = FeatureStore(tmp_path)
    uf = compute_user_features(train, users, ig, ref, 4.0)
    itf = compute_item_features(train, items, ig, ref, 4.0)
    store.write("user", uf); store.write("item", itf)
    ctx = TrainingContext(items=items, item_genres=ig, store=store, cfg={}, seed=1)
    seen = train.groupby("user_id")["item_id"].agg(set)
    for model in (PopularityRecommender(), ImplicitALS(n_factors=8, n_epochs=3),
                  ALSMatrixFactorization(n_factors=8, n_epochs=3), ContentBasedRecommender()):
        model.fit(train, ctx)
        for u in (1, 2, 3):
            recs = model.recommend(u, 5, seen[u])
            assert len(recs) <= 5 and not (set(recs) & seen[u]), model.name
            assert len(set(recs)) == len(recs)
        pred = model.predict_rating(train["user_id"].to_numpy()[:20], train["item_id"].to_numpy()[:20])
        if pred is not None:
            assert pred.shape == (20,) and ((pred >= 1) & (pred <= 5)).all(), model.name
        # cold-start user and unknown item never crash
        assert isinstance(model.recommend(10_000, 5, set()), list)
