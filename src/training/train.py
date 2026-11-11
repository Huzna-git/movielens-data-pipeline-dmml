"""Stage 6 - Model training with MLflow experiment tracking.

One MLflow run per model: parameters, training metrics (per-epoch RMSE for
the factorization model), the pickled model and a sample of recommendations as
artifacts.  Every model gets its own folder:

    models/<name>/model.pkl                    the fitted model
    models/<name>/meta.json                    class, parameters, training time, MLflow run id, feature version
    models/<name>/sample_recommendations.csv   top-5 for the first three users
    models/registry.json                       name -> path + run id (read by the evaluation stage)
    models/training_summary.json               every run id, the tuned ALS parameters, the tuned hybrid weights

The hybrid model wraps the trained implicit ALS, so it is trained after it.
"""
from __future__ import annotations

import itertools
import os
import threading
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from ..common.config import load_config, path_of
from ..common.io import read_json, read_parquet, utc_now, write_csv, write_json
from ..common.logging import get_logger
from ..common.stage import StageResult
from ..features.build import compute_item_features, compute_user_features
from ..features.store import FeatureStore
from ..preparation.prepare import time_split
from ..evaluation.metrics import ndcg_at_k, precision_at_k
from .base import Recommender, TrainingContext
from .content_based import ContentBasedRecommender
from .hybrid import HybridRecommender
from .implicit_als import ImplicitALS
from .matrix_factorization import ALSMatrixFactorization
from .popularity import PopularityRecommender

# The base models are independent of each other (the orchestrator trains them in
# parallel); the hybrid reuses the trained implicit ALS and is trained last.
BASE_MODEL_NAMES = ["popularity", "implicit_als", "als_matrix_factorization", "content_based_genres"]
MODEL_NAMES = BASE_MODEL_NAMES + ["hybrid"]
_ALS_KEYS = ("n_factors", "n_epochs", "regularization", "alpha")
_REGISTRY_LOCK = threading.Lock()   # parallel training tasks share models/registry.json


def build_model(name: str, cfg: dict) -> Recommender:
    t = cfg["training"]
    if name == "popularity":
        return PopularityRecommender(**t["popularity"])
    if name == "als_matrix_factorization":
        return ALSMatrixFactorization(**t["matrix_factorization"], seed=cfg["project"]["random_seed"])
    if name == "content_based_genres":
        return ContentBasedRecommender(**t["content_based"])
    if name == "implicit_als":
        return ImplicitALS(**{k: v for k, v in t["implicit_als"].items() if k != "tuning"}, seed=cfg["project"]["random_seed"])
    if name == "hybrid":
        h = t["hybrid"]
        cf = {k: v for k, v in t["implicit_als"].items() if k in _ALS_KEYS}
        return HybridRecommender(w_cf=h["w_cf"], w_genre=h["w_genre"], w_popularity=h["w_popularity"], cf_params=cf,
                                 seed=cfg["project"]["random_seed"])
    raise ValueError(f"unknown model {name}")


def setup_mlflow(cfg: dict) -> None:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    root = Path(cfg["project"]["root"])
    uri = cfg["mlflow"]["tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = "sqlite:///" + str(root / uri[len("sqlite:///"):])   # relative db file -> project root
    mlflow.set_tracking_uri(uri)
    name = cfg["mlflow"]["experiment_name"]
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(root / "mlartifacts").as_uri())
    mlflow.set_experiment(name)


def load_context(cfg: dict) -> tuple[pd.DataFrame, TrainingContext]:
    proc = Path(cfg["paths"]["processed"])
    train = read_parquet(proc / "train.parquet")
    ctx = TrainingContext(items=read_parquet(proc / "items.parquet"), item_genres=read_parquet(proc / "item_genres.parquet"),
                          store=FeatureStore(cfg["paths"]["features"]), cfg=cfg, seed=cfg["project"]["random_seed"])
    return train, ctx


def tune_implicit_als(train: pd.DataFrame, ctx: TrainingContext, cfg: dict, log) -> dict:
    """Grid search on a validation split carved from TRAIN by the same time rule.

    Every candidate is a nested MLflow run with its validation NDCG@K /
    Precision@K; the best candidate's parameters are returned and the parent
    run is refit on the full train split with them.  Test data is never read.
    """
    t = cfg["training"]["implicit_als"]["tuning"]
    k = cfg["evaluation"]["k"]
    thr = cfg["evaluation"]["relevance_threshold"]
    fit_part, val_part, _ = time_split(train, cfg["split"]["strategy"], t["validation_fraction"], cfg["split"]["min_user_interactions"])
    seen = fit_part.groupby("user_id")["item_id"].agg(set)
    relevant = val_part[val_part["rating"] >= thr].groupby("user_id")["item_id"].agg(set)
    best, best_ndcg = None, -1.0
    for n_factors in t["n_factors"]:
        for alpha in t["alpha"]:
            cand = {"n_factors": n_factors, "alpha": alpha, "regularization": t["regularization"], "n_epochs": t["n_epochs"]}
            with mlflow.start_run(run_name=f"ials_f{n_factors}_a{alpha}", nested=True):
                m = ImplicitALS(**cand, seed=cfg["project"]["random_seed"]).fit(fit_part, ctx)
                scores = [(ndcg_at_k(r, rel, k), precision_at_k(r, rel, k))
                          for u, rel in relevant.items() for r in [m.recommend(int(u), k, seen.get(u, set()))]]
                ndcg, prec = float(np.mean([s[0] for s in scores])), float(np.mean([s[1] for s in scores]))
                mlflow.log_params(cand)
                mlflow.log_metrics({f"val_ndcg_at_{k}": ndcg, f"val_precision_at_{k}": prec, "val_users": len(scores)})
                mlflow.set_tags({"stage": "tuning", "model": "implicit_als"})
            log.info("[implicit_als] tuning f=%-3d alpha=%-5s val NDCG@%d %.4f  P@%d %.4f", n_factors, alpha, k, ndcg, k, prec)
            if ndcg > best_ndcg:
                best, best_ndcg = cand, ndcg
    log.info("[implicit_als] best candidate %s (val NDCG@%d %.4f)", best, k, best_ndcg)
    mlflow.log_metric(f"tuning_best_val_ndcg_at_{k}", best_ndcg)
    return best


def trained_implicit_als(cfg: dict) -> tuple[ImplicitALS | None, dict]:
    """The implicit ALS this run already trained (model, its constructor parameters), if any."""
    reg_path = Path(cfg["paths"]["models"]) / "registry.json"
    entry = (read_json(reg_path) if reg_path.exists() else {}).get("implicit_als")
    if not entry or not Path(entry["path"]).exists():
        return None, {k: v for k, v in cfg["training"]["implicit_als"].items() if k in _ALS_KEYS}
    return Recommender.load(Path(entry["path"])), {k: v for k, v in entry["params"].items() if k in _ALS_KEYS}


def tune_hybrid(train: pd.DataFrame, ctx: TrainingContext, cfg: dict, log, cf_params: dict) -> tuple[dict, pd.DataFrame]:
    """Grid search of the blend weights on a validation split carved from TRAIN.

    The collaborative model is refit on the fit part and the genre-profile /
    popularity features are rebuilt in memory from the fit part alone, so the
    validation rows are unseen by every signal.  Each grid point is a nested
    MLflow run; the grid is returned for ``reports/hybrid_tuning.csv``.
    """
    t = cfg["training"]["hybrid"]["tuning"]
    k, thr = cfg["evaluation"]["k"], cfg["evaluation"]["relevance_threshold"]
    like = cfg["features"]["rating_like_threshold"]
    fit_part, val_part, _ = time_split(train, cfg["split"]["strategy"], t["validation_fraction"], cfg["split"]["min_user_interactions"])
    users = read_parquet(Path(cfg["paths"]["processed"]) / "users.parquet")
    ref = int(fit_part["timestamp"].max())
    uf = compute_user_features(fit_part, users, ctx.item_genres, ref, like)
    itf = compute_item_features(fit_part, ctx.items, ctx.item_genres, ref, like)
    seed = cfg["project"]["random_seed"]
    base = ImplicitALS(**cf_params, seed=seed).fit(fit_part, ctx)
    hybrid = HybridRecommender(cf_params=cf_params, seed=seed).fit(fit_part, ctx, base=base, user_features=uf, item_features=itf)
    seen = fit_part.groupby("user_id")["item_id"].agg(set)
    relevant = val_part[val_part["rating"] >= thr].groupby("user_id")["item_id"].agg(set)
    log.info("[hybrid] tuning blend weights: fit part %d rows, validation %d rows, %d users", len(fit_part), len(val_part), len(relevant))
    rows = []
    for wg, wp in itertools.product(t["w_genre"], t["w_popularity"]):
        hybrid.set_weights(w_cf=1.0, w_genre=wg, w_popularity=wp)
        scores = [(ndcg_at_k(r, rel, k), precision_at_k(r, rel, k))
                  for u, rel in relevant.items() for r in [hybrid.recommend(int(u), k, seen.get(u, set()))]]
        ndcg, prec = float(np.mean([s[0] for s in scores])), float(np.mean([s[1] for s in scores]))
        rows.append({"w_cf": 1.0, "w_genre": wg, "w_popularity": wp, f"val_ndcg_at_{k}": ndcg, f"val_precision_at_{k}": prec,
                     "val_users": len(scores)})
        with mlflow.start_run(run_name=f"hybrid_g{wg}_p{wp}", nested=True):
            mlflow.log_params({"w_cf": 1.0, "w_genre": wg, "w_popularity": wp})
            mlflow.log_metrics({f"val_ndcg_at_{k}": ndcg, f"val_precision_at_{k}": prec, "val_users": len(scores)})
            mlflow.set_tags({"stage": "tuning", "model": "hybrid"})
    grid = pd.DataFrame(rows)
    best = grid.sort_values(f"val_ndcg_at_{k}", ascending=False, kind="stable").iloc[0]
    weights = {"w_cf": 1.0, "w_genre": float(best["w_genre"]), "w_popularity": float(best["w_popularity"])}
    log.info("[hybrid] best weights %s (val NDCG@%d %.4f)", weights, k, best[f"val_ndcg_at_{k}"])
    mlflow.log_metric(f"tuning_best_val_ndcg_at_{k}", float(best[f"val_ndcg_at_{k}"]))
    return weights, grid


def write_training_summary(cfg: dict, registry: dict) -> Path:
    """``models/training_summary.json``: one place that names every run of this training stage."""
    summary = {"runs": {n: e["mlflow_run_id"] for n, e in registry.items()},
               "trained_at": {n: e["trained_at"] for n, e in registry.items()}}
    if "implicit_als" in registry:
        summary["implicit_als_params"] = {k: v for k, v in registry["implicit_als"]["params"].items() if k in _ALS_KEYS}
    if "hybrid" in registry:
        summary["best_hybrid_weights"] = {k: registry["hybrid"]["params"][k] for k in ("w_cf", "w_genre", "w_popularity")}
        summary["hybrid_tuned_on_validation_split"] = registry["hybrid"].get("tuned", False)
    return write_json(summary, path_of(cfg, "models") / "training_summary.json")


def train_model(name: str, cfg: dict | None = None) -> StageResult:
    cfg = cfg or load_config()
    log = get_logger("training", cfg["paths"]["logs"])
    res = StageResult(f"train:{name}")
    setup_mlflow(cfg)
    train, ctx = load_context(cfg)
    model = build_model(name, cfg)
    log.info("[%s] training on %d interactions, %d users, %d items", name, len(train), train["user_id"].nunique(), len(ctx.item_ids))

    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"model": name, "stage": "training", "dataset": cfg["dataset"]["source"],
                         "split": cfg["split"]["strategy"]})
        tuned = False
        if isinstance(model, ImplicitALS) and cfg["training"]["implicit_als"].get("tuning", {}).get("enabled"):
            best = tune_implicit_als(train, ctx, cfg, log)      # nested runs; the parent is refit with the winner
            model = ImplicitALS(**best, seed=cfg["project"]["random_seed"])
            tuned = True
        base, extra_artifacts = None, []
        if isinstance(model, HybridRecommender):
            base, cf_params = trained_implicit_als(cfg)          # the collaborative half: this run's implicit ALS
            weights = {"w_cf": model.w_cf, "w_genre": model.w_genre, "w_popularity": model.w_popularity}
            if cfg["training"]["hybrid"].get("tuning", {}).get("enabled"):
                weights, grid = tune_hybrid(train, ctx, cfg, log, cf_params)
                extra_artifacts.append(write_csv(grid, path_of(cfg, "reports") / "hybrid_tuning.csv"))
                tuned = True
            model = HybridRecommender(**weights, cf_params=cf_params, seed=cfg["project"]["random_seed"])
            model.fit(train, ctx, base=base)
            log.info("[hybrid] collaborative half: %s", "the trained implicit_als model" if base is not None else "fitted here")
        mlflow.log_params({**model.params(), "train_rows": len(train), "test_fraction": cfg["split"]["test_fraction"],
                           "tuned_on_validation_split": tuned})
        if isinstance(model, HybridRecommender):
            pass                                                # fitted above, on the tuned weights
        elif isinstance(model, ALSMatrixFactorization):
            model.fit(train, ctx, callback=lambda e, rmse: (mlflow.log_metric("train_rmse_epoch", rmse, step=e),
                                                             log.info("[%s] epoch %2d/%d  train RMSE %.4f", name, e, model.n_epochs, rmse)))
        else:
            model.fit(train, ctx)
        res.done()
        metrics = {**model.train_metrics(), "fit_seconds": res.seconds}
        mlflow.log_metrics(metrics)
        model_dir = path_of(cfg, "models", name)
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model.save(model_dir / "model.pkl")
        mlflow.log_artifact(str(path), artifact_path="model")
        for a in extra_artifacts:
            mlflow.log_artifact(str(a), artifact_path="tuning")
        # a human-readable sample: top-5 for the first three users
        titles = ctx.items.set_index("item_id")["title"]
        seen = train.groupby("user_id")["item_id"].agg(set)
        sample = [{"user_id": int(u), "rank": r + 1, "item_id": it, "title": titles.get(it, "?")}
                  for u in sorted(seen.index)[:3] for r, it in enumerate(model.recommend(int(u), 5, seen[u]))]
        sample_path = write_csv(pd.DataFrame(sample), model_dir / "sample_recommendations.csv")
        mlflow.log_artifact(str(sample_path), artifact_path="samples")
        run_id = run.info.run_id

    fmeta_path = Path(cfg["paths"]["features"]) / "feature_metadata.json"
    feature_version = read_json(fmeta_path)["feature_version"] if fmeta_path.exists() else None
    trained_at = utc_now()
    meta_path = write_json({"name": name, "class": type(model).__name__, "params": model.params(), "train_metrics": metrics,
                            "train_seconds": res.seconds, "tuned_on_validation_split": tuned, "mlflow_run_id": run_id,
                            "feature_version": feature_version, "dataset": cfg["dataset"]["source"],
                            "train_rows": int(len(train)), "trained_at": trained_at}, model_dir / "meta.json")
    reg_path = path_of(cfg, "models") / "registry.json"
    with _REGISTRY_LOCK:
        registry = read_json(reg_path) if reg_path.exists() else {}
        registry[name] = {"path": str(path), "mlflow_run_id": run_id, "trained_at": trained_at, "params": model.params(),
                          "train_metrics": metrics, "tuned": tuned}
        write_json(registry, reg_path)
        summary_path = write_training_summary(cfg, registry)
    res.outputs = {"model": str(path), "meta": str(meta_path), "registry": str(reg_path),
                   "training_summary": str(summary_path), "mlflow_run_id": run_id}
    res.metrics = metrics
    log.info("[%s] trained in %.1fs | %s | mlflow run %s", name, res.seconds, metrics, run_id[:8])
    return res


def run_training(cfg: dict | None = None) -> list[StageResult]:
    cfg = cfg or load_config()
    return [train_model(n, cfg) for n in MODEL_NAMES]


if __name__ == "__main__":
    run_training()
