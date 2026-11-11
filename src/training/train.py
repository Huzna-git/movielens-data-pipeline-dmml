"""Stage 6 - Model training with MLflow experiment tracking.

One MLflow run per model: parameters, training metrics (per-epoch RMSE for
the factorization model), the pickled model and a sample of recommendations as
artifacts.  ``models/registry.json`` maps each model to its run id so the
evaluation stage can log its metrics into the same run.
"""
from __future__ import annotations

import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from ..common.config import load_config, path_of
from ..common.io import read_json, read_parquet, utc_now, write_csv, write_json
from ..common.logging import get_logger
from ..common.stage import StageResult
from ..features.store import FeatureStore
from ..preparation.prepare import time_split
from ..evaluation.metrics import ndcg_at_k, precision_at_k
from .base import Recommender, TrainingContext
from .content_based import ContentBasedRecommender
from .implicit_als import ImplicitALS
from .matrix_factorization import ALSMatrixFactorization
from .popularity import PopularityRecommender

MODEL_NAMES = ["popularity", "implicit_als", "als_matrix_factorization", "content_based_genres"]


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
        mlflow.log_params({**model.params(), "train_rows": len(train), "test_fraction": cfg["split"]["test_fraction"],
                           "tuned_on_validation_split": tuned})
        if isinstance(model, ALSMatrixFactorization):
            model.fit(train, ctx, callback=lambda e, rmse: (mlflow.log_metric("train_rmse_epoch", rmse, step=e),
                                                             log.info("[%s] epoch %2d/%d  train RMSE %.4f", name, e, model.n_epochs, rmse)))
        else:
            model.fit(train, ctx)
        res.done()
        metrics = {**model.train_metrics(), "fit_seconds": res.seconds}
        mlflow.log_metrics(metrics)
        path = model.save(path_of(cfg, "models") / f"{name}.pkl")
        mlflow.log_artifact(str(path), artifact_path="model")
        # a human-readable sample: top-5 for the first three users
        titles = ctx.items.set_index("item_id")["title"]
        seen = train.groupby("user_id")["item_id"].agg(set)
        sample = [{"user_id": int(u), "rank": r + 1, "item_id": it, "title": titles.get(it, "?")}
                  for u in sorted(seen.index)[:3] for r, it in enumerate(model.recommend(int(u), 5, seen[u]))]
        sample_path = write_csv(pd.DataFrame(sample), path_of(cfg, "models") / f"{name}_sample_recommendations.csv")
        mlflow.log_artifact(str(sample_path), artifact_path="samples")
        run_id = run.info.run_id

    reg_path = path_of(cfg, "models") / "registry.json"
    registry = read_json(reg_path) if reg_path.exists() else {}
    registry[name] = {"path": str(path), "mlflow_run_id": run_id, "trained_at": utc_now(), "params": model.params(),
                      "train_metrics": metrics}
    write_json(registry, reg_path)
    res.outputs = {"model": str(path), "registry": str(reg_path), "mlflow_run_id": run_id}
    res.metrics = metrics
    log.info("[%s] trained in %.1fs | %s | mlflow run %s", name, res.seconds, metrics, run_id[:8])
    return res


def run_training(cfg: dict | None = None) -> list[StageResult]:
    cfg = cfg or load_config()
    return [train_model(n, cfg) for n in MODEL_NAMES]


if __name__ == "__main__":
    run_training()
