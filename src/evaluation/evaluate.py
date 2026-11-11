"""Stage 7 - Model evaluation.

For every trained model: top-K recommendations for each test user (the items
they rated in train are excluded), Precision@K, Recall@K, NDCG@K, Coverage and
Diversity; RMSE and MAE over the held-out ratings for the models that predict
explicit ratings.  Metrics go to the model's MLflow run and to
``reports/model_evaluation.{md,csv}`` plus a comparison chart.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from ..common.config import load_config, path_of
from ..common.io import read_json, read_parquet, write_csv, write_json
from ..common.logging import get_logger
from ..common.stage import StageResult
from ..training.base import Recommender
from ..training.train import setup_mlflow
from ..transformation.transform import GENRE_COLUMNS
from . import metrics as M


def evaluate_model(model: Recommender, train: pd.DataFrame, test: pd.DataFrame, item_vectors: dict[int, np.ndarray],
                   catalog_size: int, k: int, relevance_threshold: float) -> tuple[dict, pd.DataFrame]:
    seen = train.groupby("user_id")["item_id"].agg(set)
    relevant = test[test["rating"] >= relevance_threshold].groupby("user_id")["item_id"].agg(set)
    rows, all_recs = [], []
    for u, rel in relevant.items():
        recs = model.recommend(int(u), k, seen.get(u, set()))
        all_recs.append(recs)
        rows.append({"user_id": int(u), "n_relevant": len(rel), "n_recommended": len(recs),
                     "precision": M.precision_at_k(recs, rel, k), "recall": M.recall_at_k(recs, rel, k),
                     "ndcg": M.ndcg_at_k(recs, rel, k), "diversity": M.intra_list_diversity(recs, item_vectors)})
    per_user = pd.DataFrame(rows)
    out = {f"precision_at_{k}": per_user["precision"].mean(), f"recall_at_{k}": per_user["recall"].mean(),
           f"ndcg_at_{k}": per_user["ndcg"].mean(), "coverage": M.coverage(all_recs, catalog_size),
           "diversity": per_user["diversity"].mean(), "users_evaluated": int(len(per_user)),
           "users_with_empty_list": int((per_user["n_recommended"] == 0).sum())}
    pred = model.predict_rating(test["user_id"].to_numpy(), test["item_id"].to_numpy())
    if pred is not None:
        out["rmse"] = M.rmse(test["rating"].to_numpy(), pred)
        out["mae"] = M.mae(test["rating"].to_numpy(), pred)
    return {k_: (round(float(v), 4) if isinstance(v, float) else v) for k_, v in out.items()}, per_user


def write_results(results: pd.DataFrame, k: int, reports: Path, cfg: dict) -> tuple[Path, Path, Path]:
    cols = [f"precision_at_{k}", f"recall_at_{k}", f"ndcg_at_{k}", "coverage", "diversity", "rmse", "mae"]
    table = results.set_index("model")[cols]
    shown = table.astype(object).where(table.notna(), None)   # tabulate prints NaN as "nan"; None honours missingval
    csv = write_csv(results, reports / "model_evaluation.csv")
    best = table[f"ndcg_at_{k}"].idxmax()
    md = [f"# Model evaluation (K = {k})", "",
          f"Dataset: {cfg['dataset']['source']} · split: {cfg['split']['strategy']} (test fraction {cfg['split']['test_fraction']}) · "
          f"a held-out rating >= {cfg['evaluation']['relevance_threshold']} is relevant · items rated in train are excluded from each user's list.", "",
          shown.to_markdown(floatfmt=".4f", missingval="n/a"), "",
          f"Users evaluated: {int(results['users_evaluated'].iloc[0])} (test users with at least one relevant held-out item).", "",
          f"**Best ranking model by NDCG@{k}: `{best}`.**", "",
          "Precision/Recall/NDCG@K are averaged over users. Coverage = share of the catalogue recommended to at least one user. "
          "Diversity = 1 - mean pairwise cosine similarity of genre vectors within a user's list, averaged over users. "
          "RMSE/MAE are over all held-out ratings for the models that predict explicit ratings.", ""]
    md_path = reports / "model_evaluation.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    fig_dir = reports / "figures"
    fig_dir.mkdir(exist_ok=True)
    rank_cols = cols[:5]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [5, 2]})
    x = np.arange(len(rank_cols)); w = 0.8 / len(table)
    palette = ["#6B7280", "#2563EB", "#16A34A", "#D97706"]
    for n, (name, row) in enumerate(table.iterrows()):
        axes[0].bar(x + n * w - 0.4 + w / 2, row[rank_cols].to_numpy(dtype=float), w, label=name, color=palette[n % len(palette)])
    axes[0].set_xticks(x); axes[0].set_xticklabels([c.replace("_at_", "@") for c in rank_cols]); axes[0].set_ylim(0, 1)
    axes[0].set_title(f"Ranking metrics (K={k})"); axes[0].legend(frameon=False, fontsize=8); axes[0].grid(axis="y", alpha=.3)
    err = table[["rmse", "mae"]].dropna()
    for n, (name, row) in enumerate(err.iterrows()):
        axes[1].bar(np.arange(2) + n * w - 0.4 + w / 2, row.to_numpy(dtype=float), w, color=palette[list(table.index).index(name) % len(palette)])
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["RMSE", "MAE"]); axes[1].set_title("Rating error"); axes[1].grid(axis="y", alpha=.3)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig_path = fig_dir / "metrics_comparison.png"
    fig.savefig(fig_path, dpi=150); plt.close(fig)
    return csv, md_path, fig_path


def run_evaluation(cfg: dict | None = None) -> StageResult:
    cfg = cfg or load_config()
    log = get_logger("evaluation", cfg["paths"]["logs"])
    res = StageResult("evaluation")
    setup_mlflow(cfg)
    proc, reports = Path(cfg["paths"]["processed"]), path_of(cfg, "reports")
    k, thr = cfg["evaluation"]["k"], cfg["evaluation"]["relevance_threshold"]
    train, test = read_parquet(proc / "train.parquet"), read_parquet(proc / "test.parquet")
    items, ig = read_parquet(proc / "items.parquet"), read_parquet(proc / "item_genres.parquet")
    item_vectors = dict(zip(ig["item_id"].astype(int).tolist(), ig[GENRE_COLUMNS].to_numpy(dtype=float)))
    registry = read_json(path_of(cfg, "models") / "registry.json")
    log.info("K=%d, relevance threshold=%.1f, test rows=%d, test users=%d, catalogue=%d items",
             k, thr, len(test), test["user_id"].nunique(), len(items))

    results = []
    for name, entry in registry.items():
        model = Recommender.load(Path(entry["path"]))
        m, per_user = evaluate_model(model, train, test, item_vectors, len(items), k, thr)
        write_csv(per_user, reports / "per_user" / f"{name}_per_user_metrics.csv")
        with mlflow.start_run(run_id=entry["mlflow_run_id"]):
            mlflow.log_metrics({k_: v for k_, v in m.items() if isinstance(v, (int, float))})
            mlflow.set_tag("stage", "evaluated")
        results.append({"model": name, **m})
        log.info("%-26s P@%d %.4f | R@%d %.4f | NDCG@%d %.4f | coverage %.4f | diversity %.4f | RMSE %s | MAE %s",
                 name, k, m[f"precision_at_{k}"], k, m[f"recall_at_{k}"], k, m[f"ndcg_at_{k}"], m["coverage"], m["diversity"],
                 f"{m['rmse']:.4f}" if "rmse" in m else "n/a", f"{m['mae']:.4f}" if "mae" in m else "n/a")

    df = pd.DataFrame(results)
    csv, md, fig = write_results(df, k, reports, cfg)
    write_json({"k": k, "relevance_threshold": thr, "results": results}, reports / "model_evaluation.json")
    with mlflow.start_run(run_name="evaluation_summary"):
        mlflow.set_tags({"stage": "evaluation", "k": str(k)})
        mlflow.log_artifact(str(csv)); mlflow.log_artifact(str(md)); mlflow.log_artifact(str(fig))
    res.outputs = {"csv": str(csv), "markdown": str(md), "figure": str(fig)}
    res.metrics = {r["model"]: {k_: v for k_, v in r.items() if k_ != "model"} for r in results}
    best = df.set_index("model")[f"ndcg_at_{k}"].idxmax()
    log.info("best model by NDCG@%d: %s", k, best)
    log.info("evaluation complete -> %s", md)
    return res.done()


if __name__ == "__main__":
    run_evaluation()
