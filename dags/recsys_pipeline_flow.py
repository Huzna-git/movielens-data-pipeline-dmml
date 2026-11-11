"""Orchestration: the end-to-end pipeline as a Prefect flow.

    Ingest -> Validate -> Prepare -> Transform -> Features -> Train (x3, in parallel) -> Evaluate -> Manifest

Each task wraps one stage function from ``src``; the stage result of the
upstream task is passed as an input so Prefect records the dependency, and a
failed stage stops everything downstream.  Run it with

    python dags/recsys_pipeline_flow.py            # one run, local
    make pipeline                                   # same, through the Makefile
    make prefect-server  (then)  make pipeline      # runs appear in the Prefect UI

``deploy_schedule()`` registers a daily schedule (see the Makefile target
``schedule``) so the same flow can be served on an interval.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from prefect import flow, task, get_run_logger  # noqa: E402
from prefect.task_runners import ThreadPoolTaskRunner  # noqa: E402

from src.common.config import load_config  # noqa: E402
from src.common.io import write_json, utc_now  # noqa: E402
from src.evaluation.evaluate import run_evaluation  # noqa: E402
from src.features.build import run_feature_engineering  # noqa: E402
from src.ingestion.ingest import run_ingestion  # noqa: E402
from src.preparation.prepare import run_preparation  # noqa: E402
from src.training.train import MODEL_NAMES, train_model  # noqa: E402
from src.transformation.transform import run_transformation  # noqa: E402
from src.validation.validate import run_validation  # noqa: E402


@task(name="ingest_data", retries=2, retry_delay_seconds=15)
def ingest(cfg: dict) -> dict:
    return run_ingestion(cfg).to_dict()


@task(name="validate_data")
def validate(cfg: dict, upstream: dict) -> dict:
    r = run_validation(cfg)
    if r.metrics["checks_failed"] and cfg.get("validation", {}).get("fail_on_error", False):
        raise RuntimeError(f"{r.metrics['checks_failed']} validation checks failed")
    return r.to_dict()


@task(name="prepare_data")
def prepare(cfg: dict, upstream: dict) -> dict:
    return run_preparation(cfg).to_dict()


@task(name="transform_data")
def transform(cfg: dict, upstream: dict) -> dict:
    return run_transformation(cfg).to_dict()


@task(name="build_features")
def features(cfg: dict, upstream: dict) -> dict:
    return run_feature_engineering(cfg).to_dict()


@task(name="train_model", task_run_name="train_{model_name}")
def train(cfg: dict, model_name: str, upstream: dict) -> dict:
    return train_model(model_name, cfg).to_dict()


@task(name="evaluate_models")
def evaluate(cfg: dict, trained: list[dict]) -> dict:
    return run_evaluation(cfg).to_dict()


@task(name="write_run_manifest")
def write_manifest(cfg: dict, stages: list[dict]) -> str:
    manifest = {"flow": "recommendation-pipeline", "finished_at": utc_now(), "config": cfg["project"]["name"],
                "dataset": cfg["dataset"]["source"], "stages": stages}
    path = write_json(manifest, Path(cfg["paths"]["reports"]) / "run_manifest.json")
    return str(path)


@flow(name="recommendation-pipeline", log_prints=True, task_runner=ThreadPoolTaskRunner(max_workers=3))
def recommendation_pipeline(config_path: str | None = None) -> dict:
    log = get_run_logger()
    cfg = load_config(config_path)
    log.info("pipeline start: dataset=%s split=%s", cfg["dataset"]["source"], cfg["split"]["strategy"])
    s1 = ingest(cfg)
    s2 = validate(cfg, s1)
    s3 = prepare(cfg, s2)
    s4 = transform(cfg, s3)
    s5 = features(cfg, s4)
    futures = [train.submit(cfg, name, s5) for name in MODEL_NAMES]   # fan-out: three models in parallel
    trained = [f.result() for f in futures]                            # fan-in
    s7 = evaluate(cfg, trained)
    manifest = write_manifest(cfg, [s1, s2, s3, s4, s5, *trained, s7])
    best = max(s7["metrics"].items(), key=lambda kv: kv[1][f"ndcg_at_{cfg['evaluation']['k']}"])[0]
    log.info("pipeline finished; best model by NDCG: %s; manifest %s", best, manifest)
    return {"best_model": best, "manifest": manifest, "evaluation": s7["metrics"]}


def deploy_schedule(cron: str = "0 2 * * *") -> None:
    """Serve the flow on a schedule (daily 02:00 by default)."""
    recommendation_pipeline.serve(name="recommendation-pipeline-daily", cron=cron)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        deploy_schedule()
    else:
        recommendation_pipeline()
