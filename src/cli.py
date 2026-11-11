"""Run one stage, or all of them in order, without the orchestrator:

    python -m src.cli ingest | validate | prepare | transform | features | train | evaluate | all
"""
from __future__ import annotations

import argparse
import json

from .common.config import load_config
from .evaluation.evaluate import run_evaluation
from .features.build import run_feature_engineering
from .ingestion.ingest import run_ingestion
from .preparation.prepare import run_preparation
from .training.train import run_training
from .transformation.transform import run_transformation
from .validation.validate import run_validation

STAGES = {
    "ingest": run_ingestion, "validate": run_validation, "prepare": run_preparation, "transform": run_transformation,
    "features": run_feature_engineering, "train": run_training, "evaluate": run_evaluation,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Recommendation data pipeline")
    ap.add_argument("stage", choices=list(STAGES) + ["all"])
    ap.add_argument("--config", default=None, help="path to pipeline.yaml (default configs/pipeline.yaml)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    names = list(STAGES) if args.stage == "all" else [args.stage]
    for n in names:
        out = STAGES[n](cfg)
        results = out if isinstance(out, list) else [out]
        for r in results:
            print(json.dumps({"stage": r.stage, "seconds": r.seconds, "metrics": r.metrics}, default=str))


if __name__ == "__main__":
    main()
