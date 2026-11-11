# Recommendation data pipeline - entry points.  `make help` lists them.
PY ?= python
CONFIG ?= configs/pipeline.yaml
export MLFLOW_DISABLE_AGENT_HINT=1

.PHONY: help setup pipeline stages ingest validate prepare transform features train evaluate \
        test mlflow-ui prefect-server schedule synthetic report clean

help:            ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup:           ## create a virtualenv in .venv and install the requirements
	$(PY) -m venv .venv && .venv/bin/pip install -r requirements.txt

pipeline:        ## run the whole pipeline as the Prefect flow (orchestrated)
	$(PY) dags/recsys_pipeline_flow.py

stages:          ## run every stage in order WITHOUT Prefect (plain Python)
	$(PY) -m src.cli all --config $(CONFIG)

ingest validate prepare transform features train evaluate:   ## run one stage
	$(PY) -m src.cli $@ --config $(CONFIG)

synthetic:       ## run the whole pipeline on the documented synthetic dataset (with injected noise)
	$(PY) -m src.cli all --config configs/pipeline.synthetic.yaml

test:            ## unit tests (validation, split, metrics, features, models)
	$(PY) -m pytest -q tests

mlflow-ui:       ## MLflow experiment tracking UI on http://127.0.0.1:5000
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

prefect-server:  ## Prefect orchestration UI on http://127.0.0.1:4200 (then `make pipeline` in another shell)
	prefect server start

schedule:        ## serve the flow on a daily schedule (02:00) through Prefect
	$(PY) dags/recsys_pipeline_flow.py serve

report:          ## rebuild reports/final_report.pdf from reports/final_report.md (needs Chrome)
	$(PY) scripts/build_report.py

clean:           ## remove generated data, models and tracking state
	rm -rf data/raw/* data/processed/* data/features/* data/errors/* models/* \
	       mlflow.db mlartifacts mlruns logs reports/per_user
	touch data/raw/.gitkeep data/processed/.gitkeep data/features/.gitkeep data/errors/.gitkeep models/.gitkeep
