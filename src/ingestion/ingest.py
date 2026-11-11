"""Stage 1 - Data ingestion.

Lands user, item and interaction data in ``data/raw`` as CSV together with an
ingestion manifest (source, checksum, row counts, timestamp).
"""
from __future__ import annotations

from pathlib import Path

from ..common.config import load_config, path_of
from ..common.io import write_csv, write_json, utc_now
from ..common.logging import get_logger
from ..common.stage import StageResult
from . import movielens, synthetic


def run_ingestion(cfg: dict | None = None) -> StageResult:
    cfg = cfg or load_config()
    log = get_logger("ingestion", cfg["paths"]["logs"])
    res = StageResult("ingestion")
    raw = path_of(cfg, "raw")
    source = cfg["dataset"]["source"]
    log.info("source = %s", source)

    manifest = {"source": source, "ingested_at": utc_now()}
    if source == "movielens-100k":
        ml = cfg["dataset"]["movielens"]
        archive = movielens.download(ml["url"], raw / "ml-100k.zip", ml.get("sha256"), log)
        tables = movielens.extract_tables(archive)
        manifest.update({"url": ml["url"], "sha256": movielens.sha256_of(archive)})
    elif source == "synthetic":
        s = cfg["dataset"]["synthetic"]
        tables, corruption = synthetic.generate(
            s["n_users"], s["n_items"], s["n_interactions"], s["noise_rate"], cfg["project"]["random_seed"]
        )
        manifest["injected_corruption"] = corruption
        log.info("synthetic corruption injected: %s", corruption)
    else:
        raise ValueError(f"unknown dataset.source: {source}")

    counts = {}
    for name, df in tables.items():
        out = write_csv(df, raw / f"{name}.csv")
        counts[name] = len(df)
        res.outputs[name] = str(out)
        log.info("wrote %-13s %7d rows x %d cols -> %s", name, len(df), df.shape[1], out.name)
    manifest["row_counts"] = counts
    manifest["columns"] = {n: list(df.columns) for n, df in tables.items()}
    res.outputs["manifest"] = str(write_json(manifest, raw / "ingestion_manifest.json"))
    res.metrics = counts
    log.info("ingestion complete: %s", counts)
    return res.done()


if __name__ == "__main__":
    run_ingestion()
