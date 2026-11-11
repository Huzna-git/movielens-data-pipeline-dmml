"""One logger configuration for every stage: console + ``logs/pipeline.log``."""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_CONFIGURED = False
FORMAT = "%(asctime)s | %(name)-14s | %(levelname)-7s | %(message)s"


def get_logger(name: str, log_dir: str | Path | None = None) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        root = logging.getLogger("pipeline")
        root.setLevel(logging.INFO)
        root.propagate = False
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(FORMAT, "%Y-%m-%d %H:%M:%S"))
        root.addHandler(handler)
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(Path(log_dir) / "pipeline.log", encoding="utf-8")
            fh.setFormatter(logging.Formatter(FORMAT, "%Y-%m-%d %H:%M:%S"))
            root.addHandler(fh)
        _CONFIGURED = True
    return logging.getLogger(f"pipeline.{name}")
