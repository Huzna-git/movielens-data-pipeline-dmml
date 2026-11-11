"""Configuration loading.

Every stage reads the same YAML file (configs/pipeline.yaml) through
``load_config``.  Relative paths in the file are resolved against the project
root so a stage behaves identically whether it is run from the project root,
from a Prefect worker or from a test.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "pipeline.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Load the pipeline configuration and resolve every path under ``paths``.

    ``PIPELINE_CONFIG`` in the environment overrides the default file, which
    is how the tests point the stages at a temporary directory.
    """
    cfg_path = Path(path or os.environ.get("PIPELINE_CONFIG", DEFAULT_CONFIG))
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    root = Path(os.environ.get("PIPELINE_ROOT", PROJECT_ROOT))
    cfg["project"]["root"] = str(root)
    for key, value in cfg["paths"].items():
        p = Path(value)
        cfg["paths"][key] = str(p if p.is_absolute() else root / p)
    return cfg


def path_of(cfg: dict[str, Any], key: str, *parts: str) -> Path:
    """Return ``cfg['paths'][key]`` joined with ``parts``, creating the folder."""
    base = Path(cfg["paths"][key])
    base.mkdir(parents=True, exist_ok=True)
    return base.joinpath(*parts) if parts else base
