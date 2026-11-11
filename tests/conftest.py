import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from src.common.config import load_config  # noqa: E402


@pytest.fixture
def cfg(tmp_path):
    """The real config with every path redirected to a temporary folder."""
    c = load_config(ROOT / "configs" / "pipeline.yaml")
    for key in c["paths"]:
        c["paths"][key] = str(tmp_path / key)
    c["project"]["root"] = str(tmp_path)
    return c
