"""A stage records what it produced: timing, row counts, output files.

Every ``run_*`` function returns a ``StageResult`` and the orchestrator writes
them to ``reports/run_manifest.json`` so a run is auditable end to end.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any

from .io import utc_now


@dataclass
class StageResult:
    stage: str
    started_at: str = field(default_factory=utc_now)
    finished_at: str = ""
    seconds: float = 0.0
    outputs: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    _t0: float = field(default_factory=time.perf_counter, repr=False)

    def done(self) -> "StageResult":
        self.finished_at = utc_now()
        self.seconds = round(time.perf_counter() - self._t0, 2)
        return self

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_t0", None)
        return d
