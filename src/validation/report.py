"""Write the validation report as JSON (machine) and Markdown (human)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..common.io import write_json, utc_now


def build_summary(table_stats: dict, issues: pd.DataFrame, checks_run: list[dict]) -> dict:
    per_check = []
    for c in checks_run:
        n = int(((issues["table"] == c["table"]) & (issues["check"] == c["check"])).sum()) if len(issues) else 0
        status = "PASS" if n == 0 else ("WARN" if c["severity"] == "warning" else "FAIL")
        per_check.append({**c, "failures": n, "status": status})
    return {"generated_at": utc_now(), "tables": table_stats, "checks": per_check}


def write_report(summary: dict, issues: pd.DataFrame, report_dir: Path) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(summary, report_dir / "validation_report.json")
    lines = ["# Data validation report", "", f"Generated: {summary['generated_at']}", ""]
    lines += ["## Tables", "", "| Table | Rows in | Error rows (removed) | Warning rows (kept) | Rows out |",
              "|---|---:|---:|---:|---:|"]
    for name, t in summary["tables"].items():
        lines.append(f"| {name} | {t['rows_in']:,} | {t['error_rows']:,} | {t['warning_rows']:,} | {t['rows_out']:,} |")
    lines += ["", "## Checks", "", "| Table | Check | Column(s) | Severity | Failures | Status |", "|---|---|---|---|---:|---|"]
    for c in summary["checks"]:
        lines.append(f"| {c['table']} | {c['check']} | {c['columns']} | {c['severity']} | {c['failures']:,} | {c['status']} |")
    if len(issues):
        lines += ["", "## Sample of flagged rows (first 25)", "", issues.head(25).to_markdown(index=False)]
    md_path = report_dir / "validation_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
