"""Stage 2 - Data validation.

Reads the raw CSVs as strings, coerces types, runs the pandera schemas and the
row-level checks, writes the invalid rows to ``data/errors`` with the reason,
the clean rows to ``data/processed/<table>_validated.parquet`` and the report
to ``reports/validation_report.{json,md}``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..common.config import load_config, path_of
from ..common.io import write_csv, write_parquet
from ..common.logging import get_logger
from ..common.stage import StageResult
from . import checks as C
from .report import build_summary, write_report
from .schemas import TYPE_SPECS, interactions_schema, items_schema, users_schema

KEYS = {"users": ["user_id"], "items": ["item_id"], "interactions": ["user_id", "item_id"]}


def _checks_catalog(cfg: dict) -> list[dict]:
    v = cfg["validation"]
    rows = []
    for t, spec in TYPE_SPECS.items():
        rows.append({"table": t, "check": "invalid_type", "columns": ",".join(spec), "severity": "error"})
        rows.append({"table": t, "check": "missing_value", "columns": "non-nullable columns", "severity": "error"})
        rows.append({"table": t, "check": "duplicate_row", "columns": "all", "severity": "error"})
        rows.append({"table": t, "check": "conflicting_duplicate_key", "columns": ",".join(KEYS[t]), "severity": "error"})
    rows += [
        {"table": "users", "check": "duplicate_key", "columns": "user_id", "severity": "error"},
        {"table": "users", "check": "out_of_range", "columns": f"age in [{v['user_age_min']}, {v['user_age_max']}]", "severity": "error"},
        {"table": "users", "check": "invalid_category", "columns": "gender in {M,F}", "severity": "error"},
        {"table": "items", "check": "duplicate_key", "columns": "item_id", "severity": "error"},
        {"table": "items", "check": "unparseable_date", "columns": "release_date (dd-Mon-yyyy)", "severity": "error"},
        {"table": "items", "check": "missing_release_date", "columns": "release_date", "severity": "warning"},
        {"table": "items", "check": "duplicate_title", "columns": "title", "severity": "warning"},
        {"table": "interactions", "check": "out_of_range", "columns": f"rating in [{v['rating_min']}, {v['rating_max']}]; timestamp in [{v['timestamp_min']}, {v['timestamp_max']}]", "severity": "error"},
        {"table": "interactions", "check": "unknown_user", "columns": "user_id -> users.user_id", "severity": "error"},
        {"table": "interactions", "check": "unknown_item", "columns": "item_id -> items.item_id", "severity": "error"},
    ]
    return rows


def validate_tables(cfg: dict, raw: dict[str, pd.DataFrame], log=None) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """Validate the three raw frames.  Returns (clean frames, issues, table stats).

    Pure function of its inputs so the tests can feed it corrupted frames.
    """
    schemas = {"users": users_schema(cfg), "items": items_schema(cfg), "interactions": interactions_schema(cfg)}
    issues: list[C.Issue] = []
    coerced: dict[str, pd.DataFrame] = {}
    for name in ("users", "items", "interactions"):
        df, type_issues = C.coerce_types(name, raw[name], TYPE_SPECS[name])
        coerced[name] = df
        issues += type_issues
        issues += C.schema_issues(name, df, schemas[name])
        issues += C.duplicate_issues(name, df, KEYS[name])
    issues += C.date_parse_issues("items", coerced["items"], "release_date", "%d-%b-%Y")
    issues += C.duplicate_title_warnings("items", coerced["items"])

    # Referential integrity is checked against the users/items that are themselves valid.
    err_rows = {t: {i.row for i in issues if i.table == t and i.severity == "error"} for t in coerced}
    valid_users = coerced["users"].loc[~coerced["users"].index.isin(err_rows["users"]), "user_id"]
    valid_items = coerced["items"].loc[~coerced["items"].index.isin(err_rows["items"]), "item_id"]
    issues += C.referential_issues("interactions", coerced["interactions"], "user_id", valid_users, "unknown_user")
    issues += C.referential_issues("interactions", coerced["interactions"], "item_id", valid_items, "unknown_item")

    issues_df = pd.DataFrame([i.to_dict() for i in issues],
                             columns=["table", "row", "column", "check", "severity", "value"])
    clean, stats = {}, {}
    for name, df in coerced.items():
        t = issues_df[issues_df["table"] == name]
        bad = set(t.loc[t["severity"] == "error", "row"])
        warn = set(t.loc[t["severity"] == "warning", "row"]) - bad
        clean[name] = df.loc[~df.index.isin(bad)].reset_index(drop=True)
        stats[name] = {"rows_in": int(len(df)), "error_rows": len(bad), "warning_rows": len(warn),
                       "rows_out": int(len(clean[name]))}
        if log:
            log.info("%-13s rows in %7d | errors %5d | warnings %5d | rows out %7d",
                     name, len(df), len(bad), len(warn), len(clean[name]))
    return clean, issues_df, stats


def run_validation(cfg: dict | None = None) -> StageResult:
    cfg = cfg or load_config()
    log = get_logger("validation", cfg["paths"]["logs"])
    res = StageResult("validation")
    raw_dir, err_dir, proc_dir = path_of(cfg, "raw"), path_of(cfg, "errors"), path_of(cfg, "processed")
    raw = {n: pd.read_csv(raw_dir / f"{n}.csv", dtype=str, keep_default_na=True) for n in ("users", "items", "interactions")}
    log.info("loaded raw tables: %s", {n: len(df) for n, df in raw.items()})

    clean, issues, stats = validate_tables(cfg, raw, log)

    # Error dataset: the raw rows that were rejected, with every reason joined.
    for name, df in raw.items():
        t = issues[(issues["table"] == name) & (issues["severity"] == "error")]
        if len(t):
            reasons = t.groupby("row")["check"].agg(lambda s: ";".join(sorted(set(s))))
            bad = df.loc[reasons.index].copy()
            bad.insert(0, "reasons", reasons.values)
            bad.insert(0, "source_row", reasons.index)
            res.outputs[f"{name}_errors"] = str(write_csv(bad, err_dir / f"{name}_errors.csv"))
    res.outputs["issues"] = str(write_csv(issues, err_dir / "validation_issues.csv"))
    for name, df in clean.items():
        res.outputs[name] = str(write_parquet(df, proc_dir / f"{name}_validated.parquet"))

    summary = build_summary(stats, issues, _checks_catalog(cfg))
    jp, mp = write_report(summary, issues, path_of(cfg, "reports"))
    res.outputs["report_json"], res.outputs["report_md"] = str(jp), str(mp)
    failed = [c for c in summary["checks"] if c["status"] == "FAIL"]
    warned = [c for c in summary["checks"] if c["status"] == "WARN"]
    for c in summary["checks"]:
        if c["status"] != "PASS":
            log.warning("%-5s %-13s %-26s %6d rows", c["status"], c["table"], c["check"], c["failures"])
    res.metrics = {**{f"{t}_rows_out": s["rows_out"] for t, s in stats.items()},
                   "checks_run": len(summary["checks"]), "checks_failed": len(failed), "checks_warned": len(warned),
                   "error_rows": int(sum(s["error_rows"] for s in stats.values()))}
    log.info("validation complete: %d checks, %d FAIL, %d WARN, %d error rows -> %s",
             len(summary["checks"]), len(failed), len(warned), res.metrics["error_rows"], mp)
    return res.done()


if __name__ == "__main__":
    run_validation()
