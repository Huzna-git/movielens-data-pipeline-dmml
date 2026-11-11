"""Row-level validation checks.  Each returns a list of ``Issue`` records.

An ``Issue`` names the table, the row, the column, the check that failed and
its severity: ``error`` rows are removed from the clean output and written to
the error dataset; ``warning`` rows are kept but reported.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd
import pandera.pandas as pa


@dataclass
class Issue:
    table: str
    row: int
    column: str
    check: str
    severity: str  # "error" | "warning"
    value: str

    def to_dict(self) -> dict:
        return asdict(self)


def coerce_types(table: str, df: pd.DataFrame, spec: dict[str, str]) -> tuple[pd.DataFrame, list[Issue]]:
    """Coerce raw string columns to their declared types; a value that cannot be
    coerced is an ``invalid_type`` error on its row (the cell becomes <NA>)."""
    issues: list[Issue] = []
    out = pd.DataFrame(index=df.index)
    for col, kind in spec.items():
        if col not in df.columns:
            raise KeyError(f"{table}: required column '{col}' is missing from the raw data")
        raw = df[col]
        if kind in ("int", "float"):
            num = pd.to_numeric(raw, errors="coerce")
            bad = raw.notna() & num.isna()
            if kind == "int":
                non_integral = num.notna() & (num != np.floor(num))
                bad = bad | non_integral
                num = num.where(~non_integral)
            for idx in df.index[bad]:
                issues.append(Issue(table, int(idx), col, "invalid_type", "error", str(raw.loc[idx])))
            out[col] = num.astype(pd.Int64Dtype()) if kind == "int" else num.astype(pd.Float64Dtype())
        else:
            s = raw.astype("string").str.strip()
            out[col] = s.mask(s == "", pd.NA)
    return out, issues


def schema_issues(table: str, df: pd.DataFrame, schema: pa.DataFrameSchema) -> list[Issue]:
    """Run the pandera schema lazily and translate every failure case to an Issue."""
    try:
        schema.validate(df, lazy=True)
        return []
    except pa.errors.SchemaErrors as exc:
        fc = exc.failure_cases
    issues: list[Issue] = []
    for _, r in fc.iterrows():
        if pd.isna(r.get("index")):
            continue  # a column-level failure (dtype) is already caught by coerce_types
        check = str(r["check"])
        if check.startswith("not_nullable"):
            name = "missing_value"
        elif "unique" in check:
            name = "duplicate_key"
        elif check.startswith("in_range") or check.startswith("greater_than") or check.startswith("less_than"):
            name = f"out_of_range"
        elif check.startswith("isin"):
            name = "invalid_category"
        elif check.startswith("str_length"):
            name = "empty_string"
        else:
            name = check.split("(")[0]
        issues.append(Issue(table, int(r["index"]), str(r["column"]), name, "error", str(r["failure_case"])))
    return issues


def duplicate_issues(table: str, df: pd.DataFrame, key: list[str]) -> list[Issue]:
    """Exact duplicate rows (second and later copies) are ``duplicate_row`` errors;
    rows that repeat the key with different content are ``conflicting_duplicate_key``."""
    issues: list[Issue] = []
    exact = df.duplicated(keep="first")
    for idx in df.index[exact]:
        issues.append(Issue(table, int(idx), ",".join(key), "duplicate_row", "error", "exact copy of an earlier row"))
    rest = df[~exact]
    conflict = rest.duplicated(subset=key, keep="first")
    for idx in rest.index[conflict]:
        issues.append(Issue(table, int(idx), ",".join(key), "conflicting_duplicate_key", "error",
                            " / ".join(str(rest.loc[idx, k]) for k in key)))
    return issues


def referential_issues(table: str, df: pd.DataFrame, column: str, valid: Iterable, check: str) -> list[Issue]:
    valid_set = set(pd.Series(list(valid)).dropna().astype("int64").tolist())
    col = df[column]
    bad = col.notna() & ~col.isin(valid_set)
    return [Issue(table, int(idx), column, check, "error", str(col.loc[idx])) for idx in df.index[bad]]


def date_parse_issues(table: str, df: pd.DataFrame, column: str, fmt: str) -> list[Issue]:
    """A date that is present but does not parse is an error; a missing date is
    a warning (the row is kept and the feature layer treats the year as unknown)."""
    issues: list[Issue] = []
    raw = df[column]
    parsed = pd.to_datetime(raw, format=fmt, errors="coerce")
    for idx in df.index[raw.notna() & parsed.isna()]:
        issues.append(Issue(table, int(idx), column, "unparseable_date", "error", str(raw.loc[idx])))
    for idx in df.index[raw.isna()]:
        issues.append(Issue(table, int(idx), column, "missing_release_date", "warning", "<NA>"))
    return issues


def duplicate_title_warnings(table: str, df: pd.DataFrame) -> list[Issue]:
    dup = df.duplicated(subset=["title"], keep=False) & df["title"].notna()
    return [Issue(table, int(idx), "title", "duplicate_title", "warning", str(df.loc[idx, "title"]))
            for idx in df.index[dup]]
