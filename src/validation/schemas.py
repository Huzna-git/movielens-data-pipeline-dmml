"""Pandera schemas: the declarative part of validation (types, ranges, nulls, keys).

Row-level checks pandera cannot express (referential integrity between tables,
duplicate keys with conflicting values, type coercion failures) live in
``checks.py``.  Both feed the same error dataset.
"""
from __future__ import annotations

import pandas as pd
import pandera.pandas as pa


def users_schema(cfg: dict) -> pa.DataFrameSchema:
    v = cfg["validation"]
    return pa.DataFrameSchema(
        {
            "user_id": pa.Column(pd.Int64Dtype(), pa.Check.ge(1), nullable=False, unique=True,
                         report_duplicates="exclude_first"),   # the first copy of a key is the record; later copies are the errors
            "age": pa.Column(pd.Int64Dtype(), pa.Check.in_range(v["user_age_min"], v["user_age_max"]), nullable=False),
            "gender": pa.Column(str, pa.Check.isin(["M", "F", "m", "f"]), nullable=False),
            "occupation": pa.Column(str, pa.Check.str_length(min_value=1), nullable=False),
            "zip_code": pa.Column(str, nullable=True),
        },
        name="users",
    )


def items_schema(cfg: dict) -> pa.DataFrameSchema:
    return pa.DataFrameSchema(
        {
            "item_id": pa.Column(pd.Int64Dtype(), pa.Check.ge(1), nullable=False, unique=True,
                         report_duplicates="exclude_first"),   # the first copy of a key is the record; later copies are the errors
            "title": pa.Column(str, pa.Check.str_length(min_value=1), nullable=False),
            "release_date": pa.Column(str, nullable=True),
            "imdb_url": pa.Column(str, nullable=True),
            "genres": pa.Column(str, nullable=True),
        },
        name="items",
    )


def interactions_schema(cfg: dict) -> pa.DataFrameSchema:
    v = cfg["validation"]
    ts_min = int(pd.Timestamp(v["timestamp_min"], tz="UTC").timestamp())
    ts_max = int(pd.Timestamp(v["timestamp_max"], tz="UTC").timestamp())
    return pa.DataFrameSchema(
        {
            "user_id": pa.Column(pd.Int64Dtype(), pa.Check.ge(1), nullable=False),
            "item_id": pa.Column(pd.Int64Dtype(), pa.Check.ge(1), nullable=False),
            "rating": pa.Column(pd.Float64Dtype(), pa.Check.in_range(v["rating_min"], v["rating_max"]), nullable=False),
            "timestamp": pa.Column(pd.Int64Dtype(), pa.Check.in_range(ts_min, ts_max), nullable=False),
        },
        name="interactions",
    )


# Which columns each raw table must carry and what type they must coerce to.
TYPE_SPECS = {
    "users": {"user_id": "int", "age": "int", "gender": "str", "occupation": "str", "zip_code": "str"},
    "items": {"item_id": "int", "title": "str", "release_date": "str", "imdb_url": "str", "genres": "str"},
    "interactions": {"user_id": "int", "item_id": "int", "rating": "float", "timestamp": "int"},
}
