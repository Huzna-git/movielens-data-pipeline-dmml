"""Ingest the public MovieLens 100K dataset (GroupLens) into standard raw CSVs.

The archive holds three pipe/tab separated files that this module converts
into the pipeline's canonical raw layout:

* ``users.csv``        user_id, age, gender, occupation, zip_code
* ``items.csv``        item_id, title, release_date, imdb_url, genres
* ``interactions.csv`` user_id, item_id, rating, timestamp (unix seconds)

Nothing is cleaned here on purpose: ingestion lands the data as it is so the
validation stage has the real thing to check.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
    "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, expected_sha256: str | None, log) -> Path:
    """Download ``url`` to ``dest`` unless a verified copy is already there."""
    if dest.exists() and (expected_sha256 is None or sha256_of(dest) == expected_sha256):
        log.info("archive already present and verified: %s", dest.name)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", url)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    digest = sha256_of(dest)
    if expected_sha256 and digest != expected_sha256:
        raise RuntimeError(f"checksum mismatch for {dest.name}: {digest} != {expected_sha256}")
    log.info("downloaded %s (%.1f MB, sha256 %s...)", dest.name, dest.stat().st_size / 1e6, digest[:12])
    return dest


def _read_member(zf: zipfile.ZipFile, name: str) -> bytes:
    member = next(n for n in zf.namelist() if n.endswith("/" + name) or n == name)
    return zf.read(member)


def extract_tables(archive: Path) -> dict[str, pd.DataFrame]:
    """Parse the three source files into canonical frames (strings untouched)."""
    with zipfile.ZipFile(archive) as zf:
        users = pd.read_csv(
            io.BytesIO(_read_member(zf, "u.user")), sep="|", header=None,
            names=["user_id", "age", "gender", "occupation", "zip_code"], dtype=str,
        )
        item_cols = ["item_id", "title", "release_date", "video_release_date", "imdb_url"] + GENRES
        items = pd.read_csv(
            io.BytesIO(_read_member(zf, "u.item")), sep="|", header=None, names=item_cols,
            encoding="latin-1", dtype=str,
        )
        inter = pd.read_csv(
            io.BytesIO(_read_member(zf, "u.data")), sep="\t", header=None,
            names=["user_id", "item_id", "rating", "timestamp"], dtype=str,
        )
    flags = items[GENRES].fillna("0").astype(int).to_numpy().astype(bool)
    items["genres"] = ["|".join(g for g, on in zip(GENRES, row) if on) for row in flags]
    items = items[["item_id", "title", "release_date", "imdb_url", "genres"]]
    return {"users": users, "items": items, "interactions": inter}
