"""Tiny local cache so we don't re-hit the NBA API. Parquet under data/cache/."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CACHE_DIR = Path("data/cache")


def cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.parquet"


def save_parquet(df: pd.DataFrame, name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(name)
    df.to_parquet(path, index=False)
    return path


def load_parquet(name: str) -> pd.DataFrame | None:
    path = cache_path(name)
    if not path.exists():
        return None
    return pd.read_parquet(path)
