"""Local parquet store: one table per dataset, upsert-by-key semantics."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from cfb.config import CONFIG, Config

log = logging.getLogger(__name__)

# Primary keys used to de-duplicate on write.
TABLE_KEYS: dict[str, list[str]] = {
    "games": ["game_id"],
    "lines": ["game_id", "provider"],
    "team_games": ["game_id", "team"],
    "advanced_season": ["season", "team"],
    "sp_ratings": ["season", "team"],
    "talent": ["season", "team"],
    "returning": ["season", "team"],
    "plays": ["play_id"],
    "drives": ["drive_id"],
    "pbp_states": ["play_id"],
}


class Store:
    """Thin parquet warehouse under ``<data_dir>/processed``."""

    def __init__(self, cfg: Config | None = None, subdir: str = "processed"):
        self.cfg = cfg or CONFIG
        self.root: Path = self.cfg.data_dir / subdir
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, table: str) -> Path:
        return self.root / f"{table}.parquet"

    def exists(self, table: str) -> bool:
        return self.path(table).exists()

    def read(self, table: str, default_empty: bool = True) -> pd.DataFrame:
        p = self.path(table)
        if not p.exists():
            if default_empty:
                return pd.DataFrame()
            raise FileNotFoundError(f"No such table: {p}")
        return pd.read_parquet(p)

    def write(self, table: str, df: pd.DataFrame) -> Path:
        p = self.path(table)
        df.to_parquet(p, index=False)
        log.info("wrote %s rows=%d -> %s", table, len(df), p)
        return p

    def upsert(self, table: str, df: pd.DataFrame, keys: list[str] | None = None) -> pd.DataFrame:
        """Merge ``df`` into the table, newest rows winning on key collision."""
        if df is None or df.empty:
            return self.read(table)
        keys = keys or TABLE_KEYS.get(table) or []
        existing = self.read(table)
        if existing.empty:
            merged = df.copy()
        else:
            merged = pd.concat([existing, df], ignore_index=True)
        present = [k for k in keys if k in merged.columns]
        if present:
            merged = merged.drop_duplicates(subset=present, keep="last")
        sort_cols = [c for c in ("season", "week", "start_date", "game_id") if c in merged.columns]
        if sort_cols:
            merged = merged.sort_values(sort_cols, kind="stable").reset_index(drop=True)
        self.write(table, merged)
        return merged

    def tables(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.parquet"))

    def summary(self) -> pd.DataFrame:
        rows = []
        for t in self.tables():
            df = self.read(t)
            seasons = ""
            if "season" in df.columns and len(df):
                seasons = f"{int(df['season'].min())}-{int(df['season'].max())}"
            rows.append({
                "table": t,
                "rows": len(df),
                "seasons": seasons,
                "mb": round(self.path(t).stat().st_size / 1e6, 2),
            })
        return pd.DataFrame(rows)
