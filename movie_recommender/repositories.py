"""Data access layer (Repository pattern).

The engine only needs a `MovieRepository`: anything with `label` and `load()`.
To support a new source (a database, an API, MovieLens), write one new class;
nothing else in the project changes.
"""
import json
import logging
from pathlib import Path
from typing import Protocol

import pandas as pd

from .exceptions import DataValidationError

log = logging.getLogger(__name__)

REQUIRED = ("title", "genres", "overview", "vote_average", "vote_count")
OPTIONAL_TEXT = ("keywords", "cast", "director")
COLUMNS = ("title", "year", "genres", "keywords", "overview", "cast", "director",
           "vote_average", "vote_count")


class MovieRepository(Protocol):
    label: str

    def load(self) -> pd.DataFrame:
        """Return a validated DataFrame with exactly `COLUMNS`."""


def normalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Validate the schema, fill gaps and de-duplicate. Every repository ends here,
    so the rest of the system can rely on one consistent shape."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise DataValidationError(f"{source}: missing required columns {missing}")
    df = df.copy()
    for col in OPTIONAL_TEXT:
        df[col] = df[col].fillna("") if col in df else ""
    if "year" not in df:
        df["year"] = float("nan")
    df["overview"] = df["overview"].fillna("")
    df["genres"] = df["genres"].fillna("")
    df["vote_average"] = pd.to_numeric(df["vote_average"], errors="coerce").fillna(0.0)
    df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce").fillna(0).astype(int)

    df = df.dropna(subset=["title"])
    df = df.drop_duplicates(subset="title", keep="first")   # titles are the lookup key
    return df[list(COLUMNS)].reset_index(drop=True)


class SampleCsvRepository:
    """Small bundled dataset so the project runs with zero setup."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self.label = "Bundled sample dataset"

    def load(self) -> pd.DataFrame:
        return normalize(pd.read_csv(self._path), str(self._path))


class TmdbRepository:
    """The Kaggle 'TMDB 5000 Movie Dataset' (movies CSV, optional credits CSV)."""

    def __init__(self, movies_csv: Path, credits_csv: Path | None = None) -> None:
        self._movies_csv = Path(movies_csv)
        self._credits_csv = Path(credits_csv) if credits_csv else None
        self.label = "TMDB 5000 dataset"

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(self._movies_csv)
        df["year"] = pd.to_datetime(df.get("release_date"), errors="coerce").dt.year
        df["genres"] = df["genres"].map(_join_names)
        df["keywords"] = df["keywords"].map(_join_names)
        if self._credits_csv and self._credits_csv.exists():
            credits = pd.read_csv(self._credits_csv).rename(columns={"movie_id": "id"})
            credits["cast"] = credits["cast"].map(lambda c: _join_names(c, limit=3))
            credits["director"] = credits["crew"].map(_find_director)
            df = df.merge(credits[["id", "cast", "director"]], on="id", how="left")
        return normalize(df, str(self._movies_csv))


def default_repository(data_dir: Path = Path("data")) -> MovieRepository:
    """Use the full TMDB files if the user added them, else the bundled sample."""
    tmdb = data_dir / "tmdb_5000_movies.csv"
    if tmdb.exists():
        return TmdbRepository(tmdb, data_dir / "tmdb_5000_credits.csv")
    return SampleCsvRepository(data_dir / "sample_movies.csv")


# ---- TMDB JSON-in-CSV parsing helpers -------------------------------------

def _parse_json_list(cell) -> list[dict]:
    try:
        parsed = json.loads(cell)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _join_names(cell, limit: int | None = None) -> str:
    return "|".join(item.get("name", "") for item in _parse_json_list(cell)[:limit])


def _find_director(cell) -> str:
    return next((p["name"] for p in _parse_json_list(cell) if p.get("job") == "Director"), "")
