import json

import pandas as pd
import pytest

from movie_recommender.exceptions import DataValidationError
from movie_recommender.repositories import COLUMNS, SampleCsvRepository, TmdbRepository, default_repository


def test_sample_repository_returns_standard_schema():
    df = SampleCsvRepository("data/sample_movies.csv").load()
    assert list(df.columns) == list(COLUMNS) and len(df) > 0 and df["title"].is_unique


def test_missing_columns_are_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"title": ["X"]}).to_csv(bad, index=False)
    with pytest.raises(DataValidationError, match="missing required columns"):
        SampleCsvRepository(bad).load()


def test_tmdb_json_columns_are_parsed(tmp_path):
    path = tmp_path / "tmdb_5000_movies.csv"
    pd.DataFrame([{
        "id": 1, "title": "Test Movie", "release_date": "2010-07-16", "overview": "A plot.",
        "vote_average": 8.0, "vote_count": 100,
        "genres": json.dumps([{"id": 1, "name": "Action"}, {"id": 2, "name": "Drama"}]),
        "keywords": json.dumps([{"id": 9, "name": "heist"}]),
    }]).to_csv(path, index=False)
    row = TmdbRepository(path).load().iloc[0]
    assert row["genres"] == "Action|Drama" and row["year"] == 2010 and row["keywords"] == "heist"


def test_default_repository_falls_back_to_sample(tmp_path):
    assert isinstance(default_repository(tmp_path), SampleCsvRepository)
