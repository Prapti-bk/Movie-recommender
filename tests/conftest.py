import pytest

from movie_recommender import RecommenderService
from movie_recommender.repositories import SampleCsvRepository


@pytest.fixture(scope="session")
def service() -> RecommenderService:
    return RecommenderService.from_repository(SampleCsvRepository("data/sample_movies.csv"))
