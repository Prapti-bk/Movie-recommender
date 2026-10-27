import pytest

from movie_recommender import Filters
from movie_recommender.exceptions import EmptyProfileError, MovieNotFoundError


def test_excludes_query_movie_and_respects_limit(service):
    recs = service.similar_to("The Matrix", Filters(limit=5))
    assert len(recs) == 5 and "The Matrix" not in [r.title for r in recs]


def test_lookup_is_case_insensitive(service):
    assert service.similar_to("the matrix", Filters(limit=1))


def test_similar_movies_share_genre(service):
    recs = service.similar_to("Interstellar", Filters(limit=5))
    assert sum("Science Fiction" in r.genres for r in recs) >= 3


def test_genre_filter(service):
    recs = service.similar_to("Toy Story", Filters(limit=5, genres=frozenset({"Animation"})))
    assert recs and all("Animation" in r.genres for r in recs)


def test_min_rating_filter(service):
    recs = service.similar_to("Inception", Filters(limit=10, min_rating=8.0))
    assert all(r.rating >= 8.0 for r in recs)


def test_taste_profile_prefers_liked_genre(service):
    recs = service.for_profile(["The Godfather", "Goodfellas"], filters=Filters(limit=5))
    assert sum("Crime" in r.genres for r in recs) >= 3
    assert not {"The Godfather", "Goodfellas"} & {r.title for r in recs}


def test_genre_only_profile(service):
    recs = service.for_profile(favourite_genres=["Horror"], filters=Filters(limit=3))
    assert len(recs) == 3 and all("Horror" in r.genres for r in recs)


def test_unknown_title_raises(service):
    with pytest.raises(MovieNotFoundError):
        service.similar_to("Not A Real Movie")


def test_empty_profile_raises(service):
    with pytest.raises(EmptyProfileError):
        service.for_profile()


def test_filters_validate_input():
    with pytest.raises(ValueError):
        Filters(limit=0)
    with pytest.raises(ValueError):
        Filters(quality_weight=1.5)


def test_results_sorted_by_blended_score_descending(service):
    recs = service.similar_to("Inception", Filters(limit=8, quality_weight=0.0))
    sims = [r.similarity for r in recs]
    assert sims == sorted(sims, reverse=True)
