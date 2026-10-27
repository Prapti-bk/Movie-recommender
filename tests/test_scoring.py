import numpy as np

from movie_recommender.scoring import blend, normalized_weighted_rating, top_k


def test_top_k_matches_full_sort():
    rng = np.random.default_rng(0)
    scores = rng.random(200)
    eligible = rng.random(200) > 0.3
    expected = [i for i in np.argsort(-scores) if eligible[i]][:10]
    assert list(top_k(scores, eligible, 10)) == expected


def test_top_k_handles_k_larger_than_eligible_and_none_eligible():
    scores = np.array([0.1, 0.9, 0.5])
    assert list(top_k(scores, np.array([True, False, True]), 10)) == [2, 0]
    assert top_k(scores, np.zeros(3, bool), 5).size == 0


def test_weighted_rating_is_bounded_and_discounts_few_votes():
    ratings = np.array([9.0, 8.4, 5.0])
    votes = np.array([3, 30000, 20000])
    wr = normalized_weighted_rating(ratings, votes)
    assert wr.min() == 0.0 and wr.max() == 1.0
    assert wr[1] > wr[0]          # 8.4 from 30k beats 9.0 from 3 votes


def test_blend_extremes():
    sim, qual = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert np.allclose(blend(sim, qual, 0.0), sim)
    assert np.allclose(blend(sim, qual, 1.0), qual)
