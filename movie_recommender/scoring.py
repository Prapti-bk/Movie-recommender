"""Ranking maths. Pure NumPy functions, independently testable."""
import numpy as np


def normalized_weighted_rating(ratings: np.ndarray, votes: np.ndarray,
                               percentile: float = 0.60) -> np.ndarray:
    """IMDB-style Bayesian average scaled to [0, 1].

        WR = v/(v+m) * R  +  m/(v+m) * C

    R = movie rating, v = its votes, C = mean rating of all movies, m = vote
    count at `percentile`. Few votes pull a rating toward the global mean, so a
    9.0 from 3 voters cannot outrank an 8.4 from 30,000.
    """
    mean_rating = ratings.mean()
    min_votes = np.quantile(votes, percentile)
    wr = votes / (votes + min_votes + 1e-9) * ratings + min_votes / (votes + min_votes + 1e-9) * mean_rating
    span = wr.max() - wr.min()
    return (wr - wr.min()) / span if span > 0 else np.zeros_like(wr)


def blend(similarity: np.ndarray, quality: np.ndarray, quality_weight: float) -> np.ndarray:
    """Convex combination of relevance and quality."""
    return (1.0 - quality_weight) * similarity + quality_weight * quality


def top_k(scores: np.ndarray, eligible: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest-scoring eligible items, best first.

    Uses argpartition (O(n)) to find the top k, then sorts only those k
    (O(k log k)); cheaper than sorting all n scores when n is large.
    """
    k = min(k, int(eligible.sum()))
    if k == 0:
        return np.array([], dtype=int)
    masked = np.where(eligible, scores, -np.inf)
    candidates = np.argpartition(-masked, k - 1)[:k]
    return candidates[np.argsort(-masked[candidates])]
