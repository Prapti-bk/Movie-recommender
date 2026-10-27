"""RecommenderService: the public API. UI code talks only to this class."""
import dataclasses
import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import RecommenderConfig
from .exceptions import EmptyProfileError, MovieNotFoundError
from .features import build_feature_text, genre_query_text
from .index import SimilarityIndex
from .models import Filters, Recommendation
from .repositories import MovieRepository
from .scoring import blend, normalized_weighted_rating, top_k

log = logging.getLogger(__name__)


class RecommenderService:
    """Content-based recommendations over an in-memory movie catalogue.

    Precomputed at start-up (so each request is fast):
      * `_row_by_title`  dict[str, int]          title -> row, O(1) lookup
      * `_genre_masks`   dict[str, bool array]   inverted index: genre -> movies, so
                                                 genre filtering is a vectorized OR
      * `_quality`       float array             normalized weighted rating per movie
      * `_index`         SimilarityIndex         sparse TF-IDF matrix
    """

    def __init__(self, movies: pd.DataFrame, config: RecommenderConfig | None = None):
        config = config or RecommenderConfig()
        self._movies = movies.reset_index(drop=True)
        self._config = config
        self._index = SimilarityIndex(
            build_feature_text(self._movies, config.weights), config.ngram_range)

        self._row_by_title = {t.lower(): i for i, t in enumerate(self._movies["title"])}
        genre_sets = [frozenset(filter(None, g.split("|"))) for g in self._movies["genres"]]
        self.genres: list[str] = sorted(set().union(*genre_sets))
        self._genre_masks = {
            g: np.fromiter((g in s for s in genre_sets), dtype=bool, count=len(genre_sets))
            for g in self.genres
        }
        self._quality = normalized_weighted_rating(
            self._movies["vote_average"].to_numpy(float),
            self._movies["vote_count"].to_numpy(float),
            config.vote_percentile)
        log.info("Indexed %d movies, %d genres", len(self._index), len(self.genres))

    @classmethod
    def from_repository(cls, repo: MovieRepository,
                        config: RecommenderConfig | None = None) -> "RecommenderService":
        return cls(repo.load(), config)

    # ---------------- public API ----------------
    @property
    def titles(self) -> list[str]:
        return sorted(self._movies["title"])

    def similar_to(self, title: str, filters: Filters | None = None) -> list[Recommendation]:
        """Movies most similar to `title`. Raises MovieNotFoundError."""
        filters = filters or Filters()
        row = self._row_for(title)
        sims = self._index.similarity_to_row(row)
        return self._recommend(sims, exclude=[row], filters=filters)

    def for_profile(self, liked_titles: Iterable[str] = (), favourite_genres: Iterable[str] = (),
                    filters: Filters | None = None) -> list[Recommendation]:
        """Recommend from a taste profile of liked movies and/or favourite genres.

        Raises EmptyProfileError if both are empty.
        """
        filters = filters or Filters()
        liked_rows = [self._row_for(t) for t in liked_titles]
        genres = frozenset(favourite_genres)
        if not liked_rows and not genres:
            raise EmptyProfileError("Pick at least one liked movie or favourite genre.")

        profile = np.zeros((1, self._index.vectorize("").shape[1]))
        if liked_rows:
            profile += self._index.centroid(liked_rows)
        if genres:
            query = genre_query_text(genres, self._config.weights)
            profile += self._config.genre_profile_weight * self._index.vectorize(query)
        if not liked_rows:                        # genres are the only signal: also filter
            filters = dataclasses.replace(filters, genres=filters.genres | genres)

        sims = self._index.similarity_to_vector(profile)
        return self._recommend(sims, exclude=liked_rows, filters=filters)

    # ---------------- internals ----------------
    def _row_for(self, title: str) -> int:
        try:
            return self._row_by_title[title.lower()]
        except KeyError:
            raise MovieNotFoundError(f"'{title}' is not in the catalogue.") from None

    def _eligible(self, filters: Filters, exclude: list[int]) -> np.ndarray:
        mask = (self._movies["vote_average"] >= filters.min_rating).to_numpy(copy=True)
        if filters.genres:
            mask &= np.logical_or.reduce([self._genre_masks[g] for g in filters.genres
                                          if g in self._genre_masks] or [np.zeros_like(mask)])
        mask[exclude] = False
        return mask

    def _recommend(self, sims: np.ndarray, exclude: list[int], filters: Filters):
        scores = blend(sims, self._quality, filters.quality_weight)
        rows = top_k(scores, self._eligible(filters, exclude), filters.limit)
        return [self._to_recommendation(r, sims[r]) for r in rows]

    def _to_recommendation(self, row: int, similarity: float) -> Recommendation:
        m = self._movies.iloc[row]
        return Recommendation(
            title=m["title"],
            year=None if pd.isna(m["year"]) else int(m["year"]),
            genres=tuple(filter(None, m["genres"].split("|"))),
            rating=float(m["vote_average"]),
            overview=m["overview"],
            similarity=round(float(similarity), 3),
        )
