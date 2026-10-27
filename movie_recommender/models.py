"""Plain, immutable data objects that cross the service boundary."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Filters:
    """User-controlled constraints on a recommendation request (validated on creation)."""

    limit: int = 10
    min_rating: float = 0.0
    genres: frozenset[str] = frozenset()
    quality_weight: float = 0.15     # 0 = pure similarity, 1 = pure popularity/quality

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("limit must be at least 1")
        if not 0.0 <= self.quality_weight <= 1.0:
            raise ValueError("quality_weight must be between 0 and 1")


@dataclass(frozen=True)
class Recommendation:
    title: str
    year: int | None
    genres: tuple[str, ...]
    rating: float
    overview: str
    similarity: float
