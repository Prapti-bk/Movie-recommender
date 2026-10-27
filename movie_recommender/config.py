"""All tunable knobs in one place. Change behaviour here, not deep in the code."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FeatureWeights:
    """How many times each field is repeated in the text fed to TF-IDF (more = more influence)."""

    genres: int = 3
    keywords: int = 2
    cast: int = 1
    director: int = 1
    overview: int = 1


@dataclass(frozen=True)
class RecommenderConfig:
    weights: FeatureWeights = field(default_factory=FeatureWeights)
    ngram_range: tuple[int, int] = (1, 2)
    vote_percentile: float = 0.60       # the "m" in the weighted-rating formula
    genre_profile_weight: float = 0.5   # pull of favourite genres in a taste profile
