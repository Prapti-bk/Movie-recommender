"""Feature engineering: turn each movie's metadata into one weighted text document.

Pure functions (DataFrame in, Series out) with no state: trivial to test and tweak.
"""
import pandas as pd

from .config import FeatureWeights


def to_tokens(pipe_separated: str) -> str:
    """'Science Fiction|Action' -> 'sciencefiction action'.

    Spaces are removed inside a name so multi-word values ('Science Fiction',
    'Christopher Nolan') stay a single token instead of leaking into plot words.
    """
    parts = (p.replace(" ", "").lower() for p in pipe_separated.split("|"))
    return " ".join(p for p in parts if p)


def build_feature_text(movies: pd.DataFrame, weights: FeatureWeights) -> pd.Series:
    """One text document per movie. Repeating a field boosts its TF-IDF influence."""

    def weighted(column: str, repeat: int) -> pd.Series:
        return (movies[column].map(to_tokens) + " ") * repeat

    return (
        weighted("genres", weights.genres)
        + weighted("keywords", weights.keywords)
        + weighted("cast", weights.cast)
        + weighted("director", weights.director)
        + (movies["overview"] + " ") * weights.overview
    )


def genre_query_text(genres, weights: FeatureWeights) -> str:
    """Text equivalent of 'I like these genres', weighted like real movie documents."""
    return (to_tokens("|".join(sorted(genres))) + " ") * weights.genres
