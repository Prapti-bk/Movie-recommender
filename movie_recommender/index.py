"""Similarity index: owns the TF-IDF vectorizer and the (sparse) movie matrix."""
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityIndex:
    """Fit once on all movie documents, then answer 'how similar is everything to X?'.

    The matrix is stored as a SciPy CSR *sparse* matrix: a movie uses only a few
    dozen of the thousands of vocabulary terms, so this is far smaller than a
    dense array. We never build the full N x N similarity matrix; each query
    computes just one row of it (O(N) memory instead of O(N^2)).
    """

    def __init__(self, documents: pd.Series, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self._vectorizer = TfidfVectorizer(stop_words="english", ngram_range=ngram_range)
        self._matrix: csr_matrix = self._vectorizer.fit_transform(documents)

    def __len__(self) -> int:
        return self._matrix.shape[0]

    def vectorize(self, text: str) -> np.ndarray:
        """Dense (1, vocab) TF-IDF vector for arbitrary text."""
        return self._vectorizer.transform([text]).toarray()

    def centroid(self, rows: Sequence[int]) -> np.ndarray:
        """Average vector of several movies: a simple 'taste profile'. Shape (1, vocab)."""
        return np.asarray(self._matrix[list(rows)].mean(axis=0))

    def similarity_to_row(self, row: int) -> np.ndarray:
        return cosine_similarity(self._matrix[row], self._matrix).ravel()

    def similarity_to_vector(self, vector: np.ndarray) -> np.ndarray:
        return cosine_similarity(vector, self._matrix).ravel()
