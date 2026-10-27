"""Content-based movie recommendation engine (TF-IDF + cosine similarity)."""
from .models import Filters, Recommendation
from .service import RecommenderService

__all__ = ["Filters", "Recommendation", "RecommenderService"]
