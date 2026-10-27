"""Domain-specific exceptions, so callers can react to precise failure modes."""


class RecommenderError(Exception):
    """Base class for all errors raised by this package."""


class MovieNotFoundError(RecommenderError, KeyError):
    """The requested title is not in the dataset."""


class EmptyProfileError(RecommenderError, ValueError):
    """A taste profile needs at least one liked movie or favourite genre."""


class DataValidationError(RecommenderError, ValueError):
    """A data source is missing required columns."""
