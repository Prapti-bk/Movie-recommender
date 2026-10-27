"""Optional poster lookup via the TMDB API. Framework-free and fails soft."""
import logging
import os

import requests

log = logging.getLogger(__name__)
SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
IMAGE_BASE = "https://image.tmdb.org/t/p/w342"


class PosterClient:
    def __init__(self, api_key: str | None = None, timeout: float = 5.0) -> None:
        self._api_key = api_key or os.getenv("TMDB_API_KEY")
        self._timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    def url_for(self, title: str, year: int | None = None) -> str | None:
        """Poster URL, or None if disabled / not found / the network fails."""
        if not self.enabled:
            return None
        params = {"api_key": self._api_key, "query": title}
        if year:
            params["year"] = year
        try:
            response = requests.get(SEARCH_URL, params=params, timeout=self._timeout)
            response.raise_for_status()
            path = response.json()["results"][0]["poster_path"]
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            log.warning("Poster lookup failed for %r: %s", title, exc)
            return None
        return f"{IMAGE_BASE}{path}" if path else None
