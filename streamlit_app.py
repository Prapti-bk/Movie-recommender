"""Streamlit UI. Presentation only: all logic lives in the movie_recommender package."""
import streamlit as st

from movie_recommender import Filters, Recommendation, RecommenderService
from movie_recommender.exceptions import RecommenderError
from movie_recommender.posters import PosterClient
from movie_recommender.repositories import default_repository

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource
def get_service() -> tuple[RecommenderService, str]:
    repo = default_repository()
    return RecommenderService.from_repository(repo), repo.label


@st.cache_data(show_spinner=False)
def poster_for(title: str, year: int | None) -> str | None:
    return PosterClient().url_for(title, year)


def render(recommendations: list[Recommendation]) -> None:
    if not recommendations:
        st.warning("No movies match those filters. Try relaxing them.")
        return
    for start in range(0, len(recommendations), 5):
        for col, rec in zip(st.columns(5), recommendations[start:start + 5]):
            with col:
                if url := poster_for(rec.title, rec.year):
                    st.image(url)
                st.markdown(f"**{rec.title}**" + (f" ({rec.year})" if rec.year else ""))
                st.caption(f"⭐ {rec.rating:.1f} · {', '.join(rec.genres)}")
                st.caption(f"Match: {rec.similarity:.0%}")
                with st.expander("Plot"):
                    st.write(rec.overview)


service, source = get_service()
st.title("🎬 Movie Recommender")
st.caption(f"Content-based filtering · TF-IDF + cosine similarity · Data: {source}")

with st.sidebar:
    st.header("Filters")
    filters = Filters(
        limit=st.slider("Number of results", 5, 20, 10, step=5),
        min_rating=st.slider("Minimum rating", 0.0, 9.0, 0.0, step=0.5),
        genres=frozenset(st.multiselect("Limit to genres", service.genres)),
        quality_weight=st.slider("Favour highly-rated movies", 0.0, 0.5, 0.15, step=0.05),
    )

similar_tab, taste_tab = st.tabs(["🔎 Similar to a movie", "🎯 Based on my taste"])

with similar_tab:
    chosen = st.selectbox("Pick a movie you like", service.titles)
    if st.button("Recommend", key="similar"):
        render(service.similar_to(chosen, filters))

with taste_tab:
    liked = st.multiselect("Movies you've enjoyed", service.titles)
    favourite = st.multiselect("Favourite genres", service.genres, key="fav")
    if st.button("Find movies for me", key="profile"):
        try:
            render(service.for_profile(liked, favourite, filters))
        except RecommenderError as exc:
            st.warning(str(exc))
