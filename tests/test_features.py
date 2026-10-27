import pandas as pd

from movie_recommender.config import FeatureWeights
from movie_recommender.features import build_feature_text, to_tokens


def test_tokens_keep_multiword_names_together():
    assert to_tokens("Science Fiction|Action") == "sciencefiction action"
    assert to_tokens("") == ""


def test_weights_repeat_fields():
    df = pd.DataFrame({"genres": ["Drama"], "keywords": [""], "cast": [""],
                       "director": [""], "overview": ["a story"]})
    text = build_feature_text(df, FeatureWeights(genres=3, overview=1)).iloc[0]
    assert text.count("drama") == 3 and text.count("a story") == 1
