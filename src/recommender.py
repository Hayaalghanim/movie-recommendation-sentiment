import pandas as pd


def create_baseline_predictions(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Create baseline predicted ratings using the average rating of each movie.
    """
    movie_avg = ratings.groupby("movieId")["rating"].mean().reset_index()
    movie_avg.columns = ["movieId", "predicted_rating"]

    baseline = ratings.merge(movie_avg, on="movieId", how="left")
    return baseline


def map_sentiment_to_rating_scale(sentiment_score):
    """
    Convert sentiment score to rating scale:
    -1 -> 1
     0 -> 3
     1 -> 5
    """
    mapping = {
        -1: 1,
        0: 3,
        1: 5
    }
    return mapping.get(sentiment_score, 3)


def create_hybrid_predictions(baseline: pd.DataFrame, sentiment_score: float, alpha: float) -> pd.DataFrame:
    """
    Create hybrid predictions using:
    FinalScore = alpha * PredictedRating + (1 - alpha) * SentimentScore
    """
    hybrid = baseline.copy()

    sentiment_rating = map_sentiment_to_rating_scale(sentiment_score)

    hybrid["sentiment_rating"] = sentiment_rating
    hybrid["alpha"] = alpha
    hybrid["hybrid_prediction"] = (
        alpha * hybrid["predicted_rating"] +
        (1 - alpha) * hybrid["sentiment_rating"]
    )

    return hybrid