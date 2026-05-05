import pandas as pd


def create_baseline_predictions(ratings: pd.DataFrame) -> pd.DataFrame:
    movie_avg = ratings.groupby("movieId")["rating"].mean().reset_index()
    movie_avg.columns = ["movieId", "predicted_rating"]

    baseline = ratings.merge(movie_avg, on="movieId", how="left")
    return baseline


def clean_movie_title(title: str) -> str:
    return (
        str(title)
        .replace("&", "and")
        .lower()
        .strip()
    )


def add_movie_specific_sentiment(
    baseline: pd.DataFrame,
    movies: pd.DataFrame,
    tweets: pd.DataFrame
) -> pd.DataFrame:
    hybrid_data = baseline.copy()

    # Create sentiment from ratings:
    # rating >= 4  -> positive sentiment
    # rating <= 2  -> negative sentiment
    # rating 2.5-3.5 -> neutral sentiment
    hybrid_data["sentiment_score"] = hybrid_data["rating"].apply(
        lambda x: 1 if x >= 4 else (-1 if x <= 2 else 0)
    )

    # Average sentiment for each movie
    movie_sentiment = (
        hybrid_data.groupby("movieId")["sentiment_score"]
        .mean()
        .reset_index()
    )

    movie_sentiment.columns = ["movieId", "avg_sentiment_score"]

    # Merge movie sentiment back into predictions
    hybrid_data = hybrid_data.drop(columns=["sentiment_score"])
    hybrid_data = hybrid_data.merge(movie_sentiment, on="movieId", how="left")

    # Convert sentiment score from [-1, 1] to rating scale [1, 5]
    hybrid_data["sentiment_rating"] = 3 + (hybrid_data["avg_sentiment_score"] * 2)

    return hybrid_data

def create_hybrid_predictions(hybrid_data: pd.DataFrame, alpha: float) -> pd.DataFrame:
    hybrid = hybrid_data.copy()

    hybrid["alpha"] = alpha
    
    hybrid["alpha"] = alpha
    hybrid["hybrid_prediction"] = (
        alpha * hybrid["predicted_rating"]
        + (1 - alpha) * hybrid["sentiment_rating"]
    )

    return hybrid
