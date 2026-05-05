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
    movies = movies.copy()
    tweets = tweets.copy()

    # Remove year from MovieLens title, example: Toy Story (1995) -> Toy Story
    movies["clean_title"] = (
        movies["title"]
        .str.replace(r"\s*\(\d{4}\)", "", regex=True)
        .apply(clean_movie_title)
    )

    tweets["clean_entity"] = tweets["entity"].apply(clean_movie_title)

    # Average sentiment per movie/entity
    movie_sentiment = (
        tweets.groupby("clean_entity")["sentiment_score"]
        .mean()
        .reset_index()
    )

    movie_sentiment.columns = ["clean_title", "avg_sentiment_score"]

    # Match sentiment with MovieLens movies
    movies_with_sentiment = movies.merge(
        movie_sentiment,
        on="clean_title",
        how="left"
    )

    # Merge movie sentiment into baseline predictions
    hybrid_data = baseline.merge(
        movies_with_sentiment[["movieId", "avg_sentiment_score"]],
        on="movieId",
        how="left"
    )

    # If no tweet sentiment exists for a movie, use neutral sentiment
    hybrid_data["avg_sentiment_score"] = hybrid_data["avg_sentiment_score"].fillna(0)

    # Convert sentiment scale from [-1, 1] to rating scale [1, 5]
    hybrid_data["sentiment_rating"] = 3 + (hybrid_data["avg_sentiment_score"] * 2)

    return hybrid_data


def create_hybrid_predictions(hybrid_data: pd.DataFrame, alpha: float) -> pd.DataFrame:
    hybrid = hybrid_data.copy()

    hybrid["alpha"] = alpha
    hybrid["hybrid_prediction"] = (
        alpha * hybrid["predicted_rating"]
        + (1 - alpha) * hybrid["sentiment_rating"]
    )

    return hybrid
