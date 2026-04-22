import pandas as pd


def create_baseline_predictions(ratings: pd.DataFrame) -> pd.DataFrame:
    """
    Create baseline predicted ratings using the average rating of each movie
    """
    movie_avg = ratings.groupby("movieId")["rating"].mean().reset_index()
    movie_avg.columns = ["movieId", "predicted_rating"]

    baseline = ratings.merge(movie_avg, on="movieId", how="left")
    return baseline
