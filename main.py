import os
import pandas as pd

from src.preprocessing import preprocess_tweets, encode_sentiment
from src.recommender import create_baseline_predictions, create_hybrid_predictions
from src.evaluation import calculate_rmse, calculate_mae


def main():
    print("Loading datasets...")

    # Load MovieLens data
    ratings = pd.read_csv("data/ratings.csv")
    movies = pd.read_csv("data/movies.csv")

    # Load Twitter data
    tweets_train = pd.read_csv("data/twitter_training.csv", header=None)
    tweets_val = pd.read_csv("data/twitter_validation.csv", header=None)

    tweets_train.columns = ["id", "entity", "sentiment", "text"]
    tweets_val.columns = ["id", "entity", "sentiment", "text"]

    # Print shapes
    print("\nBefore preprocessing:")
    print("Tweets train shape:", tweets_train.shape)

    # Preprocess tweets
    tweets_train = preprocess_tweets(tweets_train)

    print("\nAfter preprocessing:")
    print("Tweets train shape:", tweets_train.shape)
    print("\nProcessed tweets preview:")
    print(tweets_train[["sentiment", "text", "clean_text"]].head())

    # Encode sentiment labels
    tweets_train = encode_sentiment(tweets_train)

    print("\nSentiment encoding preview:")
    print(tweets_train[["sentiment", "sentiment_score"]].head())

    # Create baseline predictions
    baseline = create_baseline_predictions(ratings)

    print("\nBaseline preview:")
    print(baseline[["userId", "movieId", "rating", "predicted_rating"]].head())

    # Calculate baseline evaluation metrics
    baseline_rmse = calculate_rmse(baseline["rating"], baseline["predicted_rating"])
    baseline_mae = calculate_mae(baseline["rating"], baseline["predicted_rating"])

    print("\nBaseline Evaluation Results:")
    print("RMSE:", baseline_rmse)
    print("MAE:", baseline_mae)

    # Create results folder
    os.makedirs("results", exist_ok=True)

    # Save baseline predictions
    baseline.to_csv("results/baseline_predictions.csv", index=False)

    # Calculate average sentiment score
    avg_sentiment_score = round(tweets_train["sentiment_score"].mean())

    print("\nAverage sentiment score:")
    print(avg_sentiment_score)

    # Test hybrid model with different alpha values
    alpha_values = [0.2, 0.5, 0.8]
    hybrid_results = []

    for alpha in alpha_values:
        hybrid = create_hybrid_predictions(baseline, avg_sentiment_score, alpha)

        hybrid_rmse = calculate_rmse(hybrid["rating"], hybrid["hybrid_prediction"])
        hybrid_mae = calculate_mae(hybrid["rating"], hybrid["hybrid_prediction"])

        print(f"\nHybrid Model Results for alpha = {alpha}:")
        print("RMSE:", hybrid_rmse)
        print("MAE:", hybrid_mae)

        hybrid_results.append({
            "model": "Hybrid",
            "alpha": alpha,
            "RMSE": hybrid_rmse,
            "MAE": hybrid_mae
        })

        # Save hybrid predictions
        hybrid.to_csv(f"results/hybrid_predictions_alpha_{alpha}.csv", index=False)

    # Save evaluation results
    results_df = pd.DataFrame(hybrid_results)

    baseline_row = pd.DataFrame([{
        "model": "Baseline",
        "alpha": "-",
        "RMSE": baseline_rmse,
        "MAE": baseline_mae
    }])

    final_results = pd.concat([baseline_row, results_df], ignore_index=True)
    final_results.to_csv("results/evaluation_results.csv", index=False)

    # Save metrics
    with open("results/metrics.txt", "w") as f:
        f.write("Evaluation Results\n")
        f.write("==================\n\n")
        f.write(f"Baseline RMSE: {baseline_rmse:.4f}\n")
        f.write(f"Baseline MAE: {baseline_mae:.4f}\n\n")

        for _, row in results_df.iterrows():
            f.write(f"Hybrid alpha={row['alpha']}\n")
            f.write(f"RMSE: {row['RMSE']:.4f}\n")
            f.write(f"MAE: {row['MAE']:.4f}\n\n")

    print("\nAll results saved in results/ folder.")


if __name__ == "__main__":
    main()