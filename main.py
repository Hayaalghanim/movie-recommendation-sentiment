import pandas as pd
from src.preprocessing import preprocess_tweets, encode_sentiment

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

    tweets_train = preprocess_tweets(tweets_train)

    print("\nAfter preprocessing:")
    print("Tweets train shape:", tweets_train.shape)
    print("\nProcessed tweets preview:")
    print(tweets_train[["sentiment", "text", "clean_text"]].head())

    tweets_train = encode_sentiment(tweets_train)

    print("\nSentiment encoding preview:")
    print(tweets_train[["sentiment", "sentiment_score"]].head())

if __name__ == "__main__":
    main()
