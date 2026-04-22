import re
import pandas as pd


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_tweets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=["text", "sentiment"])
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"] != ""]
    return df

def encode_sentiment(df):
    mapping = {
        "Positive": 1,
        "Neutral": 0,
        "Negative": -1
    }
    
    df = df.copy()
    df["sentiment_score"] = df["sentiment"].map(mapping)
    
    return df

