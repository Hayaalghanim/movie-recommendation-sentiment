# Movie Recommendation System Using Sentiment Analysis

This project implements a movie recommendation system that combines collaborative filtering with sentiment analysis on tweets.

## Project Structure

- `data/` - datasets used in the project  
  - `ratings.csv`
  - `movies.csv`
  - `twitter_training.csv`
  - `twitter_validation.csv`

- `src/` - source code modules  
  - `preprocessing.py`
  - `sentiment.py`
  - `recommender.py`
  - `evaluation.py`

- `results/` - saved outputs and future experiment results

- `main.py` - main pipeline execution script

## Datasets

### 1. MovieLens Dataset
Used for movie ratings and recommendation experiments.

### 2. Twitter Sentiment Dataset
Used for tweet preprocessing and sentiment label preparation.

## Goal

To compare a baseline recommendation model with a hybrid model that combines predicted ratings and sentiment scores.

## Current Implementation

The current version performs:

1. Loading movie and Twitter datasets  
2. Cleaning tweet text  
3. Encoding sentiment labels  
4. Creating baseline movie rating predictions using average movie ratings  
5. Evaluating predictions using:
   - RMSE
   - MAE

## Future Extension

The next version will integrate sentiment scores directly into the recommendation model to create a hybrid recommender system.

## Requirements

- Python 3.11+
- pandas
- numpy
- scikit-learn

## How to Run

1. Install dependencies:<br>
   ```bash
   pip install pandas numpy scikit-learn

2. Run the code:<br>
   ```bash
   python3 main.py
