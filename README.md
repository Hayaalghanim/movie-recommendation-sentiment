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
Used for movie ratings and recommendation experiments.<br>
Link: https://grouplens.org/datasets/movielens/latest/

### 2. Twitter Sentiment Dataset
Used for tweet preprocessing and sentiment label preparation.<br>
Link: https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis

## Goal

To compare a baseline recommendation model with a hybrid model that combines predicted ratings and sentiment scores.

## Current Implementation

The current version performs:

1. Loading movie and Twitter datasets  
2. Cleaning tweet text  
3. Encoding sentiment labels  
4. Creating baseline movie rating predictions using average movie ratings
5. Generating hybrid predictions using sentiment scores  
6. Evaluating predictions using:
   - RMSE
   - MAE

## Evaluation

The system is evaluated using:

- **RMSE (Root Mean Square Error)** – measures prediction accuracy  
- **MAE (Mean Absolute Error)** – measures average prediction error  

Lower values indicate better performance.

## Results

Baseline Model:
- RMSE: 0.8762  
- MAE: 0.6668  

Hybrid Model:
- α = 0.2 → RMSE: 0.9100, MAE: 0.6940  
- α = 0.5 → RMSE: 0.8895, MAE: 0.6800  
- α = 0.8 → RMSE: 0.8783, MAE: 0.6683  

The best hybrid result was achieved at α = 0.8, but it did not significantly outperform the baseline model.

## Key Insight

Sentiment integration had limited impact on performance.  
This is mainly due to weak alignment between sentiment data and movie ratings.

## Future Work

- Use real movie-linked sentiment data (e.g., reviews instead of generic tweets)
- Improve sentiment quality using advanced models
- Evaluate using ranking metrics (Top-N) in addition to RMSE/MAE

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

## Notes

The sentiment processing logic is implemented in the main pipeline for simplicity.  
The `sentiment.py` file is included for modular structure and future extension.
