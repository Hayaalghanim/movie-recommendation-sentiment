import numpy as np

def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))

def calculate_mae(actual, predicted):
    return np.mean(np.abs(actual - predicted))
