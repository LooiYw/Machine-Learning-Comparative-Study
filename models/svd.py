import gc
import json
import os
import time
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from preprocessing import load_ratings_preprocessed, load_ratings_scaled
from utils import get_splits, evaluate, save_results

# Values are sklearn defaults unless marked "original"
SVD_BASELINE = dict(
    n_components=20,                      # original (sklearn default is 2)
    algorithm='randomized',
    n_iter=5,
    n_oversamples=10,
    power_iteration_normalizer='auto',
    random_state=42,                      # original, so results can be repeated
)

RIDGE_BASELINE = dict(
    alpha=1.0,                            # main value to tune
    fit_intercept=True,
    solver='auto',
    random_state=42,                      # original (only used by some solvers)
)

# Columns that are NOT used as model inputs
NON_FEATURE_COLS = ['userId', 'movieId', 'rating', 'split']

# Results file
RESULTS_FILE = os.path.join('results', 'model_evaluation_results.csv')

# The two datasets we compare: (name, loader function)
DATASETS = [
    ('SVD_PREPROCESSED', load_ratings_preprocessed),
    ('SVD_SCALED', load_ratings_scaled),
]


# Build the model: fill missing values, SVD, then Ridge (params override the baselines)
def build_model(svd_params=None, ridge_params=None):
    return Pipeline([
        # users/movies not seen in train have missing stats, so fill them with the mean
        ('imputer', SimpleImputer(strategy='mean')),
        ('svd', TruncatedSVD(**{**SVD_BASELINE, **(svd_params or {})})),
        ('ridge', Ridge(**{**RIDGE_BASELINE, **(ridge_params or {})})),
    ])


# Train and test SVD + Ridge on both datasets, then save the results
def run_svd(svd_params=None, ridge_params=None):
    results = []
    for label, loader in DATASETS:
        print(f"\n=== MODEL: {label} ===")
        df = loader()
        
        # Unpack splits and meta
        splits, meta = get_splits(df)
        del df               
        gc.collect()

        X_train, y_train = splits['train']
        
        # Build model takes both param sets
        model = build_model(svd_params, ridge_params)

        print(f"Training {label}...")
        
        # Track memory and time
        tracemalloc.start()
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        train_mem_mb = peak_mem / 1e6
        
        print(f"  Train Time: {train_time:.2f}s | Train Memory: {train_mem_mb:.2f}MB")

        for name in ('train', 'val', 'test'):
            X, y = splits[name]

            # 4. GET META FOR THE SPLIT AND PASS TO EVALUATE
            eval_meta = meta[name]
            res = evaluate(name, model, X, y, eval_meta)
            
            # 5. ADD TRAIN_MEM_MB TO RESULTS
            res.update(model=label, train_time_s=train_time, train_mem_mb=train_mem_mb)
            results.append(res)

        # Save both parameter dictionaries
        used = {'svd': {**SVD_BASELINE, **(svd_params or {})},
                'ridge': {**RIDGE_BASELINE, **(ridge_params or {})}}
        save_results(results[-3:], used)
    return results

if __name__ == "__main__":
    run_svd()