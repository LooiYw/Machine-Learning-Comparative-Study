import gc
import json
import os
import time
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR
from preprocessing import load_ratings_preprocessed, load_ratings_scaled
from utils import get_splits, evaluate, save_results

BASELINE_PARAMS = dict(
    epsilon=0.0,
    tol=1e-4,
    C=1.0,
    loss='epsilon_insensitive',
    fit_intercept=True,
    intercept_scaling=1.0,
    dual='auto',
    random_state=42,
    max_iter=1000,
)

# Columns that are NOT used as model inputs
NON_FEATURE_COLS = ['userId', 'movieId', 'rating', 'split']

# Results file
RESULTS_FILE = os.path.join('results', 'model_evaluation_results.csv')

# The two datasets we compare: (name, loader function)
DATASETS = [
    ('SVM_PREPROCESSED', load_ratings_preprocessed),
    ('SVM_SCALED', load_ratings_scaled),
]


# Build the model: fill missing values first, then LinearSVR (params overrides the baseline)
def build_model(params=None):
    merged = {**BASELINE_PARAMS, **(params or {})}
    return Pipeline([
        # users/movies not seen in train have missing stats, so fill them with the mean
        ('imputer', SimpleImputer(strategy='mean')),
        ('svr', LinearSVR(**merged)),
    ])


# Train and test the SVM on both datasets, then save the results
def run_svm(params=None):
    results = []
    for label, loader in DATASETS:
        print(f"\n=== MODEL: {label} ===")
        df = loader()
        
        # Unpack splits and meta
        splits, meta = get_splits(df)
        del df               
        gc.collect()

        X_train, y_train = splits['train']
        model = build_model(params)

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

        save_results(results[-3:], {**BASELINE_PARAMS, **(params or {})})
    return results

if __name__ == "__main__":
    run_svm()