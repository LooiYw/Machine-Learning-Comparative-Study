import gc
import json
import os
import time
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from preprocessing import load_ratings_preprocessed, load_ratings_scaled
from utils import get_splits, evaluate, save_results

# Values are sklearn defaults unless marked "original"
BASELINE_PARAMS = dict(
    n_neighbors=20,      # original value (sklearn default is 5)
    weights='uniform',
    algorithm='auto',
    leaf_size=30,
    p=2,                 # Euclidean distance
    metric='minkowski',
    n_jobs=-1,           # use all CPU cores
)

# Columns that are NOT used as model inputs
NON_FEATURE_COLS = ['userId', 'movieId', 'rating', 'split']

# Results file
RESULTS_FILE = os.path.join('results', 'model_evaluation_results.csv')

# The two datasets we compare: (name, loader function)
DATASETS = [
    ('KNN_PREPROCESSED', load_ratings_preprocessed),
    ('KNN_SCALED', load_ratings_scaled),
]


# Build the model: fill missing values first, then KNN (params overrides the baseline)
def build_model(params=None):
    merged = {**BASELINE_PARAMS, **(params or {})}
    return Pipeline([
        # users/movies not seen in train have missing stats, so fill them with the mean
        ('imputer', SimpleImputer(strategy='mean')),
        ('knn', KNeighborsRegressor(**merged)),
    ])


# Train and test KNN on both datasets, then save the results
def run_knn(params=None):
    results = []
    for label, loader in DATASETS:
        print(f"\n=== MODEL: {label} ===")
        df = loader()
        
        # 2. UNPACK SPLITS AND META
        splits, meta = get_splits(df)
        del df               
        gc.collect()

        X_train, y_train = splits['train']
        model = build_model(params)

        print(f"Training {label}...")
        
        # 3. TRACK MEMORY AND TIME DURING TRAINING
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
    run_knn()