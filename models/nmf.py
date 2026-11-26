import gc
import json
import os
import time
import tracemalloc
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from preprocessing import load_ratings_preprocessed, load_ratings_scaled
from utils import get_splits, evaluate, save_results

# Values are sklearn defaults
NMF_BASELINE = dict(
    n_components=None,
    init=None,
    solver='cd',
    beta_loss='frobenius',
    tol=1e-4,
    max_iter=200,
    alpha_W=0.0,
    alpha_H='same',
    l1_ratio=0.0,
    random_state=42,
)

RIDGE_BASELINE = dict(
    alpha=1.0,
    fit_intercept=True,
    solver='auto',
    random_state=42,
)

# Columns that are NOT used as model inputs
NON_FEATURE_COLS = ['userId', 'movieId', 'rating', 'split']

# Results file
RESULTS_FILE = os.path.join('results', 'model_evaluation_results.csv')

# The two datasets we compare: (name, loader function, has negative numbers?)
DATASETS = [
    ('NMF_PREPROCESSED', load_ratings_preprocessed, False),
    ('NMF_SCALED', load_ratings_scaled, True),
]


# Build the model: fill missing values, (optional MinMax), NMF, then Ridge
def build_model(nmf_params=None, ridge_params=None, make_nonnegative=False):
    # users/movies not seen in train have missing stats, so fill them with the mean
    steps = [('imputer', SimpleImputer(strategy='mean'))]
    if make_nonnegative:
        # clip=True keeps val/test values inside 0-1, so NMF never gets a negative number
        steps.append(('nonneg', MinMaxScaler(clip=True)))
    steps.append(('nmf', NMF(**{**NMF_BASELINE, **(nmf_params or {})})))
    steps.append(('ridge', Ridge(**{**RIDGE_BASELINE, **(ridge_params or {})})))
    return Pipeline(steps)


# Train and test NMF + Ridge on both datasets, then save the results
def run_nmf(nmf_params=None, ridge_params=None):
    results = []
    # NMF dataset list includes the has_negatives boolean
    for label, loader, has_negatives in DATASETS:
        print(f"\n=== MODEL: {label} ===")
        df = loader()
        
        # Unpack splits and meta
        splits, meta = get_splits(df)
        del df               
        gc.collect()

        X_train, y_train = splits['train']
        
        # Build model handles the has_negatives flag
        model = build_model(nmf_params, ridge_params, make_nonnegative=has_negatives)

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
        used = {'nmf': {**NMF_BASELINE, **(nmf_params or {})},
                'ridge': {**RIDGE_BASELINE, **(ridge_params or {})}}
        save_results(results[-3:], used)
    return results

if __name__ == "__main__":
    run_nmf()
