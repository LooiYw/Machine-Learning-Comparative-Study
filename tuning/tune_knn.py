import gc
import os
import sys
import time
import tracemalloc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Add project root so imports work from inside tune/
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils
from utils import get_splits, evaluate, save_results
from models.knn import build_model, DATASETS, BASELINE_PARAMS

RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
TUNED_RESULTS_PATH = os.path.join(RESULTS_DIR, 'tuned_model_evaluation_results.csv')
TRIALS_DIR = os.path.join(RESULTS_DIR, 'tuning_trials')
TUNED_IMAGES_DIR = os.path.join(ROOT_DIR, 'images', 'tuning')

GRID = [{'n_neighbors': k} for k in [3, 5, 10, 15, 25, 50, 75, 100]]


# Calculate clipped RMSE (1 to 5 stars)
def calc_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, np.clip(y_pred, 1, 5))))


# Sample n rows randomly to speed up tuning
def sample_rows(X, y, n):
    if n is None or n >= len(y):
        return X, y
    idx = np.random.default_rng(42).choice(len(y), size=n, replace=False)
    return X[idx], y[idx]


# Save validation and learning curves
def save_plots(labels, train_rmses, val_rmses, best_cfg, X_tune, y_tune, X_val_tune, y_val_tune, label):
    os.makedirs(TUNED_IMAGES_DIR, exist_ok=True)

    # Validation curve
    plt.figure(figsize=(9, 5))
    plt.plot(labels, train_rmses, label='Train RMSE', marker='o')
    plt.plot(labels, val_rmses, label='Validation RMSE', marker='s')
    plt.xlabel('n_neighbors')
    plt.ylabel('RMSE')
    plt.title(f'{label} - Validation Curve')
    plt.xticks(rotation=30, ha='right')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(TUNED_IMAGES_DIR, f"{label.lower()}_val_curve.png"))
    plt.close()

    # Learning curve
    sizes, train_curve, val_curve = [], [], []
    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
        X_part, y_part = sample_rows(X_tune, y_tune, max(int(len(y_tune) * frac), 100))
        model = build_model(best_cfg)
        model.fit(X_part, y_part)
        X_eval, y_eval = sample_rows(X_part, y_part, 20000)
        sizes.append(len(y_part))
        train_curve.append(calc_rmse(y_eval, model.predict(X_eval)))
        val_curve.append(calc_rmse(y_val_tune, model.predict(X_val_tune)))

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, train_curve, label='Train RMSE', marker='o')
    plt.plot(sizes, val_curve, label='Validation RMSE', marker='s')
    plt.xlabel('Training Set Size')
    plt.ylabel('RMSE')
    plt.title(f'{label} - Learning Curve (Best Params)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(TUNED_IMAGES_DIR, f"{label.lower()}_learning_curve.png"))
    plt.close()
    print(f"  Saved graphs in {TUNED_IMAGES_DIR}")


# Overwrite existing rows for this model and save to tuned CSV
def save_tuned_csv(label, rows, full_params):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    algo, dataset = label.split("_", 1) if "_" in label else (label, "")

    if os.path.exists(TUNED_RESULTS_PATH):
        try:
            old = pd.read_csv(TUNED_RESULTS_PATH)
            if 'Algorithm' in old.columns and 'Dataset' in old.columns:
                keep = ~((old['Algorithm'] == algo) & (old['Dataset'] == dataset.lower()))
                old[keep].to_csv(TUNED_RESULTS_PATH, index=False)
        except pd.errors.EmptyDataError:
            pass

    orig_file = utils.RESULTS_FILE
    utils.RESULTS_FILE = TUNED_RESULTS_PATH
    try:
        save_results(rows, full_params)
    finally:
        utils.RESULTS_FILE = orig_file
    print(f"  Saved results to {TUNED_RESULTS_PATH}")


# Tune n_neighbors for KNN across all datasets
def tune_knn():
    for label, loader in DATASETS:
        if "SCALED" not in label.upper():
            continue  # Skips the preprocessed dataset
            
        print(f"\n=== TUNING {label} ===")
        df = loader()        
        splits, meta = get_splits(df)
        del df
        gc.collect()

        X_train, y_train = splits['train']
        X_val, y_val = splits['val']

        X_tune, y_tune = sample_rows(X_train, y_train, 100000)
        X_val_tune, y_val_tune = sample_rows(X_val, y_val, 50000)
        X_tr_eval, y_tr_eval = sample_rows(X_tune, y_tune, 20000)

        labels, train_rmses, val_rmses, trial_rows = [], [], [], []
        best_cfg, best_val = None, float('inf')

        for cfg in GRID:
            model = build_model(cfg)
            start = time.time()
            model.fit(X_tune, y_tune)
            fit_time = time.time() - start

            train_rmse = calc_rmse(y_tr_eval, model.predict(X_tr_eval))
            val_rmse = calc_rmse(y_val_tune, model.predict(X_val_tune))

            labels.append(f"k={cfg['n_neighbors']}")
            train_rmses.append(train_rmse)
            val_rmses.append(val_rmse)
            trial_rows.append({**cfg, 'train_rmse': train_rmse, 'val_rmse': val_rmse, 'fit_time_s': fit_time})
            print(f"  {cfg} -> Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

            if val_rmse < best_val:
                best_val, best_cfg = val_rmse, cfg

        os.makedirs(TRIALS_DIR, exist_ok=True)
        pd.DataFrame(trial_rows).to_csv(os.path.join(TRIALS_DIR, f"{label.lower()}_trials.csv"), index=False)
        save_plots(labels, train_rmses, val_rmses, best_cfg, X_tune, y_tune, X_val_tune, y_val_tune, label)

        best_model = build_model(best_cfg)
        tracemalloc.start()
        start = time.time()
        best_model.fit(X_train, y_train)
        train_time = time.time() - start
        _, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        rows = []
        for split_name in ('train', 'val', 'test'):
            X_s, y_s = splits[split_name]
            res = evaluate(split_name, best_model, X_s, y_s, meta[split_name])
            res.update(model=label, train_time_s=train_time, train_mem_mb=peak_mem / 1e6)
            rows.append(res)

        save_tuned_csv(label, rows, {**BASELINE_PARAMS, **best_cfg})
        gc.collect()


if __name__ == "__main__":
    tune_knn()