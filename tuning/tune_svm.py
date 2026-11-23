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

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils
from utils import get_splits, evaluate, save_results
from models.svm import build_model, DATASETS, BASELINE_PARAMS

RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
TUNED_RESULTS_PATH = os.path.join(RESULTS_DIR, 'tuned_model_evaluation_results.csv')
TRIALS_DIR = os.path.join(RESULTS_DIR, 'tuning_trials')
TUNED_IMAGES_DIR = os.path.join(ROOT_DIR, 'images', 'tuning')

# Separate parameter grids (tuned one at a time)
PARAM_SWEEPS = [
    ('C', [1e-6, 1e-5, 1e-4, 1e-3, 1.0]),
    ('epsilon', [0.0, 0.1, 0.25, 0.5]),
    ('loss', ['squared_epsilon_insensitive', 'epsilon_insensitive']),
    ('tol', [1e-5, 1e-4, 1e-3, 1e-2]),
    ('intercept_scaling', [0.1, 1.0, 10.0]),
    ('fit_intercept', [True, False]),
    ('max_iter', [1000, 2000, 5000]),
]


def calc_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, np.clip(y_pred, 1, 5))))


def sample_rows(X, y, n):
    if n is None or n >= len(y):
        return X, y
    idx = np.random.default_rng(42).choice(len(y), size=n, replace=False)
    return X[idx], y[idx]


def make_cfg(base_cfg, param_name, val):
    cfg = {**base_cfg, param_name: val}
    # LinearSVR requires dual=True when loss='epsilon_insensitive'
    if cfg.get('loss') == 'epsilon_insensitive':
        cfg['dual'] = True
    else:
        cfg['dual'] = False
    return cfg


def save_val_curve(x_labels, train_rmses, val_rmses, param_name, label):
    os.makedirs(TUNED_IMAGES_DIR, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(x_labels, train_rmses, label='Train RMSE', marker='o')
    plt.plot(x_labels, val_rmses, label='Validation RMSE', marker='s')
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.title(f'SVM: {param_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(TUNED_IMAGES_DIR, f"{label.lower()}_{param_name}.png"))
    plt.close()


def save_learning_curve(best_cfg, X_train, y_train, X_val_tune, y_val_tune, label):
    sizes, train_curve, val_curve = [], [], []
    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
        X_part, y_part = sample_rows(X_train, y_train, max(int(len(y_train) * frac), 100))
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
    plt.title('SVM: Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(TUNED_IMAGES_DIR, f"{label.lower()}_learning.png"))
    plt.close()


def save_tuned_csv(label, rows, full_params):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    algo, dataset = label.split("_", 1) if "_" in label else (label, "")

    if os.path.exists(TUNED_RESULTS_PATH):
        try:
            old = pd.read_csv(TUNED_RESULTS_PATH)
            if 'Algorithm' in old.columns and 'Dataset' in old.columns:
                # Force both sides to uppercase so it always finds and deletes the old records
                mask_algo = old['Algorithm'].astype(str).str.upper() == algo.upper()
                mask_dataset = old['Dataset'].astype(str).str.upper() == dataset.upper()
                
                keep = ~(mask_algo & mask_dataset)
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


def tune_svm():
    for label, loader in DATASETS:
        if "SCALED" not in label.upper():
            continue

        print(f"\n=== TUNING {label} ===")
        df = loader()
        splits, meta = get_splits(df)
        del df
        gc.collect()

        X_train, y_train = splits['train']
        X_val, y_val = splits['val']

        X_val_tune, y_val_tune = sample_rows(X_val, y_val, 50000)
        X_tr_eval, y_tr_eval = sample_rows(X_train, y_train, 20000)

        best_cfg = {}
        trial_rows = []

        for param_name, values in PARAM_SWEEPS:
            print(f"\n--- Tuning {param_name} ---")
            x_labels, train_rmses, val_rmses = [], [], []
            best_val = float('inf')
            best_val_param = values[0]

            for val in values:
                cfg = make_cfg(best_cfg, param_name, val)
                model = build_model(cfg)
                start = time.time()
                model.fit(X_train, y_train)
                fit_time = time.time() - start

                train_rmse = calc_rmse(y_tr_eval, model.predict(X_tr_eval))
                val_rmse = calc_rmse(y_val_tune, model.predict(X_val_tune))

                x_labels.append(str(val))
                train_rmses.append(train_rmse)
                val_rmses.append(val_rmse)
                trial_rows.append({
                    'tuned_param': param_name,
                    **cfg,
                    'train_rmse': train_rmse,
                    'val_rmse': val_rmse,
                    'fit_time_s': fit_time,
                })
                print(f"  {param_name}={val} | full={cfg} -> Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

                if val_rmse < best_val:
                    best_val = val_rmse
                    best_val_param = val

            best_cfg = make_cfg(best_cfg, param_name, best_val_param)
            save_val_curve(x_labels, train_rmses, val_rmses, param_name, label)

        os.makedirs(TRIALS_DIR, exist_ok=True)
        pd.DataFrame(trial_rows).to_csv(os.path.join(TRIALS_DIR, f"{label.lower()}_trials.csv"), index=False)
        save_learning_curve(best_cfg, X_train, y_train, X_val_tune, y_val_tune, label)
        print(f"\nBest SVM config: {best_cfg}")
        print(f"Saved graphs in {TUNED_IMAGES_DIR}")

        # Final evaluation with all best parameters
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
    tune_svm()