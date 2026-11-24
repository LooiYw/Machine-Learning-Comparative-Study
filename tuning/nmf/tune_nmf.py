import gc
import os
import sys
import time
import tracemalloc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# File structure: <project_root>/tuning/nmf/tune_nmf.py
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(MODEL_DIR, '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils
from utils import get_splits, evaluate, save_results
from models.nmf import build_model, DATASETS, NMF_BASELINE, RIDGE_BASELINE

# Model-specific outputs inside tuning/nmf/
HISTORY_CSV = os.path.join(MODEL_DIR, 'nmf_tune_history.csv')
IMAGES_DIR = os.path.join(MODEL_DIR, 'images')

# Shared evaluation summary (append-only)
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
TUNED_RESULTS_PATH = os.path.join(RESULTS_DIR, 'tuned_model_evaluation_results.csv')

# Sequential parameter sweeps across NMF and Ridge head
PARAM_SWEEPS = [
    ('nmf__n_components', [10, 20, 30, 40, 45, 50, 55]),
    ('ridge__alpha', [0.01, 0.1, 1.0, 10.0, 100.0]),
    ('nmf__alpha_W', [0.0, 0.01, 0.1, 1.0]),
    ('nmf__l1_ratio', [0.0, 0.5, 1.0]),
    ('nmf__tol', [1e-4, 1e-3]),
    ('nmf__max_iter', [300, 500, 800]),
]


def calc_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, np.clip(y_pred, 1, 5))))


def sample_rows(X, y, meta_df, n):
    if n is None or n >= len(y):
        return X, y, meta_df
    idx = np.random.default_rng(42).choice(len(y), size=n, replace=False)
    X_sub = X.iloc[idx] if hasattr(X, 'iloc') else X[idx]
    y_sub = y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
    meta_sub = meta_df.iloc[idx].reset_index(drop=True) if meta_df is not None else None
    return X_sub, y_sub, meta_sub


def split_cfg(cfg):
    nmf_cfg = {k.split('__', 1)[1]: v for k, v in cfg.items() if k.startswith('nmf__')}
    ridge_cfg = {k.split('__', 1)[1]: v for k, v in cfg.items() if k.startswith('ridge__')}
    return nmf_cfg, ridge_cfg


def save_val_curve(x_labels, train_rmses, val_rmses, param_name, label):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    clean_name = param_name.replace('__', '_')
    plt.figure(figsize=(8, 5))
    plt.plot(x_labels, train_rmses, label='Train RMSE', marker='o')
    plt.plot(x_labels, val_rmses, label='Validation RMSE', marker='s')
    plt.xlabel(clean_name)
    plt.ylabel('RMSE')
    plt.title(f'NMF: {clean_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, f"{label.lower()}_{clean_name}.png"))
    plt.close()


def save_learning_curve(best_cfg, has_negatives, X_train, y_train, X_val_tune, y_val_tune, label):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    nmf_cfg, ridge_cfg = split_cfg(best_cfg)
    sizes, train_curve, val_curve = [], [], []
    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
        X_part, y_part, _ = sample_rows(X_train, y_train, None, max(int(len(y_train) * frac), 100))
        model = build_model(nmf_cfg, ridge_cfg, make_nonnegative=has_negatives)
        model.fit(X_part, y_part)
        X_eval, y_eval, _ = sample_rows(X_part, y_part, None, 20000)
        sizes.append(len(y_part))
        train_curve.append(calc_rmse(y_eval, model.predict(X_eval)))
        val_curve.append(calc_rmse(y_val_tune, model.predict(X_val_tune)))

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, train_curve, label='Train RMSE', marker='o')
    plt.plot(sizes, val_curve, label='Validation RMSE', marker='s')
    plt.xlabel('Training Set Size')
    plt.ylabel('RMSE')
    plt.title('NMF: Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGES_DIR, f"{label.lower()}_learning.png"))
    plt.close()


def append_via_save_results(target_path, rows, full_params):
    """Appends rows using utils.save_results so headers match model_evaluation_results.csv."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    orig_file = utils.RESULTS_FILE
    utils.RESULTS_FILE = target_path
    try:
        save_results(rows, full_params)
    finally:
        utils.RESULTS_FILE = orig_file


def tune_nmf():
    for label, loader, has_negatives in DATASETS:
        if "SCALED" not in label.upper():
            continue

        print(f"\n=== TUNING {label} ===")
        df = loader()
        splits, meta = get_splits(df)
        del df
        gc.collect()

        # Strictly use train and val only; test split is completely untouched
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']
        n_features = X_train.shape[1]

        X_val_tune, y_val_tune, meta_val_tune = sample_rows(X_val, y_val, meta['val'], 50000)
        X_tr_eval, y_tr_eval, meta_tr_eval = sample_rows(X_train, y_train, meta['train'], 20000)

        best_cfg = {}
        all_baselines = {
            **{f"nmf__{k}": v for k, v in NMF_BASELINE.items()},
            **{f"ridge__{k}": v for k, v in RIDGE_BASELINE.items()},
        }

        for param_name, values in PARAM_SWEEPS:
            if param_name == 'nmf__n_components':
                values = [v for v in values if v < n_features]

            print(f"\n--- Tuning {param_name} ---")
            x_labels, train_rmses, val_rmses = [], [], []
            best_val = float('inf')
            best_val_param = best_cfg.get(param_name, all_baselines.get(param_name, values[0]))

            for val in values:
                cfg = {**best_cfg, param_name: val}
                nmf_cfg, ridge_cfg = split_cfg(cfg)
                tracemalloc.start()
                try:
                    model = build_model(nmf_cfg, ridge_cfg, make_nonnegative=has_negatives)
                    start = time.time()
                    model.fit(X_train, y_train)
                    fit_time = time.time() - start
                    _, peak_mem = tracemalloc.get_traced_memory()
                except Exception as e:
                    print(f"  {param_name}={val} SKIPPED: {e}")
                    continue
                finally:
                    if tracemalloc.is_tracing():
                        tracemalloc.stop()

                tr_res = evaluate('train', model, X_tr_eval, y_tr_eval, meta_tr_eval)
                val_res = evaluate('val', model, X_val_tune, y_val_tune, meta_val_tune)

                train_rmse = tr_res.get('RMSE', tr_res.get('rmse'))
                val_rmse = val_res.get('RMSE', val_res.get('rmse'))

                x_labels.append(str(val))
                train_rmses.append(train_rmse)
                val_rmses.append(val_rmse)

                trial_rows = []
                for res in (tr_res, val_res):
                    res.update(model=label, train_time_s=fit_time, train_mem_mb=peak_mem / 1e6)
                    trial_rows.append(res)

                used = {
                    'nmf': {**NMF_BASELINE, **nmf_cfg},
                    'ridge': {**RIDGE_BASELINE, **ridge_cfg},
                }
                append_via_save_results(HISTORY_CSV, trial_rows, used)

                print(f"  {param_name}={val} -> Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f}")

                if val_rmse < best_val:
                    best_val = val_rmse
                    best_val_param = val

            best_cfg[param_name] = best_val_param
            save_val_curve(x_labels, train_rmses, val_rmses, param_name, label)

        save_learning_curve(best_cfg, has_negatives, X_train, y_train, X_val_tune, y_val_tune, label)
        print(f"\nBest NMF config: {best_cfg}")
        print(f"Saved history to {HISTORY_CSV}")
        print(f"Saved graphs in {IMAGES_DIR}")

        # Final evaluation on full train and val splits only (test set untouched)
        nmf_cfg, ridge_cfg = split_cfg(best_cfg)
        best_model = build_model(nmf_cfg, ridge_cfg, make_nonnegative=has_negatives)
        tracemalloc.start()
        try:
            start = time.time()
            best_model.fit(X_train, y_train)
            train_time = time.time() - start
            _, peak_mem = tracemalloc.get_traced_memory()
        finally:
            if tracemalloc.is_tracing():
                tracemalloc.stop()

        rows = []
        for split_name in ('train', 'val'):
            X_s, y_s = splits[split_name]
            res = evaluate(split_name, best_model, X_s, y_s, meta[split_name])
            res.update(model=label, train_time_s=train_time, train_mem_mb=peak_mem / 1e6)
            rows.append(res)

        full_used = {
            'nmf': {**NMF_BASELINE, **nmf_cfg},
            'ridge': {**RIDGE_BASELINE, **ridge_cfg},
        }
        append_via_save_results(TUNED_RESULTS_PATH, rows, full_used)
        print(f"Appended final train/val results to {TUNED_RESULTS_PATH}")
        gc.collect()


if __name__ == "__main__":
    tune_nmf()