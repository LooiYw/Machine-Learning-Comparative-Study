import csv
import datetime
import gc
import itertools
import os
import sys
import time
import tracemalloc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

# File structure
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(MODEL_DIR, '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils
from utils import get_splits, evaluate, save_results
from models.knn import build_model, DATASETS, BASELINE_PARAMS

# Save model history and plots directly in tuning/knn/
HISTORY_CSV = os.path.join(MODEL_DIR, 'knn_tune_history.csv')

# Shared evaluation summary (append-only)
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
TUNED_RESULTS_PATH = os.path.join(RESULTS_DIR, 'tuned_model_evaluation_results.csv')

# Subsample training rows during grid trials for fast neighbor queries; final fit uses full train
TUNE_TRAIN_ROWS = 150000

# Categorical / discrete params plotted as grouped bar charts
CATEGORICAL_PARAMS = {'weights', 'p', 'metric', 'algorithm'}

# Exhaustive Grid Search space across KNN hyperparameters
KNN_PARAM_GRID = {
    'n_neighbors': [5, 15, 20, 30, 40, 50, 75, 100],
    'weights': ['uniform', 'distance'],
    'p': [1, 2],
    'leaf_size': [15, 30, 50],
}


# Compute RMSE after clipping predictions to valid 1-5 rating range
def calc_rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, np.clip(y_pred, 1, 5))))


# Randomly sample n rows with a fixed seed for faster evaluation
def sample_rows(X, y, meta_df, n):
    if n is None or n >= len(y):
        return X, y, meta_df
    idx = np.random.default_rng(42).choice(len(y), size=n, replace=False)
    X_sub = X.iloc[idx] if hasattr(X, 'iloc') else X[idx]
    y_sub = y.iloc[idx] if hasattr(y, 'iloc') else y[idx]
    meta_sub = meta_df.iloc[idx].reset_index(drop=True) if meta_df is not None else None
    return X_sub, y_sub, meta_sub


# Append a timestamped divider row to separate new runs in the CSV
def mark_new_run(target_path, run_tag):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    if not os.path.exists(target_path) or os.path.getsize(target_path) == 0:
        return
    with open(target_path, 'r', encoding='utf-8', newline='') as f:
        header = next(csv.reader(f), None)
    if not header:
        return
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    divider = [f"##### NEW RUN {ts} | {run_tag} #####"] + [""] * (len(header) - 1)
    with open(target_path, 'a', encoding='utf-8', newline='') as f:
        csv.writer(f).writerow(divider)


# Append evaluation rows to target CSV using utils.save_results format
def append_via_save_results(target_path, rows, full_params):
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    orig_file = utils.RESULTS_FILE
    utils.RESULTS_FILE = target_path
    try:
        save_results(rows, full_params)
    finally:
        utils.RESULTS_FILE = orig_file


# Plot multi-line comparison of Validation RMSE across n_neighbors for each (weights, p) combo
def save_grid_search_curve(trial_records):
    combos = sorted({(r['cfg']['weights'], r['cfg']['p']) for r in trial_records}, key=lambda x: (str(x[0]), x[1]))
    fig, ax = plt.subplots(figsize=(8, 5))

    for weights_val, p_val in combos:
        grouped_k = {}
        for r in trial_records:
            if r['cfg']['weights'] == weights_val and r['cfg']['p'] == p_val:
                grouped_k.setdefault(r['cfg']['n_neighbors'], []).append(r['val_rmse'])
        ks = sorted(grouped_k.keys())
        val_rmses = [float(np.mean(grouped_k[k])) for k in ks]
        metric_name = 'Manhattan (p=1)' if p_val == 1 else ('Euclidean (p=2)' if p_val == 2 else f'p={p_val}')
        ax.plot(ks, val_rmses, marker='o', label=f'{weights_val}, {metric_name}')

    ax.set_xlabel('n_neighbors')
    ax.set_ylabel('Validation RMSE')
    ax.set_title('KNN: Grid Search Validation RMSE')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "knn_grid_search.png"), dpi=150)
    plt.close(fig)


# Plot marginal parameter effects across grid trials (bar for categorical, line for numeric)
def save_param_marginal_plots(trial_records, param_names):
    for param_name in param_names:
        grouped = {}
        for r in trial_records:
            val = r['cfg'][param_name]
            grouped.setdefault(val, []).append((r['train_rmse'], r['val_rmse']))

        if len(grouped) <= 1:
            continue

        is_categorical = (
            param_name in CATEGORICAL_PARAMS
            or any(isinstance(v, (str, bool)) for v in grouped.keys())
        )

        if is_categorical:
            keys = sorted(grouped.keys(), key=lambda x: str(x))
            x_labels = [str(k) for k in keys]
            mean_tr = [float(np.mean([p[0] for p in grouped[k]])) for k in keys]
            mean_va = [float(np.mean([p[1] for p in grouped[k]])) for k in keys]

            fig, ax = plt.subplots(figsize=(8, 5))
            x = np.arange(len(x_labels))
            width = 0.35
            rects1 = ax.bar(x - width / 2, mean_tr, width, label='Mean Train RMSE', color='#4C72B0')
            rects2 = ax.bar(x + width / 2, mean_va, width, label='Mean Validation RMSE', color='#DD8452')
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels)

            all_vals = mean_tr + mean_va
            ymin, ymax = min(all_vals), max(all_vals)
            pad = max((ymax - ymin) * 0.25, 0.02)
            ax.set_ylim(max(0.0, ymin - pad), ymax + pad)

            for rect in rects1:
                h = rect.get_height()
                ax.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
            for rect in rects2:
                h = rect.get_height()
                ax.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)
        else:
            keys = sorted(grouped.keys(), key=lambda x: float(x))
            x_labels = [str(k) for k in keys]
            mean_tr = [float(np.mean([p[0] for p in grouped[k]])) for k in keys]
            mean_va = [float(np.mean([p[1] for p in grouped[k]])) for k in keys]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(x_labels, mean_tr, label='Mean Train RMSE', marker='o')
            ax.plot(x_labels, mean_va, label='Mean Validation RMSE', marker='s')
            ax.grid(True, linestyle='--', alpha=0.6)

        ax.set_xlabel(param_name)
        ax.set_ylabel('RMSE')
        ax.set_title(f'KNN: {param_name}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, f"knn_{param_name}.png"), dpi=150)
        plt.close(fig)


# Plot train vs validation RMSE across increasing training set sizes
def save_learning_curve(best_cfg, X_train, y_train, X_val_tune, y_val_tune):
    sizes, train_curve, val_curve = [], [], []
    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
        X_part, y_part, _ = sample_rows(X_train, y_train, None, max(int(len(y_train) * frac), 100))
        model = build_model(best_cfg)
        model.fit(X_part, y_part)
        X_eval, y_eval, _ = sample_rows(X_part, y_part, None, 20000)
        sizes.append(len(y_part))
        train_curve.append(calc_rmse(y_eval, model.predict(X_eval)))
        val_curve.append(calc_rmse(y_val_tune, model.predict(X_val_tune)))
        del model
        gc.collect()

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, train_curve, label='Train RMSE', marker='o')
    plt.plot(sizes, val_curve, label='Validation RMSE', marker='s')
    plt.xlabel('Training Set Size')
    plt.ylabel('RMSE')
    plt.title('KNN: Learning Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "knn_learning.png"), dpi=150)
    plt.close()


# Run exhaustive Grid Search for KNN, save plots, and log results
def tune_knn():
    for label, loader in DATASETS:
        if "SCALED" not in label.upper():
            continue

        print(f"\n=== TUNING {label} (GRID SEARCH) ===")
        mark_new_run(HISTORY_CSV, f"KNN GridSearch | {label}")

        df = loader()
        splits, meta = get_splits(df)
        del df
        gc.collect()

        # Use train and val splits only; keep test split untouched
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']

        X_val_tune, y_val_tune, meta_val_tune = sample_rows(X_val, y_val, meta['val'], 50000)
        X_tr_eval, y_tr_eval, meta_tr_eval = sample_rows(X_train, y_train, meta['train'], 20000)

        # Subsample training rows during grid trials for faster neighbor queries
        X_fit, y_fit, _ = sample_rows(X_train, y_train, None, TUNE_TRAIN_ROWS)
        print(f"Grid trials fit on {len(y_fit)} rows (full train = {len(y_train)})")

        param_names = list(KNN_PARAM_GRID.keys())
        grid_combos = [dict(zip(param_names, vals)) for vals in itertools.product(*KNN_PARAM_GRID.values())]

        best_val = float('inf')
        best_cfg = dict(BASELINE_PARAMS)
        trial_records = []

        for idx, cfg in enumerate(grid_combos, start=1):
            full_cfg = {**BASELINE_PARAMS, **cfg}
            print(f"\n[Grid {idx}/{len(grid_combos)}] cfg={cfg}", flush=True)
            model = None
            try:
                model = build_model(full_cfg)
                start = time.time()
                model.fit(X_fit, y_fit)
                fit_time = time.time() - start
                tr_res = evaluate('train', model, X_tr_eval, y_tr_eval, meta_tr_eval)
                val_res = evaluate('val', model, X_val_tune, y_val_tune, meta_val_tune)
            except Exception as e:
                print(f"  Grid {idx} SKIPPED: {e}")
                del model
                gc.collect()
                continue

            train_rmse = tr_res.get('RMSE', tr_res.get('rmse'))
            val_rmse = val_res.get('RMSE', val_res.get('rmse'))
            gap = val_rmse - train_rmse

            trial_rows = []
            for res in (tr_res, val_res):
                res.update(model=label, train_time_s=fit_time, train_mem_mb=float('nan'))
                trial_rows.append(res)

            append_via_save_results(HISTORY_CSV, trial_rows, full_cfg)

            trial_records.append({
                'trial': idx,
                'cfg': cfg,
                'train_rmse': train_rmse,
                'val_rmse': val_rmse,
            })

            print(f"  -> Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | Gap: {gap:.4f} | fit {fit_time:.1f}s")

            if val_rmse < best_val:
                best_val = val_rmse
                best_cfg = dict(full_cfg)

            del model
            gc.collect()

        if trial_records:
            save_grid_search_curve(trial_records)
            save_param_marginal_plots(trial_records, param_names)

        save_learning_curve(best_cfg, X_train, y_train, X_val_tune, y_val_tune)
        print(f"\nBest KNN config (Val RMSE={best_val:.4f}): {best_cfg}")
        print(f"Saved history to {HISTORY_CSV}")
        print(f"Saved graphs in {MODEL_DIR}")

        # Final evaluation on full train and val splits (test set untouched)
        best_model = build_model(best_cfg)
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

        mark_new_run(TUNED_RESULTS_PATH, f"KNN GridSearch | {label}")
        append_via_save_results(TUNED_RESULTS_PATH, rows, best_cfg)
        print(f"Appended final train/val results to {TUNED_RESULTS_PATH}")
        gc.collect()


if __name__ == "__main__":
    tune_knn()