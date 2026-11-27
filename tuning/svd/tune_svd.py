import csv
import datetime
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

# File structure
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(MODEL_DIR, '..', '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import utils
from utils import get_splits, evaluate, save_results
from models.svd import build_model, DATASETS, SVD_BASELINE, RIDGE_BASELINE

# Save model history and plots
HISTORY_CSV = os.path.join(MODEL_DIR, 'svd_tune_history.csv')

# Shared evaluation summary (append-only)
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
TUNED_RESULTS_PATH = os.path.join(RESULTS_DIR, 'tuned_model_evaluation_results.csv')

# Categorical params plotted as grouped bar charts
CATEGORICAL_PARAMS = {'ridge__fit_intercept'}

# Sequential parameter sweeps
PARAM_SWEEPS = [
    ('svd__n_components', [5, 10, 20, 30, 40, 45, 50, 52, 54, 55]),
    ('ridge__alpha', [0.01, 0.1, 1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0]),
    ('svd__n_iter', [3, 5, 10]),
    ('svd__n_oversamples', [5, 10, 20]),
    ('ridge__fit_intercept', [True, False]),
]


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


# Split combined config dict into separate SVD and Ridge parameter dicts
def split_cfg(cfg):
    svd_cfg = {k.split('__', 1)[1]: v for k, v in cfg.items() if k.startswith('svd__')}
    ridge_cfg = {k.split('__', 1)[1]: v for k, v in cfg.items() if k.startswith('ridge__')}
    return svd_cfg, ridge_cfg


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


# Plot bar chart for categorical params or line chart for numeric params and save to MODEL_DIR
def save_val_curve(x_labels, train_rmses, val_rmses, param_name, raw_values=None):
    clean_name = param_name.replace('__', '_')
    img_stem = clean_name if clean_name.startswith('svd_') else f"svd_{clean_name}"
    is_categorical = (
        param_name in CATEGORICAL_PARAMS
        or (raw_values is not None and any(isinstance(v, (str, bool)) for v in raw_values))
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    if is_categorical:
        x = np.arange(len(x_labels))
        width = 0.35
        rects1 = ax.bar(x - width / 2, train_rmses, width, label='Train RMSE', color='#4C72B0')
        rects2 = ax.bar(x + width / 2, val_rmses, width, label='Validation RMSE', color='#DD8452')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)

        all_vals = list(train_rmses) + list(val_rmses)
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
        use_log = False
        if raw_values is not None and len(raw_values) >= 3:
            nums = [float(v) for v in raw_values if float(v) > 0]
            if len(nums) == len(raw_values) and (max(nums) / min(nums)) >= 100:
                use_log = True

        if use_log:
            nums = [float(v) for v in raw_values]
            ax.plot(nums, train_rmses, label='Train RMSE', marker='o')
            ax.plot(nums, val_rmses, label='Validation RMSE', marker='s')
            ax.set_xscale('log')
        else:
            ax.plot(x_labels, train_rmses, label='Train RMSE', marker='o')
            ax.plot(x_labels, val_rmses, label='Validation RMSE', marker='s')
        ax.grid(True, linestyle='--', alpha=0.6)

    ax.set_xlabel(clean_name)
    ax.set_ylabel('RMSE')
    ax.set_title(f'SVD: {clean_name}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"{img_stem}.png"), dpi=150)
    plt.close(fig)


# Plot trial-by-trial and best-so-far RMSE across Sequential Search iterations
def save_sequential_search_curve(trial_records):
    trials = [r['trial'] for r in trial_records]
    train_rmses = [r['train_rmse'] for r in trial_records]
    val_rmses = [r['val_rmse'] for r in trial_records]
    best_so_far = np.minimum.accumulate(val_rmses)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(trials, train_rmses, label='Train RMSE', marker='o', alpha=0.75)
    ax.plot(trials, val_rmses, label='Validation RMSE', marker='s', alpha=0.85)
    ax.plot(trials, best_so_far, label='Best Val RMSE So Far', linestyle='--', color='#2CA02C', linewidth=2)
    ax.set_xlabel('Sequential Trial')
    ax.set_ylabel('RMSE')
    ax.set_title('SVD: Sequential Search Progress')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "svd_sequential_search.png"), dpi=150)
    plt.close(fig)


# Plot train vs validation RMSE across increasing training set sizes
def save_learning_curve(best_cfg, X_train, y_train, X_val_tune, y_val_tune):
    svd_cfg, ridge_cfg = split_cfg(best_cfg)
    sizes, train_curve, val_curve = [], [], []
    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
        X_part, y_part, _ = sample_rows(X_train, y_train, None, max(int(len(y_train) * frac), 100))
        model = build_model(svd_cfg, ridge_cfg)
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
    plt.title('SVD: Learning Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "svd_learning.png"), dpi=150)
    plt.close()


# Run sequential hyperparameter tuning for SVD, save plots, and log results
def tune_svd():
    for label, loader in DATASETS:
        if "SCALED" not in label.upper():
            continue

        print(f"\n=== TUNING {label} ===")
        mark_new_run(HISTORY_CSV, f"SVD | {label}")

        df = loader()
        splits, meta = get_splits(df)
        del df
        gc.collect()

        # Use train and val splits only; keep test split untouched
        X_train, y_train = splits['train']
        X_val, y_val = splits['val']
        n_features = X_train.shape[1]

        X_val_tune, y_val_tune, meta_val_tune = sample_rows(X_val, y_val, meta['val'], 50000)
        X_tr_eval, y_tr_eval, meta_tr_eval = sample_rows(X_train, y_train, meta['train'], 20000)

        best_cfg = {}
        trial_records = []
        all_baselines = {
            **{f"svd__{k}": v for k, v in SVD_BASELINE.items()},
            **{f"ridge__{k}": v for k, v in RIDGE_BASELINE.items()},
        }

        for param_name, values in PARAM_SWEEPS:
            if param_name == 'svd__n_components':
                values = [v for v in values if v < n_features]

            print(f"\n--- Tuning {param_name} ---")
            x_labels, raw_vals, train_rmses, val_rmses = [], [], [], []
            best_val = float('inf')
            best_val_param = best_cfg.get(param_name, all_baselines.get(param_name, values[0]))

            for val in values:
                cfg = {**best_cfg, param_name: val}
                svd_cfg, ridge_cfg = split_cfg(cfg)
                tracemalloc.start()
                try:
                    model = build_model(svd_cfg, ridge_cfg)
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
                gap = val_rmse - train_rmse

                x_labels.append(str(val))
                raw_vals.append(val)
                train_rmses.append(train_rmse)
                val_rmses.append(val_rmse)

                trial_rows = []
                for res in (tr_res, val_res):
                    res.update(model=label, train_time_s=fit_time, train_mem_mb=peak_mem / 1e6)
                    trial_rows.append(res)

                used = {
                    'svd': {**SVD_BASELINE, **svd_cfg},
                    'ridge': {**RIDGE_BASELINE, **ridge_cfg},
                }
                append_via_save_results(HISTORY_CSV, trial_rows, used)

                trial_records.append({'trial': len(trial_records) + 1, 'train_rmse': train_rmse, 'val_rmse': val_rmse})

                print(f"  {param_name}={val} -> Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | Gap: {gap:.4f}")

                if val_rmse < best_val:
                    best_val = val_rmse
                    best_val_param = val

            best_cfg[param_name] = best_val_param
            save_val_curve(x_labels, train_rmses, val_rmses, param_name, raw_values=raw_vals)

        if trial_records:
            save_sequential_search_curve(trial_records)

        save_learning_curve(best_cfg, X_train, y_train, X_val_tune, y_val_tune)
        print(f"\nBest SVD config: {best_cfg}")
        print(f"Saved history to {HISTORY_CSV}")
        print(f"Saved graphs in {MODEL_DIR}")

        # Final evaluation on full train and val splits (test set untouched)
        svd_cfg, ridge_cfg = split_cfg(best_cfg)
        best_model = build_model(svd_cfg, ridge_cfg)
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
            'svd': {**SVD_BASELINE, **svd_cfg},
            'ridge': {**RIDGE_BASELINE, **ridge_cfg},
        }
        mark_new_run(TUNED_RESULTS_PATH, f"SVD | {label}")
        append_via_save_results(TUNED_RESULTS_PATH, rows, full_used)
        print(f"Appended final train/val results to {TUNED_RESULTS_PATH}")
        gc.collect()


if __name__ == "__main__":
    tune_svd()