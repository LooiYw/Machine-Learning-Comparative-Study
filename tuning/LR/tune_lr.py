import datetime
import json
import os
import time
import tracemalloc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import utils
from preprocessing import load_ratings_scaled

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plot')
HISTORY_FILE = os.path.join(BASE_DIR, 'lr_tune_history.csv')
os.makedirs(PLOTS_DIR, exist_ok=True)

utils.RESULTS_FILE = os.path.join('results', 'tuned_model_evaluation_results.csv')

BASELINE_PARAMS = {
    'alpha': 1.0,
    'fit_intercept': True,
    'solver': 'auto',
    'random_state': 42,
}


def rmse_score(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, np.clip(y_pred, 1, 5))))


def _take_rows(data, idx):
    return data.iloc[idx] if hasattr(data, 'iloc') else data[idx]


def save_history(rows):
    df_hist = pd.DataFrame(rows)
    write_header = not os.path.exists(HISTORY_FILE)
    df_hist.to_csv(HISTORY_FILE, mode='a', header=write_header, index=False)


def plot_stage(values, train_rmses, val_rmses, param_name, best_val, stage_idx, log_x=False):
    plt.figure(figsize=(8.5, 5))
    is_numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)

    if is_numeric:
        plot_fn = plt.semilogx if log_x else plt.plot
        plot_fn(values, train_rmses, marker='o', label='Train RMSE')
        plot_fn(values, val_rmses, marker='s', label='Validation RMSE')
        plt.axvline(best_val, color='r', linestyle='--', label=f'Best={best_val}')
        for x_val, y_val in zip(values, val_rmses):
            plt.annotate(f"{y_val:.4f}", (x_val, y_val), textcoords="offset points",
                         xytext=(0, 8), ha='center', fontsize=8, color='darkorange')
    else:
        labels = [str(v) for v in values]
        x = np.arange(len(labels))
        width = 0.35
        bars_tr = plt.bar(x - width / 2, train_rmses, width, label='Train RMSE', color='#1f77b4')
        bars_va = plt.bar(x + width / 2, val_rmses, width, label='Validation RMSE', color='#ff7f0e')

        best_idx = labels.index(str(best_val))
        bars_va[best_idx].set_edgecolor('red')
        bars_va[best_idx].set_linewidth(2.0)

        for bar in bars_tr:
            h = bar.get_height()
            plt.annotate(f"{h:.4f}", (bar.get_x() + bar.get_width() / 2, h),
                         textcoords="offset points", xytext=(0, 4), ha='center', fontsize=8)
        for bar in bars_va:
            h = bar.get_height()
            plt.annotate(f"{h:.4f}", (bar.get_x() + bar.get_width() / 2, h),
                         textcoords="offset points", xytext=(0, 4), ha='center', fontsize=8)

        all_vals = train_rmses + val_rmses
        span = max(all_vals) - min(all_vals)
        margin = max(span * 0.3, 0.002)
        plt.ylim(max(0, min(all_vals) - margin), max(all_vals) + margin * 1.4)
        plt.xticks(x, labels)

    plt.title(f'Ridge (LR) Stage {stage_idx}: {param_name} (Best={best_val})')
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    save_path = os.path.join(PLOTS_DIR, f'lr_stage{stage_idx}_{param_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_search_progress(history_rows):
    df_run = pd.DataFrame(history_rows)
    trials = np.arange(1, len(df_run) + 1)
    tr_rmses = df_run['Train RMSE'].values
    va_rmses = df_run['Val RMSE'].values
    best_so_far = np.minimum.accumulate(va_rmses)

    plt.figure(figsize=(8.5, 5))
    plt.plot(trials, tr_rmses, marker='o', color='#1f77b4', label='Train RMSE')
    plt.plot(trials, va_rmses, marker='s', color='#ff7f0e', label='Validation RMSE')
    plt.plot(trials, best_so_far, linestyle='--', color='#2ca02c', linewidth=2,
             label=f'Best Val RMSE So Far ({best_so_far[-1]:.4f})')
    plt.title('Ridge Regression: Sequential Search Progress')
    plt.xlabel('Sequential Trial')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(PLOTS_DIR, 'lr_search_progress.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_learning_curve(best_params, X_train_raw, y_train, X_val_raw, y_val,
                        final_train_rmse, final_val_rmse):
    print("\n--- Generating Learning Curve (RMSE vs. Training Set Size) ---")
    n_total = len(y_train)
    rng = np.random.RandomState(42)
    shuffled_idx = rng.permutation(n_total)

    sizes, tr_list, va_list = [], [], []
    for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
        if frac == 1.0:
            n_sub = n_total
            t_r, v_r = final_train_rmse, final_val_rmse
        else:
            n_sub = int(round(n_total * frac))
            sub_idx = shuffled_idx[:n_sub]
            imp = SimpleImputer(strategy='mean')
            X_sub = imp.fit_transform(_take_rows(X_train_raw, sub_idx))
            y_sub = _take_rows(y_train, sub_idx)
            X_val_cur = imp.transform(X_val_raw)

            m = Ridge(**best_params)
            m.fit(X_sub, y_sub)
            t_r = rmse_score(y_sub, m.predict(X_sub))
            v_r = rmse_score(y_val, m.predict(X_val_cur))

        sizes.append(n_sub)
        tr_list.append(t_r)
        va_list.append(v_r)
        print(f"  Size={n_sub:<7} ({int(frac * 100):>3}%) | Train={t_r:.4f} | Val={v_r:.4f}")

    plt.figure(figsize=(8.5, 5))
    plt.plot(sizes, tr_list, marker='o', color='#1f77b4', label='Train RMSE')
    plt.plot(sizes, va_list, marker='s', color='#ff7f0e', label='Validation RMSE')
    plt.title('Ridge Regression: Learning Curve')
    plt.xlabel('Training Set Size')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(PLOTS_DIR, 'lr_learning_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def tune_lr():
    df = load_ratings_scaled()
    splits, meta = utils.get_splits(df)

    X_train_raw, y_train = splits['train']
    X_val_raw, y_val = splits['val']
    

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train_raw)
    X_val = imputer.transform(X_val_raw)
    

    stages = [
        ('alpha', [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0], True),
        ('fit_intercept', [True, False], False),
        ('solver', ['cholesky', 'auto', 'svd', 'lsqr'], False),
    ]

    current_params = BASELINE_PARAMS.copy()
    run_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    history_rows = []

    print("\n=== Sequential Tuning: Ridge Regression (LR) ===")
    for stage_idx, (param_name, values, log_x) in enumerate(stages, start=1):
        print(f"\n--- Stage {stage_idx}: Tuning {param_name} ---")
        tr_list, va_list = [], []

        for val in values:
            test_params = {**current_params, param_name: val}
            model = Ridge(**test_params)
            model.fit(X_train, y_train)
            t_rmse = rmse_score(y_train, model.predict(X_train))
            v_rmse = rmse_score(y_val, model.predict(X_val))
            tr_list.append(t_rmse)
            va_list.append(v_rmse)
            print(f"  {param_name}={str(val):<10} | Train RMSE={t_rmse:.4f} | Val RMSE={v_rmse:.4f}")

            history_rows.append({
                'Timestamp': run_time,
                'Stage': stage_idx,
                'Parameter': param_name,
                'Value': str(val),
                'Train RMSE': round(t_rmse, 4),
                'Val RMSE': round(v_rmse, 4),
                'Params Snapshot': json.dumps(test_params, default=str),
            })

        best_val = values[int(np.argmin(va_list))]
        current_params[param_name] = best_val
        print(f"  -> Locked {param_name} = {best_val} (Val RMSE={min(va_list):.4f})")
        plot_stage(values, tr_list, va_list, param_name, best_val, stage_idx, log_x)

    save_history(history_rows)
    plot_search_progress(history_rows)
    print(f"\nSaved tuning history to {HISTORY_FILE}")
    print(f"Best LR Params: {current_params}")

    final_model = Ridge(**current_params)
    tracemalloc.start()
    t0 = time.time()
    final_model.fit(X_train, y_train)
    train_time = time.time() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    train_mem_mb = peak_mem / 1e6

    final_train_rmse = rmse_score(y_train, final_model.predict(X_train))
    final_val_rmse = rmse_score(y_val, final_model.predict(X_val))
    plot_learning_curve(current_params, X_train_raw, y_train, X_val_raw, y_val,
                        final_train_rmse, final_val_rmse)

    rows = []
    for split_name, (X, y) in [('train', (X_train, y_train)), ('val', (X_val, y_val))]:
        res = utils.evaluate(split_name, final_model, X, y, meta[split_name])
        res['model'] = "LR_scaled"
        res['train_time_s'] = train_time
        res['train_mem_mb'] = train_mem_mb
        rows.append(res)

    utils.save_results(rows, current_params)
    print(f"Saved tuned LR results to {utils.RESULTS_FILE}")


if __name__ == "__main__":
    tune_lr()