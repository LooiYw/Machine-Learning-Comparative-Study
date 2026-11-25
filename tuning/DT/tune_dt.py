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
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

import utils
from preprocessing import load_ratings_scaled

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plot')
HISTORY_FILE = os.path.join(BASE_DIR, 'dt_tune_history.csv')
os.makedirs(PLOTS_DIR, exist_ok=True)

utils.RESULTS_FILE = os.path.join('results', 'tuned_model_evaluation_results.csv')

BASELINE_PARAMS = {
    'criterion': 'squared_error',
    'max_depth': 15,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'random_state': 42,
}


def rmse_score(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, np.clip(y_pred, 1, 5))))


def save_history(rows):
    df_hist = pd.DataFrame(rows)
    write_header = not os.path.exists(HISTORY_FILE)
    df_hist.to_csv(HISTORY_FILE, mode='a', header=write_header, index=False)


def plot_stage(values, train_rmses, val_rmses, param_name, best_val, stage_idx):
    plt.figure(figsize=(8.5, 5))
    is_numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)

    if is_numeric:
        plt.plot(values, train_rmses, marker='o', label='Train RMSE')
        plt.plot(values, val_rmses, marker='s', label='Validation RMSE')
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

    plt.title(f'Decision Tree Stage {stage_idx}: {param_name} (Best={best_val})')
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    save_path = os.path.join(PLOTS_DIR, f'dt_stage{stage_idx}_{param_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def tune_dt():
    df = load_ratings_scaled()
    splits, meta = utils.get_splits(df)

    imputer = SimpleImputer(strategy='mean')
    X_train, y_train = splits['train']
    X_train = imputer.fit_transform(X_train)
    X_val, y_val = imputer.transform(splits['val'][0]), splits['val'][1]
    X_test, y_test = imputer.transform(splits['test'][0]), splits['test'][1]

    stages = [
        ('max_depth', [5, 10, 15, 20, 25, 30]),
        ('min_samples_split', [2, 10, 20, 50, 100, 200]),
        ('min_samples_leaf', [1, 5, 10, 20, 50, 100]),
        ('criterion', ['squared_error', 'friedman_mse', 'poisson']),
    ]

    current_params = BASELINE_PARAMS.copy()
    run_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    history_rows = []

    print("\n=== Sequential Tuning: Decision Tree (DT) ===")
    for stage_idx, (param_name, values) in enumerate(stages, start=1):
        print(f"\n--- Stage {stage_idx}: Tuning {param_name} ---")
        tr_list, va_list = [], []

        for val in values:
            test_params = {**current_params, param_name: val}
            model = DecisionTreeRegressor(**test_params)
            model.fit(X_train, y_train)
            t_rmse = rmse_score(y_train, model.predict(X_train))
            v_rmse = rmse_score(y_val, model.predict(X_val))
            tr_list.append(t_rmse)
            va_list.append(v_rmse)
            print(f"  {param_name}={str(val):<14} | Train RMSE={t_rmse:.4f} | Val RMSE={v_rmse:.4f}")

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
        plot_stage(values, tr_list, va_list, param_name, best_val, stage_idx)

    save_history(history_rows)
    print(f"\nSaved tuning history to {HISTORY_FILE}")
    print(f"Best DT Params: {current_params}")

    final_model = DecisionTreeRegressor(**current_params)
    tracemalloc.start()
    t0 = time.time()
    final_model.fit(X_train, y_train)
    train_time = time.time() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    train_mem_mb = peak_mem / 1e6

    rows = []
    for split_name, (X, y) in [('train', (X_train, y_train)), ('val', (X_val, y_val)), ('test', (X_test, y_test))]:
        res = utils.evaluate(split_name, final_model, X, y, meta[split_name])
        res['model'] = "DT_scaled"
        res['train_time_s'] = train_time
        res['train_mem_mb'] = train_mem_mb
        rows.append(res)

    utils.save_results(rows, current_params)
    print(f"Saved tuned DT results to {utils.RESULTS_FILE}")


if __name__ == "__main__":
    tune_dt()