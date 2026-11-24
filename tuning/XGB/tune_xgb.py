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
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

import utils
from preprocessing import load_ratings_scaled

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plot')
HISTORY_FILE = os.path.join(BASE_DIR, 'xgb_tune_history.csv')
os.makedirs(PLOTS_DIR, exist_ok=True)

utils.RESULTS_FILE = os.path.join('results', 'tuned_model_evaluation_results.csv')

BASELINE_PARAMS = {
    'n_estimators': 100,
    'learning_rate': 0.3,
    'max_depth': 6,
    'subsample': 1.0,
    'colsample_bytree': 1.0,
    'tree_method': 'hist',
    'n_jobs': -1,
    'random_state': 42,
}


def rmse_score(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, np.clip(y_pred, 1, 5))))


def save_history(rows):
    df_hist = pd.DataFrame(rows)
    write_header = not os.path.exists(HISTORY_FILE)
    df_hist.to_csv(HISTORY_FILE, mode='a', header=write_header, index=False)


def plot_stage(values, train_rmses, val_rmses, param_name, best_val, stage_idx):
    plt.figure(figsize=(8, 5))
    plt.plot(values, train_rmses, marker='o', label='Train RMSE')
    plt.plot(values, val_rmses, marker='s', label='Validation RMSE')
    plt.axvline(best_val, color='r', linestyle='--', label=f'Best={best_val}')
    plt.title(f'XGBoost Stage {stage_idx}: {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    save_path = os.path.join(PLOTS_DIR, f'xgb_stage{stage_idx}_{param_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def tune_xgb():
    df = load_ratings_scaled()
    splits, meta = utils.get_splits(df)

    # XGBoost handles cold-start NaNs natively
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    stages = [
        ('max_depth', [4, 6, 8, 10, 12, 14]),
        ('learning_rate', [0.05, 0.1, 0.2, 0.3, 0.4]),
        ('n_estimators', [100, 200, 300, 500]),
        ('subsample', [0.6, 0.8, 1.0]),
        ('colsample_bytree', [0.6, 0.8, 1.0]),
    ]

    current_params = BASELINE_PARAMS.copy()
    run_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    history_rows = []

    print("\n=== Sequential Tuning: XGBoost (XGB) ===")
    for stage_idx, (param_name, values) in enumerate(stages, start=1):
        print(f"\n--- Stage {stage_idx}: Tuning {param_name} ---")
        tr_list, va_list = [], []

        for val in values:
            test_params = {**current_params, param_name: val}
            model = XGBRegressor(**test_params)
            model.fit(X_train, y_train)
            t_rmse = rmse_score(y_train, model.predict(X_train))
            v_rmse = rmse_score(y_val, model.predict(X_val))
            tr_list.append(t_rmse)
            va_list.append(v_rmse)
            print(f"  {param_name}={str(val):<6} | Train RMSE={t_rmse:.4f} | Val RMSE={v_rmse:.4f}")

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

    final_params = {**current_params, 'eval_metric': 'rmse'}
    print(f"Best XGB Params: {final_params}")

    # Final fit with eval_set to capture Boosting Learning Curve
    final_model = XGBRegressor(**final_params)
    tracemalloc.start()
    t0 = time.time()
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    train_time = time.time() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    train_mem_mb = peak_mem / 1e6

    # Save Boosting Learning Curve
    evals_result = final_model.evals_result()
    rounds = range(1, len(evals_result['validation_0']['rmse']) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, evals_result['validation_0']['rmse'], label='Train RMSE')
    plt.plot(rounds, evals_result['validation_1']['rmse'], label='Validation RMSE')
    plt.title('XGBoost Final Learning Curve (RMSE vs Boosting Rounds)')
    plt.xlabel('Boosting Rounds (n_estimators)')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(PLOTS_DIR, 'xgb_learning_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    rows = []
    for split_name, (X, y) in [('train', (X_train, y_train)), ('val', (X_val, y_val)), ('test', (X_test, y_test))]:
        res = utils.evaluate(split_name, final_model, X, y, meta[split_name])
        res['model'] = "XGB_scaled"
        res['train_time_s'] = train_time
        res['train_mem_mb'] = train_mem_mb
        rows.append(res)

    utils.save_results(rows, final_params)
    print(f"Saved tuned XGB results to {utils.RESULTS_FILE}")


if __name__ == "__main__":
    tune_xgb()