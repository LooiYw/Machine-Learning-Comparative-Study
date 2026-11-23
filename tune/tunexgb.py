import os
import time
import tracemalloc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from xgboost import XGBRegressor

import utils
from preprocessing import load_ratings_scaled

utils.RESULTS_FILE = os.path.join('results', 'tuned_model_evaluation_results.csv')
PLOTS_DIR = os.path.join('results', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

def rmse_score(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, np.clip(y_pred, 1, 5))))

def tune_xgb():
    df = load_ratings_scaled()
    splits, meta = utils.get_splits(df)

    # XGBoost handles cold-start NaNs natively
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    # 1. Tune max_depth (expanded past 10 since Val RMSE was still dropping)
    depths = [4, 6, 8, 10, 12, 14]
    train_d, val_d = [], []
    best_depth, best_val = 6, float('inf')

    for d in depths:
        model = XGBRegressor(
            n_estimators=300, learning_rate=0.1, max_depth=d,
            subsample=0.8, colsample_bytree=0.8,
            tree_method='hist', n_jobs=-1, random_state=42
        )
        model.fit(X_train, y_train)
        t_rmse = rmse_score(y_train, model.predict(X_train))
        v_rmse = rmse_score(y_val, model.predict(X_val))
        train_d.append(t_rmse)
        val_d.append(v_rmse)
        print(f"  max_depth={d:<2} | Train RMSE={t_rmse:.4f} | Val RMSE={v_rmse:.4f}")
        if v_rmse < best_val:
            best_val, best_depth = v_rmse, d

    # 2. Tune learning_rate at best_depth (expanded up to 0.4)
    lrs = [0.05, 0.1, 0.2, 0.3, 0.4]
    train_lr, val_lr = [], []
    best_lr, best_val = 0.1, float('inf')

    for lr in lrs:
        model = XGBRegressor(
            n_estimators=300, learning_rate=lr, max_depth=best_depth,
            subsample=0.8, colsample_bytree=0.8,
            tree_method='hist', n_jobs=-1, random_state=42
        )
        model.fit(X_train, y_train)
        t_rmse = rmse_score(y_train, model.predict(X_train))
        v_rmse = rmse_score(y_val, model.predict(X_val))
        train_lr.append(t_rmse)
        val_lr.append(v_rmse)
        print(f"  learning_rate={lr:<4} | Train RMSE={t_rmse:.4f} | Val RMSE={v_rmse:.4f}")
        if v_rmse < best_val:
            best_val, best_lr = v_rmse, lr

    # 3. Final model with higher n_estimators=500 to check boosting learning curve
    best_params = {
        'n_estimators': 500,
        'learning_rate': best_lr,
        'max_depth': best_depth,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'hist',
        'eval_metric': 'rmse',
        'n_jobs': -1,
        'random_state': 42,
    }
    print(f"\nBest XGB Params: {best_params}")

    # Final fit with eval_set to capture the Boosting Learning Curve
    final_model = XGBRegressor(**best_params)
    tracemalloc.start()
    t0 = time.time()
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    train_time = time.time() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    train_mem_mb = peak_mem / 1e6

    evals_result = final_model.evals_result()
    rounds = range(1, len(evals_result['validation_0']['rmse']) + 1)

    # Plot Validation Curves + Boosting Learning Curve
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(depths, train_d, marker='o', label='Train RMSE')
    axes[0].plot(depths, val_d, marker='s', label='Validation RMSE')
    axes[0].set_title('XGBoost: max_depth')
    axes[0].set_xlabel('max_depth')
    axes[0].set_ylabel('RMSE')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(lrs, train_lr, marker='o', label='Train RMSE')
    axes[1].plot(lrs, val_lr, marker='s', label='Validation RMSE')
    axes[1].set_title(f'XGBoost: learning_rate (depth={best_depth})')
    axes[1].set_ylabel('RMSE')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].plot(rounds, evals_result['validation_0']['rmse'], label='Train RMSE')
    axes[2].plot(rounds, evals_result['validation_1']['rmse'], label='Validation RMSE')
    axes[2].set_title('XGBoost Learning Curve (RMSE vs Rounds)')
    axes[2].set_xlabel('Boosting Rounds (n_estimators)')
    axes[2].set_ylabel('RMSE')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plot_path = os.path.join(PLOTS_DIR, 'xgb_tuning_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {plot_path}")

    # Final evaluation on train, val, test
    rows = []
    for split_name, (X, y) in [('train', (X_train, y_train)), ('val', (X_val, y_val)), ('test', (X_test, y_test))]:
        res = utils.evaluate(split_name, final_model, X, y, meta[split_name])
        res['model'] = "XGB_scaled"
        res['train_time_s'] = train_time
        res['train_mem_mb'] = train_mem_mb
        rows.append(res)

    utils.save_results(rows, best_params)
    print(f"Saved tuned XGB results to {utils.RESULTS_FILE}")


if __name__ == "__main__":
    tune_xgb()