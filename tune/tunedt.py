import os
import time
import tracemalloc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor

import utils
from preprocessing import load_ratings_scaled

utils.RESULTS_FILE = os.path.join('results', 'tuned_model_evaluation_results.csv')
PLOTS_DIR = os.path.join('results', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def tune_dt():
    df = load_ratings_scaled()
    splits, meta = utils.get_splits(df)

    imputer = SimpleImputer(strategy='mean')
    X_train, y_train = splits['train']
    X_train = imputer.fit_transform(X_train)
    X_val, y_val = imputer.transform(splits['val'][0]), splits['val'][1]
    X_test, y_test = imputer.transform(splits['test'][0]), splits['test'][1]

    # Stage 1: Tune max_depth
    depths = [5, 10, 15, 20, 25, 30]
    depth_tr, depth_va = [], []
    print("\n=== Tuning Decision Tree: Stage 1 (max_depth) ===")
    for d in depths:
        model = DecisionTreeRegressor(max_depth=d, random_state=42)
        model.fit(X_train, y_train)
        tr_rmse = root_mean_squared_error(y_train, np.clip(model.predict(X_train), 1, 5))
        va_rmse = root_mean_squared_error(y_val, np.clip(model.predict(X_val), 1, 5))
        depth_tr.append(tr_rmse)
        depth_va.append(va_rmse)
        print(f"  max_depth={d:<2} | Train RMSE={tr_rmse:.4f} | Val RMSE={va_rmse:.4f}")

    best_depth = depths[int(np.argmin(depth_va))]

    # Stage 2: Tune min_samples_leaf with best max_depth
    leaves = [1, 5, 10, 20, 50, 100]
    leaf_tr, leaf_va = [], []
    print(f"\n=== Tuning Decision Tree: Stage 2 (min_samples_leaf, max_depth={best_depth}) ===")
    for leaf in leaves:
        model = DecisionTreeRegressor(max_depth=best_depth, min_samples_leaf=leaf, random_state=42)
        model.fit(X_train, y_train)
        tr_rmse = root_mean_squared_error(y_train, np.clip(model.predict(X_train), 1, 5))
        va_rmse = root_mean_squared_error(y_val, np.clip(model.predict(X_val), 1, 5))
        leaf_tr.append(tr_rmse)
        leaf_va.append(va_rmse)
        print(f"  min_samples_leaf={leaf:<3} | Train RMSE={tr_rmse:.4f} | Val RMSE={va_rmse:.4f}")

    best_leaf = leaves[int(np.argmin(leaf_va))]

    # Plot both validation curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(depths, depth_tr, marker='o', label='Train RMSE')
    axes[0].plot(depths, depth_va, marker='s', label='Validation RMSE')
    axes[0].set_title('Decision Tree: max_depth')
    axes[0].set_xlabel('max_depth')
    axes[0].set_ylabel('RMSE')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(leaves, leaf_tr, marker='o', label='Train RMSE')
    axes[1].plot(leaves, leaf_va, marker='s', label='Validation RMSE')
    axes[1].set_title(f'Decision Tree: min_samples_leaf (depth={best_depth})')
    axes[1].set_xlabel('min_samples_leaf')
    axes[1].set_ylabel('RMSE')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plot_path = os.path.join(PLOTS_DIR, 'dt_tuning_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {plot_path}")

    best_params = {
        'criterion': 'squared_error',
        'max_depth': best_depth,
        'min_samples_split': 2,
        'min_samples_leaf': best_leaf,
        'random_state': 42,
    }
    print(f"\nBest DT Params: {best_params}")

    # Final fit & evaluation on train, val, test
    final_model = DecisionTreeRegressor(**best_params)
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

    utils.save_results(rows, best_params)
    print(f"Saved tuned DT results to {utils.RESULTS_FILE}")


if __name__ == "__main__":
    tune_dt()