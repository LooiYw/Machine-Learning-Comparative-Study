import os
import time
import tracemalloc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error

import utils
from preprocessing import load_ratings_scaled

# Save tuned results to a separate CSV in the same results directory
utils.RESULTS_FILE = os.path.join('results', 'tuned_model_evaluation_results.csv')
PLOTS_DIR = os.path.join('results', 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def tune_lr():
    df = load_ratings_scaled()
    splits, meta = utils.get_splits(df)

    imputer = SimpleImputer(strategy='mean')
    X_train, y_train = splits['train']
    X_train = imputer.fit_transform(X_train)
    X_val, y_val = imputer.transform(splits['val'][0]), splits['val'][1]
    X_test, y_test = imputer.transform(splits['test'][0]), splits['test'][1]

    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    train_rmses, val_rmses = [], []

    print("\n=== Tuning Ridge / Linear Regression (alpha) ===")
    for alpha in alphas:
        model = Ridge(alpha=alpha, fit_intercept=True, solver='auto', random_state=42)
        model.fit(X_train, y_train)

        tr_rmse = root_mean_squared_error(y_train, np.clip(model.predict(X_train), 1, 5))
        va_rmse = root_mean_squared_error(y_val, np.clip(model.predict(X_val), 1, 5))
        train_rmses.append(tr_rmse)
        val_rmses.append(va_rmse)
        print(f"  alpha={alpha:<7} | Train RMSE={tr_rmse:.4f} | Val RMSE={va_rmse:.4f}")

    # Plot Validation Curve
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, train_rmses, marker='o', label='Train RMSE')
    plt.plot(alphas, val_rmses, marker='s', label='Validation RMSE')
    plt.xscale('log')
    plt.xlabel('Alpha (Regularization Strength, log scale)')
    plt.ylabel('RMSE')
    plt.title('Ridge Regression Validation Curve')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plot_path = os.path.join(PLOTS_DIR, 'lr_tuning_curve.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot: {plot_path}")

    # Select best alpha based strictly on Validation RMSE
    best_alpha = alphas[int(np.argmin(val_rmses))]
    best_params = {
        'alpha': best_alpha,
        'fit_intercept': True,
        'solver': 'auto',
        'random_state': 42,
    }
    print(f"\nBest LR Params: {best_params}")

    # Final fit & evaluation (now including test split)
    final_model = Ridge(**best_params)
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
        res['model'] = "LR_scaled"
        res['train_time_s'] = train_time
        res['train_mem_mb'] = train_mem_mb
        rows.append(res)

    utils.save_results(rows, best_params)
    print(f"Saved tuned LR results to {utils.RESULTS_FILE}")


if __name__ == "__main__":
    tune_lr()