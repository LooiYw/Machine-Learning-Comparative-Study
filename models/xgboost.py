import time
import tracemalloc
import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

from preprocessing import load_ratings_preprocessed, load_ratings_scaled

# Baseline parameters (Ref: IJRTI, 2026 - max_depth=6; Emerald AJIM, 2026)
BASELINE_PARAMS = {
    'n_estimators': 100,        # Tunable: e.g., 100, 300, 500
    'learning_rate': 0.3,       # Official default (tunable: e.g., 0.05, 0.1, 0.3)
    'max_depth': 6,             # Official default (tunable: e.g., 4, 6, 8, 10)
    'subsample': 1.0,           # Tunable: e.g., 0.8, 1.0
    'colsample_bytree': 1.0,    # Tunable: e.g., 0.8, 1.0
    'tree_method': 'hist',      # Histogram-based splitting for speed on 1M rows
    'n_jobs': -1,
    'random_state': 42,
}

DROP_COLS = ['userId', 'movieId', 'rating', 'split']


def prepare_splits(df, eval_split='test'):
    train_df = df[df['split'] == 'train']
    eval_df = df[df['split'] == eval_split].copy()

    X_train = train_df.drop(columns=DROP_COLS)
    y_train = train_df['rating'].to_numpy()
    X_eval = eval_df.drop(columns=DROP_COLS)
    y_eval = eval_df['rating'].to_numpy()

    return X_train, y_train, X_eval, y_eval, eval_df[['userId', 'movieId', 'rating']]


def compute_ranking_metrics(eval_meta, y_pred, k=20, threshold=4.0):
    df = eval_meta.copy()
    df['pred'] = y_pred
    precisions, recalls = [], []

    for _, group in df.groupby('userId'):
        relevant = group[group['rating'] >= threshold]
        if len(relevant) == 0:
            continue
        top_k = group.nlargest(k, 'pred')
        hits = (top_k['rating'] >= threshold).sum()
        precisions.append(hits / min(k, len(group)))
        recalls.append(hits / len(relevant))

    return float(np.mean(precisions)), float(np.mean(recalls))


def run_xgb(df, dataset_label, params=None, eval_split='test'):
    params = params or BASELINE_PARAMS
    X_train, y_train, X_eval, y_eval, eval_meta = prepare_splits(df, eval_split)

    model = XGBRegressor(**params)

    tracemalloc.start()
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t1 = time.time()
    y_pred = np.clip(model.predict(X_eval), 1.0, 5.0)
    pred_time = time.time() - t1

    rmse = root_mean_squared_error(y_eval, y_pred)
    mae = mean_absolute_error(y_eval, y_pred)
    p20, r20 = compute_ranking_metrics(eval_meta, y_pred, k=20)

    print(f"\n--- XGBoost ({dataset_label} | {eval_split}) ---")
    print(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | P@20: {p20:.4f} | R@20: {r20:.4f}")
    print(f"Train: {train_time:.2f}s | Predict: {pred_time:.4f}s | Peak Mem: {peak_mem / 1e6:.2f} MB")
    return model, {'rmse': rmse, 'mae': mae, 'p20': p20, 'r20': r20, 'train_time': train_time, 'pred_time': pred_time, 'peak_mem_mb': peak_mem / 1e6}


if __name__ == "__main__":
    run_xgb(load_ratings_preprocessed(), "Preprocessed (Unscaled)")
    run_xgb(load_ratings_scaled(), "Scaled")