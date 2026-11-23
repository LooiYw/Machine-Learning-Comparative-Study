import gc
from xgboost import XGBRegressor

from utils import evaluate, get_splits, save_results, track_resources
from preprocessing import load_ratings_preprocessed, load_ratings_scaled

# Baseline parameters (Ref: IJRTI, 2026 - max_depth=6; Bendiaf et al., 2026, Emerald AJIM)
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


def run_xgb(df=None, dataset_label=None, params=None):
    if df is None:
        run_xgb(load_ratings_preprocessed(), "preprocessed", params=params)
        gc.collect()
        run_xgb(load_ratings_scaled(), "scaled", params=params)
        return
    params = params or BASELINE_PARAMS
    splits, meta = get_splits(df)

    # XGBoost handles cold-start NaNs natively
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    print(f"\n=== XGB ({dataset_label}) ===")

    # Time and memory for training, measured the same way as for every other model.
    # The model is created inside the block so its allocations are counted too.
    with track_resources() as res:
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)
    train_time = res['time_s']
    train_mem_mb = res['mem_mb']
    print(f"  Trained in {train_time:.2f}s | Peak Train Mem: {train_mem_mb:.2f}MB")

    rows = []
    for split_name, (X, y) in [('train', (X_train, y_train)), ('val', (X_val, y_val)), ('test', (X_test, y_test))]:
        res = evaluate(split_name, model, X, y, meta[split_name])
        res['model'] = f"XGB_{dataset_label}"
        res['train_time_s'] = train_time
        res['train_mem_mb'] = train_mem_mb
        rows.append(res)

    save_results(rows, params)
    return model, rows


if __name__ == "__main__":
    run_xgb()