import time
import tracemalloc
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from utils import evaluate, get_splits, save_results
from preprocessing import load_ratings_preprocessed, load_ratings_scaled

# Baseline parameters (Ref: Zhang et al., 2026, JCEIM; Cao, 2026, AJST)
BASELINE_PARAMS = {
    'n_estimators': 100,
    'criterion': 'squared_error',
    'max_depth': 15,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 1.0,
    'n_jobs': -1,
    'random_state': 42,
}


def run_rf(df=None, dataset_label=None, params=None):
    if df is None:
        run_rf(load_ratings_preprocessed(), "preprocessed", params=params)
        run_rf(load_ratings_scaled(), "scaled", params=params)
        return
    params = params or BASELINE_PARAMS
    splits, meta = get_splits(df)

    imputer = SimpleImputer(strategy='mean')
    X_train, y_train = splits['train']
    X_train = imputer.fit_transform(X_train)
    X_val, y_val = imputer.transform(splits['val'][0]), splits['val'][1]
    X_test, y_test = imputer.transform(splits['test'][0]), splits['test'][1]

    model = RandomForestRegressor(**params)

    print(f"\n=== RF ({dataset_label}) ===")
    tracemalloc.start()
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    train_mem_mb = peak_mem / 1e6
    print(f"  Trained in {train_time:.2f}s | Peak Train Mem: {train_mem_mb:.2f}MB")

    rows = []
    for split_name, (X, y) in [('train', (X_train, y_train)), ('val', (X_val, y_val)), ('test', (X_test, y_test))]:
        res = evaluate(split_name, model, X, y, meta[split_name])
        res['model'] = f"RF_{dataset_label}"
        res['train_time_s'] = train_time
        res['train_mem_mb'] = train_mem_mb
        rows.append(res)

    save_results(rows, params)
    return model, rows


if __name__ == "__main__":
    run_rf(load_ratings_preprocessed(), "preprocessed")
    run_rf(load_ratings_scaled(), "scaled")