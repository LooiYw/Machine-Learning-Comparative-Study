import time
import tracemalloc
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

from utils import evaluate, get_splits, save_results
from preprocessing import load_ratings_preprocessed, load_ratings_scaled

# Baseline parameters (Ref: Sharma & Dutta, 2023, JATIT; Jahnavi & Shivam, 2026, IJCRT)
BASELINE_PARAMS = {
    'alpha': 1.0,
    'fit_intercept': True,
    'solver': 'auto',
    'random_state': 42,
}


def run_lr(df=None, dataset_label=None, params=None):
    if df is None:
        run_lr(load_ratings_preprocessed(), "preprocessed", params=params)
        run_lr(load_ratings_scaled(), "scaled", params=params)
        return
    params = params or BASELINE_PARAMS
    splits, meta = get_splits(df)

    # Impute cold-start NaNs using train means only (prevents data leakage)
    imputer = SimpleImputer(strategy='mean')
    X_train, y_train = splits['train']
    X_train = imputer.fit_transform(X_train)
    X_val, y_val = imputer.transform(splits['val'][0]), splits['val'][1]
    X_test, y_test = imputer.transform(splits['test'][0]), splits['test'][1]

    model = Ridge(**params)

    print(f"\n=== LR ({dataset_label}) ===")
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
        res['model'] = f"LR_{dataset_label}"
        res['train_time_s'] = train_time
        res['train_mem_mb'] = train_mem_mb
        rows.append(res)

    save_results(rows, params)
    return model, rows


if __name__ == "__main__":
    run_lr(load_ratings_preprocessed(), "preprocessed")
    run_lr(load_ratings_scaled(), "scaled")