# import gc
# import os
# import sys
# import time
# import tracemalloc
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from sklearn.metrics import mean_squared_error

# import utils
# from utils import evaluate, save_results

# # Always save next to utils.py, no matter where you run the terminal from
# ROOT_DIR = os.path.dirname(os.path.abspath(utils.__file__))
# if ROOT_DIR not in sys.path:
#     sys.path.insert(0, ROOT_DIR)

# RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
# TUNED_RESULTS_PATH = os.path.join(RESULTS_DIR, 'tuned_model_evaluation_results.csv')
# ORIGINAL_RESULTS_PATH = os.path.join(RESULTS_DIR, 'model_evaluation_results.csv')
# TRIALS_DIR = os.path.join(RESULTS_DIR, 'tuning_trials')
# TUNED_IMAGES_DIR = os.path.join(ROOT_DIR, 'images', 'tuning')


# # RMSE with predictions clipped to 1-5 stars (same as utils.evaluate, so numbers match)
# def calc_rmse(y_true, y_pred):
#     y_pred = np.clip(y_pred, 1, 5)
#     return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# # Take n random rows (makes tuning faster). n=None means use everything.
# def sample_rows(X, y, n):
#     if n is None or n >= len(y):
#         return X, y
#     idx = np.random.default_rng(42).choice(len(y), size=n, replace=False)
#     return X[idx], y[idx]


# # Turn a config like {'n_components': 10, 'alpha': 1.0} into text for the plot axis
# def cfg_text(cfg):
#     return ', '.join(f"{k}={v}" for k, v in cfg.items())


# # Plot train vs validation RMSE for every setting tried
# def plot_validation_curve(labels, train_rmses, val_rmses, param_name, model_label, filename):
#     os.makedirs(TUNED_IMAGES_DIR, exist_ok=True)
#     plt.figure(figsize=(9, 5))
#     plt.plot(labels, train_rmses, label='Train RMSE', marker='o')
#     plt.plot(labels, val_rmses, label='Validation RMSE', marker='s')
#     plt.xlabel(param_name)
#     plt.ylabel('RMSE')
#     plt.title(f'{model_label} - Validation Curve')
#     plt.xticks(rotation=30, ha='right')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     path = os.path.join(TUNED_IMAGES_DIR, filename)
#     plt.savefig(path)
#     plt.close()
#     print(f"  Saved validation curve: {path}")


# # Learning curve: train on more and more rows, compare train vs VALIDATION split RMSE
# def plot_learning_curve(model_builder, X_train, y_train, X_val, y_val, model_label, filename):
#     os.makedirs(TUNED_IMAGES_DIR, exist_ok=True)
#     sizes, train_curve, val_curve = [], [], []
#     for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
#         X_part, y_part = sample_rows(X_train, y_train, max(int(len(y_train) * frac), 100))
#         model = model_builder()
#         model.fit(X_part, y_part)
#         X_eval, y_eval = sample_rows(X_part, y_part, 20000)  # small sample, keeps it fast
#         sizes.append(len(y_part))
#         train_curve.append(calc_rmse(y_eval, model.predict(X_eval)))
#         val_curve.append(calc_rmse(y_val, model.predict(X_val)))
#         print(f"  learning curve rows={len(y_part):,}: train={train_curve[-1]:.4f} val={val_curve[-1]:.4f}")

#     plt.figure(figsize=(8, 5))
#     plt.plot(sizes, train_curve, label='Train RMSE', marker='o')
#     plt.plot(sizes, val_curve, label='Validation RMSE', marker='s')
#     plt.xlabel('Training Set Size')
#     plt.ylabel('RMSE')
#     plt.title(f'{model_label} - Learning Curve (Best Params)')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     path = os.path.join(TUNED_IMAGES_DIR, filename)
#     plt.savefig(path)
#     plt.close()
#     print(f"  Saved learning curve: {path}")


# # Save rows into tuned_model_evaluation_results.csv (same columns as the original results file)
# def update_tuned_results(model_label, rows, full_params):
#     os.makedirs(RESULTS_DIR, exist_ok=True)

#     # Extract algorithm and dataset from model_label (e.g. "SVM_PREPROCESSED" -> "SVM", "preprocessed")
#     if "_" in model_label:
#         algo, dataset = model_label.split("_", 1)
#     else:
#         algo, dataset = model_label, ""

#     # Remove old rows of this model first, so re-running does not create duplicates
#     if os.path.exists(TUNED_RESULTS_PATH):
#         try:
#             old = pd.read_csv(TUNED_RESULTS_PATH)
#             # Only filter if the file has data and the columns exist
#             if 'Algorithm' in old.columns and 'Dataset' in old.columns:
#                 keep = ~((old['Algorithm'] == algo) & (old['Dataset'] == dataset.lower()))
#                 old = old[keep]
#                 # Save the cleaned file BACK to disk before appending new results
#                 old.to_csv(TUNED_RESULTS_PATH, index=False)
#         except pd.errors.EmptyDataError:
#             pass  # The file is completely blank, safe to ignore

#     # utils.save_results writes to utils.RESULTS_FILE, so point it to the tuned file for a moment
#     original_path = utils.RESULTS_FILE
#     utils.RESULTS_FILE = TUNED_RESULTS_PATH
#     try:
#         save_results(rows, full_params)
#     finally:
#         utils.RESULTS_FILE = original_path
#     print(f"  Saved results: {TUNED_RESULTS_PATH}")
    

# def tune_and_save(label, build, grid, splits, meta, full_params, param_name, tune_rows=None):
#     X_train, y_train = splits['train']
#     X_val, y_val = splits['val']

#     # smaller samples make the search faster (important for KNN / NMF)
#     X_tune, y_tune = sample_rows(X_train, y_train, tune_rows)
#     X_val_tune, y_val_tune = sample_rows(X_val, y_val, 50000)
#     X_tr_eval, y_tr_eval = sample_rows(X_tune, y_tune, 20000)
#     print(f"  Searching with {len(y_tune):,} train rows and {len(y_val_tune):,} validation rows")

#     # 1. try every config: fit on TRAIN, score on VALIDATION (RMSE)
#     labels, train_rmses, val_rmses, trial_rows = [], [], [], []
#     best_cfg, best_val = None, float('inf')
#     for cfg in grid:
#         try:
#             model = build(cfg)
#             start = time.time()
#             model.fit(X_tune, y_tune)
#             fit_time = time.time() - start
#             train_rmse = calc_rmse(y_tr_eval, model.predict(X_tr_eval))
#             val_rmse = calc_rmse(y_val_tune, model.predict(X_val_tune))
#         except Exception as e:
#             # one bad setting (e.g. too many components) should not stop the whole run
#             print(f"  {cfg} SKIPPED: {e}")
#             continue
#         labels.append(cfg_text(cfg))
#         train_rmses.append(train_rmse)
#         val_rmses.append(val_rmse)
#         trial_rows.append({**cfg, 'train_rmse': train_rmse, 'val_rmse': val_rmse, 'fit_time_s': fit_time})
#         print(f"  {cfg} -> Train RMSE: {train_rmse:.4f} | Val RMSE: {val_rmse:.4f} | {fit_time:.1f}s")

#         if val_rmse < best_val:  # lowest VALIDATION RMSE wins
#             best_val, best_cfg = val_rmse, cfg
#     if best_cfg is None:
#         raise RuntimeError(f"All settings failed for {label}. Check the SKIPPED messages above.")
#     print(f"  BEST: {best_cfg} (val RMSE={best_val:.4f})")

#     # 2. save the trial table and validation curve right away (so they exist even if a later step fails)
#     os.makedirs(TRIALS_DIR, exist_ok=True)
#     trials_path = os.path.join(TRIALS_DIR, f"{label.lower()}_trials.csv")
#     pd.DataFrame(trial_rows).to_csv(trials_path, index=False)
#     print(f"  Saved trials: {trials_path}")
#     plot_validation_curve(labels, train_rmses, val_rmses, param_name, label, f"{label.lower()}_val_curve.png")

#     # 3. learning curve with the best settings
#     plot_learning_curve(lambda: build(best_cfg), X_tune, y_tune, X_val_tune, y_val_tune,
#                         label, f"{label.lower()}_learning_curve.png")

#     # 4. final model: train on ALL train rows, then evaluate train / val / test (test used here only)
#     best_model = build(best_cfg)
#     tracemalloc.start()
#     start = time.time()
#     best_model.fit(X_train, y_train)
#     train_time = time.time() - start
#     _, peak_mem = tracemalloc.get_traced_memory()
#     tracemalloc.stop()
#     train_mem_mb = peak_mem / 1e6
#     print(f"  Final train time: {train_time:.2f}s | memory: {train_mem_mb:.2f}MB")

#     rows = []
#     for split_name in ('train', 'val', 'test'):
#         X_s, y_s = splits[split_name]
#         res = evaluate(split_name, best_model, X_s, y_s, meta[split_name])
#         res.update(model=label, train_time_s=train_time, train_mem_mb=train_mem_mb)
#         rows.append(res)
#     update_tuned_results(label, rows, full_params(best_cfg))

#     if not os.path.exists(ORIGINAL_RESULTS_PATH):
#         print(f"  NOTE: {ORIGINAL_RESULTS_PATH} not found - was your baseline run from another folder?")
#     gc.collect()