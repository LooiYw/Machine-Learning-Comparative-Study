import datetime
import json
import os
import time
import tracemalloc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

import utils
from preprocessing import load_ratings_scaled

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plot')
HISTORY_FILE = os.path.join(BASE_DIR, 'rf_tune_history.csv')
os.makedirs(PLOTS_DIR, exist_ok=True)

utils.RESULTS_FILE = os.path.join('results', 'tuned_model_evaluation_results.csv')

N_TRIALS = 50
SEARCH_N_ESTIMATORS = 100


def rmse_score(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, np.clip(y_pred, 1, 5))))


def _take_rows(data, idx):
    return data.iloc[idx] if hasattr(data, 'iloc') else data[idx]


def save_history(rows):
    df_hist = pd.DataFrame(rows)
    write_header = not os.path.exists(HISTORY_FILE)
    df_hist.to_csv(HISTORY_FILE, mode='a', header=write_header, index=False)


def plot_optuna_results(df_run, study, prefix, title_prefix):
    # 1. Optimization History (Val RMSE per trial + Running Best)
    trials = df_run['Trial'].values
    val_rmses = df_run['Val RMSE'].values
    best_so_far = np.minimum.accumulate(val_rmses)

    plt.figure(figsize=(8.5, 5))
    plt.scatter(trials, val_rmses, color='#ff7f0e', label='Trial Validation RMSE', zorder=3)
    plt.plot(trials, best_so_far, color='red', linewidth=2, label=f'Best So Far ({best_so_far[-1]:.4f})', zorder=4)
    plt.title(f'{title_prefix} Bayesian Optimization History (Optuna TPE)')
    plt.xlabel('Trial Number')
    plt.ylabel('Validation RMSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(PLOTS_DIR, f'{prefix}_optuna_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Hyperparameter Importance Bar Chart
    try:
        importances = optuna.importance.get_param_importances(study)
        names = list(importances.keys())[::-1]
        scores = list(importances.values())[::-1]
        plt.figure(figsize=(8, 4.5))
        bars = plt.barh(names, scores, color='#1f77b4')
        for bar in bars:
            w = bar.get_width()
            plt.annotate(f"{w:.3f}", (w, bar.get_y() + bar.get_height() / 2),
                         textcoords="offset points", xytext=(4, 0), va='center', fontsize=8.5)
        plt.xlim(0, max(scores) * 1.18)
        plt.title(f'{title_prefix} Hyperparameter Importance (fANOVA)')
        plt.xlabel('Relative Importance')
        plt.grid(True, axis='x', linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(PLOTS_DIR, f'{prefix}_optuna_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  [Info] Skipped importance plot: {e}")


def plot_n_estimators(n_list, tr_list, va_list, best_n):
    plt.figure(figsize=(8.5, 5))
    plt.plot(n_list, tr_list, marker='o', label='Train RMSE')
    plt.plot(n_list, va_list, marker='s', label='Validation RMSE')
    plt.axvline(best_n, color='r', linestyle='--', label=f'Best={best_n}')
    for x_val, y_val in zip(n_list, va_list):
        plt.annotate(f"{y_val:.4f}", (x_val, y_val), textcoords="offset points",
                     xytext=(0, 8), ha='center', fontsize=8.5, color='darkorange')
    plt.title(f'Random Forest Tree-Count Scaling (Best n_estimators={best_n})')
    plt.xlabel('n_estimators')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(PLOTS_DIR, 'rf_n_estimators_scaling.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_learning_curve(best_params, X_train_raw, y_train, X_val_raw, y_val,
                        final_train_rmse, final_val_rmse):
    print("\n--- Generating Learning Curve (RMSE vs. Training Set Size) ---")
    n_total = len(y_train)
    rng = np.random.RandomState(42)
    shuffled_idx = rng.permutation(n_total)

    sizes, tr_list, va_list = [], [], []
    for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
        if frac == 1.0:
            n_sub = n_total
            t_r, v_r = final_train_rmse, final_val_rmse
        else:
            n_sub = int(round(n_total * frac))
            sub_idx = shuffled_idx[:n_sub]
            imp = SimpleImputer(strategy='mean')
            X_sub = imp.fit_transform(_take_rows(X_train_raw, sub_idx))
            y_sub = _take_rows(y_train, sub_idx)
            X_val_cur = imp.transform(X_val_raw)

            m = RandomForestRegressor(**best_params)
            m.fit(X_sub, y_sub)
            t_r = rmse_score(y_sub, m.predict(X_sub))
            v_r = rmse_score(y_val, m.predict(X_val_cur))

        sizes.append(n_sub)
        tr_list.append(t_r)
        va_list.append(v_r)
        print(f"  Size={n_sub:<7} ({int(frac * 100):>3}%) | Train={t_r:.4f} | Val={v_r:.4f}")

    plt.figure(figsize=(8.5, 5))
    plt.plot(sizes, tr_list, marker='o', color='#1f77b4', label='Train RMSE')
    plt.plot(sizes, va_list, marker='s', color='#ff7f0e', label='Validation RMSE')
    plt.title('Random Forest: Learning Curve')
    plt.xlabel('Training Set Size')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(PLOTS_DIR, 'rf_learning_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def tune_rf():
    df = load_ratings_scaled()
    splits, meta = utils.get_splits(df)

    X_train_raw, y_train = splits['train']
    X_val_raw, y_val = splits['val']
    X_test_raw, y_test = splits['test']

    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train_raw)
    X_val = imputer.transform(X_val_raw)
    X_test = imputer.transform(X_test_raw)

    run_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    history_rows = []

    print(f"\n=== Bayesian Optimization (Optuna TPE): Random Forest (RF) | {N_TRIALS} trials ===")

    def objective(trial):
        params = {
            'n_estimators': SEARCH_N_ESTIMATORS,
            'criterion': trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse', 'poisson']),
            'max_depth': trial.suggest_int('max_depth', 12, 30, step=2),
            'max_features': trial.suggest_float('max_features', 0.2, 1.0, step=0.1),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'n_jobs': -1,
            'random_state': 42,
        }
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        t_rmse = rmse_score(y_train, model.predict(X_train))
        v_rmse = rmse_score(y_val, model.predict(X_val))

        trial_num = trial.number + 1
        print(f"  [Trial {trial_num:02d}/{N_TRIALS}] crit={params['criterion']:<13} depth={params['max_depth']:<2} "
              f"feat={params['max_features']:.1f} split={params['min_samples_split']:<2} "
              f"leaf={params['min_samples_leaf']:<2} | Train={t_rmse:.4f} | Val={v_rmse:.4f}")

        history_rows.append({
            'Timestamp': run_time,
            'Search_Method': 'Bayesian_Optuna_TPE',
            'Trial': trial_num,
            'Train RMSE': round(t_rmse, 4),
            'Val RMSE': round(v_rmse, 4),
            'Params Snapshot': json.dumps(params, default=str),
        })
        return v_rmse

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)

    # Plot Phase 1 Optuna charts using only the Optuna trial rows
    plot_optuna_results(pd.DataFrame(history_rows), study, 'rf', 'Random Forest')

    best_params = {
        'n_estimators': SEARCH_N_ESTIMATORS,
        **study.best_params,
        'n_jobs': -1,
        'random_state': 42,
    }

    # Phase 2: Scale n_estimators on the winning tree structure
    print("\n--- Phase 2: Scaling n_estimators on Best RF Structure ---")
    n_candidates = [100, 200, 300]
    tr_scale, va_scale = [], []
    for n_est in n_candidates:
        test_p = {**best_params, 'n_estimators': n_est}
        m = RandomForestRegressor(**test_p)
        m.fit(X_train, y_train)
        t_r = rmse_score(y_train, m.predict(X_train))
        v_r = rmse_score(y_val, m.predict(X_val))
        tr_scale.append(t_r)
        va_scale.append(v_r)
        print(f"  n_estimators={n_est:<3} | Train={t_r:.4f} | Val={v_r:.4f}")

        history_rows.append({
            'Timestamp': run_time,
            'Search_Method': 'Phase2_Tree_Scaling',
            'Trial': f'n_est={n_est}',
            'Train RMSE': round(t_r, 4),
            'Val RMSE': round(v_r, 4),
            'Params Snapshot': json.dumps(test_p, default=str),
        })

    # Save BOTH Phase 1 and Phase 2 together
    save_history(history_rows)

    best_n = n_candidates[int(np.argmin(va_scale))]
    best_params['n_estimators'] = best_n
    plot_n_estimators(n_candidates, tr_scale, va_scale, best_n)

    print(f"\nSaved tuning history to {HISTORY_FILE}")
    print(f"Best RF Params (Val RMSE={min(va_scale):.4f}): {best_params}")

    final_model = RandomForestRegressor(**best_params)
    tracemalloc.start()
    t0 = time.time()
    final_model.fit(X_train, y_train)
    train_time = time.time() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    train_mem_mb = peak_mem / 1e6

    final_train_rmse = rmse_score(y_train, final_model.predict(X_train))
    final_val_rmse = rmse_score(y_val, final_model.predict(X_val))
    plot_learning_curve(best_params, X_train_raw, y_train, X_val_raw, y_val,
                        final_train_rmse, final_val_rmse)

    rows = []
    for split_name, (X, y) in [('train', (X_train, y_train)), ('val', (X_val, y_val)), ('test', (X_test, y_test))]:
        res = utils.evaluate(split_name, final_model, X, y, meta[split_name])
        res['model'] = "RF_scaled"
        res['train_time_s'] = train_time
        res['train_mem_mb'] = train_mem_mb
        rows.append(res)

    utils.save_results(rows, best_params)
    print(f"Saved tuned RF results to {utils.RESULTS_FILE}")


if __name__ == "__main__":
    tune_rf()