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
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

import utils
from preprocessing import load_ratings_scaled

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plot')
HISTORY_FILE = os.path.join(BASE_DIR, 'xgb_tune_history.csv')
os.makedirs(PLOTS_DIR, exist_ok=True)

utils.RESULTS_FILE = os.path.join('results', 'tuned_model_evaluation_results.csv')

N_TRIALS = 50
MAX_ESTIMATORS = 2000
EARLY_STOPPING_ROUNDS = 50


def rmse_score(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, np.clip(y_pred, 1, 5))))


def _take_rows(data, idx):
    return data.iloc[idx] if hasattr(data, 'iloc') else data[idx]


def save_history(rows):
    df_hist = pd.DataFrame(rows)
    write_header = not os.path.exists(HISTORY_FILE)
    df_hist.to_csv(HISTORY_FILE, mode='a', header=write_header, index=False)


def plot_optuna_results(df_run, study):
    # 1. Optimization History
    trials = df_run['Trial'].values
    val_rmses = df_run['Val RMSE'].values
    best_so_far = np.minimum.accumulate(val_rmses)

    plt.figure(figsize=(8.5, 5))
    plt.scatter(trials, val_rmses, color='#ff7f0e', label='Trial Validation RMSE', zorder=3)
    plt.plot(trials, best_so_far, color='red', linewidth=2, label=f'Best So Far ({best_so_far[-1]:.4f})', zorder=4)
    plt.title('XGBoost Bayesian Optimization History (Optuna TPE + Early Stopping)')
    plt.xlabel('Trial Number')
    plt.ylabel('Validation RMSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(PLOTS_DIR, 'xgb_optuna_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Hyperparameter Importance
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
        plt.title('XGBoost Hyperparameter Importance (fANOVA)')
        plt.xlabel('Relative Importance')
        plt.grid(True, axis='x', linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(PLOTS_DIR, 'xgb_optuna_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  [Info] Skipped importance plot: {e}")


def plot_learning_curve(final_params, X_train, y_train, X_val, y_val,
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
            X_sub = _take_rows(X_train, sub_idx)
            y_sub = _take_rows(y_train, sub_idx)

            m = XGBRegressor(**final_params)
            m.fit(X_sub, y_sub)
            t_r = rmse_score(y_sub, m.predict(X_sub))
            v_r = rmse_score(y_val, m.predict(X_val))

        sizes.append(n_sub)
        tr_list.append(t_r)
        va_list.append(v_r)
        print(f"  Size={n_sub:<7} ({int(frac * 100):>3}%) | Train={t_r:.4f} | Val={v_r:.4f}")

    plt.figure(figsize=(8.5, 5))
    plt.plot(sizes, tr_list, marker='o', color='#1f77b4', label='Train RMSE')
    plt.plot(sizes, va_list, marker='s', color='#ff7f0e', label='Validation RMSE')
    plt.title('XGBoost: Learning Curve')
    plt.xlabel('Training Set Size')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(PLOTS_DIR, 'xgb_learning_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()


def tune_xgb():
    df = load_ratings_scaled()
    splits, meta = utils.get_splits(df)

    # XGBoost handles cold-start NaNs natively
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']

    run_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    history_rows = []

    print(f"\n=== Bayesian Optimization (Optuna TPE + Early Stopping): XGBoost | {N_TRIALS} trials ===")

    def objective(trial):
        params = {
            'n_estimators': MAX_ESTIMATORS,
            'max_depth': trial.suggest_int('max_depth', 4, 14),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.05),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0, step=0.05),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'tree_method': 'hist',
            'eval_metric': 'rmse',
            'early_stopping_rounds': EARLY_STOPPING_ROUNDS,
            'n_jobs': -1,
            'random_state': 42,
        }

        model = XGBRegressor(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        best_iter = getattr(model, 'best_iteration', None)
        best_n_est = (int(best_iter) + 1) if best_iter is not None else MAX_ESTIMATORS
        trial.set_user_attr('best_n_estimators', best_n_est)

        t_rmse = rmse_score(y_train, model.predict(X_train))
        v_rmse = rmse_score(y_val, model.predict(X_val))

        trial_num = trial.number + 1
        print(f"  [Trial {trial_num:02d}/{N_TRIALS}] depth={params['max_depth']:<2} "
              f"lr={params['learning_rate']:.3f} trees={best_n_est:<4} "
              f"sub={params['subsample']:.2f} col={params['colsample_bytree']:.2f} "
              f"| Train={t_rmse:.4f} | Val={v_rmse:.4f}")

        saved_snap = {**params, 'n_estimators': best_n_est}
        saved_snap.pop('early_stopping_rounds', None)
        history_rows.append({
            'Timestamp': run_time,
            'Search_Method': 'Bayesian_Optuna_TPE_EarlyStop',
            'Trial': trial_num,
            'best_n_estimators': best_n_est,
            'Train RMSE': round(t_rmse, 4),
            'Val RMSE': round(v_rmse, 4),
            'Params Snapshot': json.dumps(saved_snap, default=str),
        })
        return v_rmse

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS)

    save_history(history_rows)
    plot_optuna_results(pd.DataFrame(history_rows), study)

    best_n_est = study.best_trial.user_attrs['best_n_estimators']
    final_params = {
        **study.best_params,
        'n_estimators': best_n_est,
        'tree_method': 'hist',
        'eval_metric': 'rmse',
        'n_jobs': -1,
        'random_state': 42,
    }
    print(f"\nSaved tuning history to {HISTORY_FILE}")
    print(f"Best XGB Params (Val RMSE={study.best_value:.4f}): {final_params}")

    # Clean timed final fit (without eval_set overhead for accurate train_time_s)
    final_model = XGBRegressor(**final_params)
    tracemalloc.start()
    t0 = time.time()
    final_model.fit(X_train, y_train)
    train_time = time.time() - t0
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    train_mem_mb = peak_mem / 1e6

    # 1. Boosting Rounds Training Curve (RMSE vs. n_estimators)
    curve_model = XGBRegressor(**final_params)
    curve_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    evals_result = curve_model.evals_result()
    rounds = range(1, len(evals_result['validation_0']['rmse']) + 1)
    plt.figure(figsize=(8.5, 5))
    plt.plot(rounds, evals_result['validation_0']['rmse'], label='Train RMSE')
    plt.plot(rounds, evals_result['validation_1']['rmse'], label='Validation RMSE')
    plt.title(f'XGBoost Training Curve (Optimal Trees={best_n_est})')
    plt.xlabel('Boosting Rounds (n_estimators)')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(PLOTS_DIR, 'xgb_boosting_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Learning Curve (RMSE vs. Training Set Size) to match other models
    final_train_rmse = rmse_score(y_train, final_model.predict(X_train))
    final_val_rmse = rmse_score(y_val, final_model.predict(X_val))
    plot_learning_curve(final_params, X_train, y_train, X_val, y_val,
                        final_train_rmse, final_val_rmse)

    rows = []
    for split_name, (X, y) in [('train', (X_train, y_train)), ('val', (X_val, y_val))]:
        res = utils.evaluate(split_name, final_model, X, y, meta[split_name])
        res['model'] = "XGB_scaled"
        res['train_time_s'] = train_time
        res['train_mem_mb'] = train_mem_mb
        rows.append(res)

    utils.save_results(rows, final_params)
    print(f"Saved tuned XGB results to {utils.RESULTS_FILE}")


if __name__ == "__main__":
    tune_xgb()