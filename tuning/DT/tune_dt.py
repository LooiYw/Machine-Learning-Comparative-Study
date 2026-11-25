import datetime
import itertools
import json
import os
import time
import tracemalloc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

import utils
from preprocessing import load_ratings_scaled

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plot')
HISTORY_FILE = os.path.join(BASE_DIR, 'dt_tune_history.csv')
os.makedirs(PLOTS_DIR, exist_ok=True)

utils.RESULTS_FILE = os.path.join('results', 'tuned_model_evaluation_results.csv')


def rmse_score(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, np.clip(y_pred, 1, 5))))


def save_history(rows):
    df_hist = pd.DataFrame(rows)
    df_hist['Train RMSE'] = df_hist['Train RMSE'].round(4)
    df_hist['Val RMSE'] = df_hist['Val RMSE'].round(4)
    write_header = not os.path.exists(HISTORY_FILE)
    df_hist.to_csv(HISTORY_FILE, mode='a', header=write_header, index=False)


def plot_dt_results(df_run, best_params):
    df_step1 = df_run[df_run['Step'] == 1]
    df_step2 = df_run[df_run['Step'] == 2]

    # 1. Heatmap: max_depth vs min_samples_leaf (at best min_samples_split)
    sub_hm = df_step1[df_step1['min_samples_split'] == best_params['min_samples_split']]
    pivot = sub_hm.pivot(index='max_depth', columns='min_samples_leaf', values='Val RMSE')

    plt.figure(figsize=(9, 5.5))
    im = plt.imshow(pivot.values, cmap='viridis_r', aspect='auto')
    plt.colorbar(im, label='Validation RMSE')
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel('min_samples_leaf')
    plt.ylabel('max_depth')
    plt.title(f"Decision Tree Grid Search Heatmap (min_samples_split={best_params['min_samples_split']})")

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            plt.text(j, i, f"{val:.4f}", ha='center', va='center',
                     color='white' if val > pivot.values.mean() else 'black', fontsize=8.5)

    plt.savefig(os.path.join(PLOTS_DIR, 'dt_grid_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Top-10 Structural Configurations Bar Chart (Train vs Val RMSE)
    top10 = df_step1.sort_values('Val RMSE', kind='mergesort').head(10).reset_index(drop=True)
    labels = [
        f"d={r['max_depth']}, s={r['min_samples_split']}, l={r['min_samples_leaf']}"
        for _, r in top10.iterrows()
    ]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10.5, 5))
    plt.bar(x - width / 2, top10['Train RMSE'], width, label='Train RMSE', color='#1f77b4')
    bars_va = plt.bar(x + width / 2, top10['Val RMSE'], width, label='Validation RMSE', color='#ff7f0e')
    bars_va[0].set_edgecolor('red')
    bars_va[0].set_linewidth(2.0)

    for bar in bars_va:
        h = bar.get_height()
        plt.annotate(f"{h:.4f}", (bar.get_x() + bar.get_width() / 2, h),
                     textcoords="offset points", xytext=(0, 4), ha='center', fontsize=7.5)

    all_v = top10['Train RMSE'].tolist() + top10['Val RMSE'].tolist()
    margin = max((max(all_v) - min(all_v)) * 0.25, 0.005)
    plt.ylim(max(0, min(all_v) - margin), max(all_v) + margin * 1.4)
    plt.xticks(x, labels, rotation=30, ha='right', fontsize=8.5)
    plt.title('Decision Tree Grid Search: Top 10 Structural Configurations')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(PLOTS_DIR, 'dt_grid_top10.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Per-Parameter Slice Line Plots (evenly spaced x-axis showing exact tested values)
    struct_params = ['max_depth', 'min_samples_split', 'min_samples_leaf']
    for param_name in struct_params:
        mask = pd.Series(True, index=df_step1.index)
        for other_p in struct_params:
            if other_p != param_name:
                mask &= (df_step1[other_p] == best_params[other_p])
        sub_p = df_step1[mask].sort_values(param_name)

        values = sub_p[param_name].tolist()
        tr_vals = sub_p['Train RMSE'].tolist()
        va_vals = sub_p['Val RMSE'].tolist()
        best_val = best_params[param_name]

        x_pos = np.arange(len(values))
        best_idx = values.index(best_val)

        plt.figure(figsize=(8.5, 5))
        plt.plot(x_pos, tr_vals, marker='o', label='Train RMSE')
        plt.plot(x_pos, va_vals, marker='s', label='Validation RMSE')
        plt.axvline(best_idx, color='r', linestyle='--', label=f'Best={best_val}')

        for xi, y_val in zip(x_pos, va_vals):
            plt.annotate(f"{y_val:.4f}", (xi, y_val), textcoords="offset points",
                         xytext=(0, 8), ha='center', fontsize=8.5, color='darkorange')

        plt.xticks(x_pos, [str(v) for v in values])
        plt.margins(y=0.15)
        if param_name in ('min_samples_split', 'min_samples_leaf'):
            plt.xlabel(f"{param_name}  (← Higher Complexity | Stronger Regularization →)")
        else:
            plt.xlabel(param_name)
        plt.ylabel('RMSE')
        plt.title(f'Decision Tree Grid Search Slice: {param_name} (Best={best_val})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(PLOTS_DIR, f'dt_grid_param_{param_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Grouped Bar Chart for criterion (Step 2 at winning tree structure)
    c_labels = df_step2['criterion'].tolist()
    c_tr = df_step2['Train RMSE'].tolist()
    c_va = df_step2['Val RMSE'].tolist()

    xc = np.arange(len(c_labels))
    width = 0.35
    plt.figure(figsize=(8.5, 5))
    bars_tr = plt.bar(xc - width / 2, c_tr, width, label='Train RMSE', color='#1f77b4')
    bars_va = plt.bar(xc + width / 2, c_va, width, label='Validation RMSE', color='#ff7f0e')

    best_c_idx = c_labels.index(best_params['criterion'])
    bars_va[best_c_idx].set_edgecolor('red')
    bars_va[best_c_idx].set_linewidth(2.0)

    for bar in list(bars_tr) + list(bars_va):
        h = bar.get_height()
        plt.annotate(f"{h:.4f}", (bar.get_x() + bar.get_width() / 2, h),
                     textcoords="offset points", xytext=(0, 4), ha='center', fontsize=8.5)

    all_c = c_tr + c_va
    c_margin = max((max(all_c) - min(all_c)) * 0.3, 0.002)
    plt.ylim(max(0, min(all_c) - c_margin), max(all_c) + c_margin * 1.4)
    plt.xticks(xc, c_labels)
    plt.title(f"Decision Tree Step 2: criterion (Best={best_params['criterion']})")
    plt.xlabel('criterion')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(PLOTS_DIR, 'dt_grid_param_criterion.png'), dpi=300, bbox_inches='tight')
    plt.close()


def tune_dt():
    df = load_ratings_scaled()
    splits, meta = utils.get_splits(df)

    imputer = SimpleImputer(strategy='mean')
    X_train, y_train = splits['train']
    X_train = imputer.fit_transform(X_train)
    X_val, y_val = imputer.transform(splits['val'][0]), splits['val'][1]
    X_test, y_test = imputer.transform(splits['test'][0]), splits['test'][1]

    # Step 1: Joint grid over interacting structural parameters (4 x 7 x 6 = 168 combos)
    param_grid = {
        'max_depth': [8, 10, 12, 15],
        'min_samples_split': [2, 50, 100, 150, 200, 300, 500],
        'min_samples_leaf': [1, 10, 50, 100, 150, 200],
    }
    criteria = ['squared_error', 'friedman_mse', 'poisson']

    combos = list(itertools.product(
        param_grid['max_depth'],
        param_grid['min_samples_split'],
        param_grid['min_samples_leaf'],
    ))

    run_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    history_rows = []
    best_val_rmse = float('inf')
    best_struct = None
    trial_num = 0

    print(f"\n=== Step 1: Joint Grid Search (Structure) | {len(combos)} combinations ===")
    for max_depth, min_samples_split, min_samples_leaf in combos:
        trial_num += 1
        params = {
            'criterion': 'squared_error',
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'random_state': 42,
        }
        model = DecisionTreeRegressor(**params)
        model.fit(X_train, y_train)
        t_rmse = rmse_score(y_train, model.predict(X_train))
        v_rmse = rmse_score(y_val, model.predict(X_val))

        if v_rmse < best_val_rmse:
            best_val_rmse = v_rmse
            best_struct = params.copy()

        print(f"  [{trial_num:03d}/{len(combos)}] depth={max_depth:<2} split={min_samples_split:<3} "
              f"leaf={min_samples_leaf:<3} | Train={t_rmse:.4f} | Val={v_rmse:.4f}")

        history_rows.append({
            'Timestamp': run_time,
            'Search_Method': 'TwoStep_Grid_Search',
            'Step': 1,
            'Trial': trial_num,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'criterion': 'squared_error',
            'Train RMSE': t_rmse,
            'Val RMSE': v_rmse,
            'Params Snapshot': json.dumps(params, default=str),
        })

    # Step 2: Compare all split criteria on the winning tree structure (3 fits)
    print(f"\n=== Step 2: Criterion Comparison on Best Structure "
          f"(depth={best_struct['max_depth']}, split={best_struct['min_samples_split']}, "
          f"leaf={best_struct['min_samples_leaf']}) ===")
    best_crit_val = float('inf')
    best_params = None

    for criterion in criteria:
        trial_num += 1
        params = {**best_struct, 'criterion': criterion}
        model = DecisionTreeRegressor(**params)
        model.fit(X_train, y_train)
        t_rmse = rmse_score(y_train, model.predict(X_train))
        v_rmse = rmse_score(y_val, model.predict(X_val))

        if v_rmse < best_crit_val:
            best_crit_val = v_rmse
            best_params = params.copy()

        print(f"  criterion={criterion:<14} | Train={t_rmse:.4f} | Val={v_rmse:.4f}")

        history_rows.append({
            'Timestamp': run_time,
            'Search_Method': 'TwoStep_Grid_Search',
            'Step': 2,
            'Trial': trial_num,
            'max_depth': params['max_depth'],
            'min_samples_split': params['min_samples_split'],
            'min_samples_leaf': params['min_samples_leaf'],
            'criterion': criterion,
            'Train RMSE': t_rmse,
            'Val RMSE': v_rmse,
            'Params Snapshot': json.dumps(params, default=str),
        })

    save_history(history_rows)
    plot_dt_results(pd.DataFrame(history_rows), best_params)
    print(f"\nSaved tuning history to {HISTORY_FILE}")
    print(f"Best DT Params (Val RMSE={best_crit_val:.4f}): {best_params}")

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