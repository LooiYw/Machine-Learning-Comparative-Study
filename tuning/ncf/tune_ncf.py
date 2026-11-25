"""
Sequential hyperparameter tuning for the NeuMF (NCF) model.

REVISED TUNING ORDER (greedy coordinate descent; each stage keeps the best
configuration found, by lowest validation RMSE, and carries it forward):

    1. Optimizer + Learning rate : Adam, AdamW, SGD+Momentum, RMSprop, Adagrad.
                                    LR is tuned INDEPENDENTLY *within* each
                                    optimizer (a full LR sweep per optimizer)
                                    before the optimizers are compared against
                                    each other on their own best LR.
    2. Activation function       : ReLU, LeakyReLU, GELU, ELU, SiLU/Swish.
                                    Optimizer + LR from stage 1 are held fixed.
    3. MLP structure + LR        : mlp_structure -> num_layers -> num_neurons
                                    (unchanged mechanics, carried over as-is
                                    from the original script) followed by a
                                    learning-rate re-sweep, since a deeper or
                                    wider network can destabilise optimisation
                                    at the previously-tuned LR.
    4. Batch size + LR           : for each batch size, LR is independently
                                    re-swept (no assumed linear-scaling rule
                                    between batch size and LR) and the best
                                    (batch_size, lr) pair is kept.
    5. Dropout rate               : plain sweep, including 0.0 (no dropout).

embed_dim and weight_decay are NOT tuned here -- they're kept fixed at their
imported defaults, since they weren't part of the requested 5-stage scheme.
(The original 7-stage script tuned both; drop this block if you want them
back in the rotation.)

All winning hyperparameters accumulate in `best_config = {}`. Tuning only
evaluates on the `train` and `val` splits (never touching `test`). Each trial
logs to `tuning/ncf/ncf_tuning_history.csv` (same file/schema-merging logic as
before) and each stage saves a line/bar chart into `tuning/ncf/`.

Requires: everything models/ncf.py requires, plus matplotlib.

==============================================================================
IMPORTANT -- ASSUMPTIONS, BECAUSE models/ncf.py / preprocessing.py / utils.py
WERE NOT AVAILABLE WHEN THIS SCRIPT WAS WRITTEN
==============================================================================
This script needs three hooks in your existing code that the *original*
7-stage script did not need (it never varied optimizer, activation, or batch
size). Each hook is checked via introspection in `_check_hooks()` BEFORE any
training starts, and the script raises a RuntimeError with the exact fix if a
hook is missing -- it will NOT silently run a half-wired experiment (e.g.
quietly training every "optimizer" candidate with the same Adam optimizer).

A) NCF.__init__ must accept an `activation` argument (a string, one of
   'relu' / 'leaky_relu' / 'gelu' / 'elu' / 'silu') and use it for the MLP
   tower's nonlinearities instead of a hardcoded nn.ReLU(). Minimal patch
   inside models/ncf.py:

       ACTIVATIONS = {
           'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU,
           'gelu': nn.GELU, 'elu': nn.ELU, 'silu': nn.SiLU,
       }

       class NCF(nn.Module):
           def __init__(self, ..., activation='relu', ...):
               act_cls = ACTIVATIONS[activation]
               ...  # use act_cls() wherever the MLP tower currently does nn.ReLU()

B) train_model must accept `optimizer_name` (one of 'adam', 'adamw',
   'sgd_momentum', 'rmsprop', 'adagrad') and `momentum` (used only when
   optimizer_name == 'sgd_momentum'), and build the optimizer accordingly
   instead of always using Adam. Minimal patch:

       def build_optimizer(name, params, lr, weight_decay, momentum=0.9):
           if name == 'adam':         return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
           if name == 'adamw':        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
           if name == 'sgd_momentum': return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
           if name == 'rmsprop':      return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
           if name == 'adagrad':      return torch.optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
           raise ValueError(f"Unknown optimizer: {name}")

C) train_model must accept a `batch_size` override (falling back to the
   module-level BATCH_SIZE default when omitted) and use it to build the
   DataLoader(s) instead of always reading the global constant.

If train_model already accepts **kwargs and quietly forwards unknown ones to
the optimizer/DataLoader construction, the introspection below detects that
too (via VAR_KEYWORD) and proceeds without complaint.

==============================================================================
BUGFIX carried over from the uploaded script
==============================================================================
The uploaded script referenced `HISTORY_CSV_PATH` as a default argument
value without ever defining it (the line that built it was commented out:
`# HISTORY_CSV_PATH = os.path.join(OUTPUT_DIR, 'ncf_tuning_history.csv')`).
That raises a NameError the moment the module is imported. It's defined
properly below.
"""
import argparse
import csv
import gc
import inspect
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(OUTPUT_DIR))
# BUGFIX: this was referenced below but never defined in the uploaded script.
HISTORY_CSV_PATH = os.path.join(OUTPUT_DIR, 'ncf_tuning_history.csv')

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.ncf import (
    NCF,
    NCFPredictor,
    EMBED_DIM,
    DROPOUT,
    BATCH_SIZE,
    prepare_data,
    train_model,
)
from preprocessing import load_ratings_preprocessed, load_ratings_scaled
from utils import evaluate_nn

# ---------------------------------------------------------------------------
# Candidate grids
# ---------------------------------------------------------------------------
OPTIMIZER_CANDIDATES = ('adam', 'adamw', 'sgd_momentum', 'rmsprop', 'adagrad')
ACTIVATION_CANDIDATES = ('relu', 'leaky_relu', 'gelu', 'elu', 'silu')
MLP_STRUCTURES = ('constant', 'increasing', 'decreasing')
NUM_LAYERS_CANDIDATES = (1, 2, 3, 4, 5)
NUM_NEURONS_CANDIDATES = (64, 128, 192, 256, 384, 512)
LR_CANDIDATES = (1e-4, 5e-4, 1e-3, 5e-3, 1e-2)
BATCH_SIZE_CANDIDATES = (32, 64, 128, 256, 512)
DROPOUT_CANDIDATES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

DATASET_LOADERS = {
    'preprocessed': load_ratings_preprocessed,
    'scaled': load_ratings_scaled,
}

DEFAULT_MOMENTUM = 0.9  # only used when optimizer == 'sgd_momentum'; not tuned


# ---------------------------------------------------------------------------
# Hook detection -- fail loudly, before training, if the model/train_model
# API doesn't support what this tuning order needs.
# ---------------------------------------------------------------------------
_MISSING_ACTIVATION_HOOK_MSG = """
NCF.__init__ does not appear to accept an `activation` argument (and does not
accept **kwargs either), so the activation-function stage cannot actually
change anything about the model. Add an `activation` parameter to NCF and use
it to build the MLP tower's nonlinearities. See the module docstring (point A)
at the top of this file for the minimal patch.
""".strip()

_MISSING_OPTIMIZER_HOOK_MSG = """
train_model does not appear to accept an `optimizer_name` (or `optimizer`)
argument (and does not accept **kwargs either), so the optimizer stage cannot
actually switch optimizers -- every "optimizer" candidate would silently
train with whatever train_model currently hardcodes. Add an `optimizer_name`
(+ `momentum`) argument to train_model and build the optimizer accordingly.
See the module docstring (point B) at the top of this file for the minimal
patch.
""".strip()

_MISSING_BATCH_SIZE_HOOK_MSG = """
train_model does not appear to accept a `batch_size` argument (and does not
accept **kwargs either), so the batch-size stage cannot actually change the
batch size used during training -- every "batch_size" candidate would
silently train with the fixed global BATCH_SIZE. Add a `batch_size` argument
to train_model (defaulting to the global BATCH_SIZE when omitted) and use it
to build the DataLoader(s). See the module docstring (point C) at the top of
this file for the minimal patch.
""".strip()


def _supports(params, has_kwargs, *names):
    """True if the signature accepts **kwargs or any of the given names."""
    return has_kwargs or any(n in params for n in names)


def _signature_info(fn):
    sig = inspect.signature(fn)
    params = sig.parameters
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    return params, has_kwargs


def _check_hooks():
    """Verify NCF/train_model support activation, optimizer, and batch_size
    overrides before any training happens. Raises RuntimeError with the exact
    fix otherwise (see module docstring, points A/B/C)."""
    ncf_params, ncf_has_kwargs = _signature_info(NCF.__init__)
    if not _supports(ncf_params, ncf_has_kwargs, 'activation'):
        raise RuntimeError(_MISSING_ACTIVATION_HOOK_MSG)

    tm_params, tm_has_kwargs = _signature_info(train_model)
    if not _supports(tm_params, tm_has_kwargs, 'optimizer_name', 'optimizer'):
        raise RuntimeError(_MISSING_OPTIMIZER_HOOK_MSG)
    if not _supports(tm_params, tm_has_kwargs, 'batch_size'):
        raise RuntimeError(_MISSING_BATCH_SIZE_HOOK_MSG)

    return ncf_params, ncf_has_kwargs, tm_params, tm_has_kwargs


def _get_train_defaults():
    """Read default lr and weight_decay from train_model's signature if available."""
    sig = inspect.signature(train_model)
    params = sig.parameters
    default_lr = 1e-3
    for name in ('lr', 'learning_rate'):
        if name in params and params[name].default is not inspect.Parameter.empty:
            default_lr = params[name].default
            break
    default_wd = 0.0
    if 'weight_decay' in params and params['weight_decay'].default is not inspect.Parameter.empty:
        default_wd = params['weight_decay'].default
    return default_lr, default_wd


def build_mlp_layers(structure, num_layers, total_neurons):
    """
    Generate an `mlp_layers` tuple of length `num_layers` summing to `total_neurons`
    according to `structure` ('constant', 'increasing', or 'decreasing').
    Unchanged from the original script.
    """
    if num_layers == 1:
        return (total_neurons,)

    if structure == 'constant':
        weights = [1] * num_layers
    elif structure == 'increasing':
        weights = list(range(1, num_layers + 1))
    elif structure == 'decreasing':
        weights = list(range(num_layers, 0, -1))
    else:
        raise ValueError(f"Unknown MLP structure: {structure}")

    weight_sum = sum(weights)
    widths = [max(1, round(total_neurons * w / weight_sum)) for w in weights]

    diff = total_neurons - sum(widths)
    if diff != 0:
        idx = widths.index(max(widths))
        widths[idx] = max(1, widths[idx] + diff)

    return tuple(widths)


def rmse_of(predictor, data):
    """RMSE of a trained model's predictions on one split's tensors."""
    n = len(data['rating'])
    preds = torch.from_numpy(predictor.predict(range(n))).to(data['rating'].device)
    return float(torch.sqrt(((preds - data['rating']) ** 2).mean()))


def evaluate_train_val_splits(label, model_factory, splits, meta, train_stats):
    """Run utils.evaluate_nn on train and val splits only, attaching training cost columns."""
    rows = []
    for split_name in ('train', 'val'):
        n = len(splits[split_name]['rating'])
        X = np.arange(n)
        y = splits[split_name]['rating'].cpu().numpy()
        device = splits[split_name]['rating'].device
        res = evaluate_nn(
            split_name, model_factory(split_name), X, y,
            meta[split_name], device=device,
        )
        res['model'] = label
        res.update(train_stats)
        rows.append(res)
    return rows


def save_ncf_tuning_results(rows, extra_meta, csv_path=HISTORY_CSV_PATH):
    """Append NCF evaluation rows and tuning metadata directly to `ncf_tuning_history.csv`."""
    if hasattr(rows, 'to_dict'):
        row_list = rows.to_dict(orient='records')
    else:
        row_list = [dict(r) for r in rows]

    if not row_list:
        return

    merged_rows = [{**row, **extra_meta} for row in row_list]

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0

    if file_exists:
        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = list(reader)
    else:
        existing_fields = []
        existing_rows = []

    fieldnames = list(existing_fields)
    for row in merged_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    if file_exists and fieldnames == existing_fields:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(merged_rows)
    else:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows + merged_rows)


def run_trial(stage_name, label, config, splits, meta, info, device, dataset_label,
              ncf_params, ncf_has_kwargs, tm_params, tm_has_kwargs):
    """Train and evaluate a single NCF configuration on train/val splits only."""
    print(f"\n{'=' * 60}")
    print(f"[{stage_name}] {label} | config={config}")
    print('=' * 60)

    def build_model():
        kwargs = dict(
            n_features=info['n_features'],
            embed_dim=config['embed_dim'],
            mlp_layers=config['mlp_layers'],
            dropout=config['dropout'],
            global_mean=info['global_mean'],
        )
        if _supports(ncf_params, ncf_has_kwargs, 'activation'):
            kwargs['activation'] = config['activation']
        return NCF(info['n_users'], info['n_movies'], **kwargs)

    train_kwargs = {'use_features': True}

    if 'lr' in tm_params or tm_has_kwargs:
        train_kwargs['lr'] = config['lr']
    elif 'learning_rate' in tm_params:
        train_kwargs['learning_rate'] = config['lr']

    if _supports(tm_params, tm_has_kwargs, 'weight_decay'):
        train_kwargs['weight_decay'] = config['weight_decay']

    if _supports(tm_params, tm_has_kwargs, 'optimizer_name', 'optimizer'):
        opt_key = 'optimizer_name' if ('optimizer_name' in tm_params or tm_has_kwargs) else 'optimizer'
        train_kwargs[opt_key] = config['optimizer']
        if config['optimizer'] == 'sgd_momentum' and _supports(tm_params, tm_has_kwargs, 'momentum'):
            train_kwargs['momentum'] = config.get('momentum', DEFAULT_MOMENTUM)

    if _supports(tm_params, tm_has_kwargs, 'batch_size'):
        train_kwargs['batch_size'] = config['batch_size']

    train_val_splits = {'train': splits['train'], 'val': splits['val']}
    model, stats, train_params = train_model(build_model, train_val_splits, info, device, **train_kwargs)

    rmse = {}
    for split_name in ('train', 'val'):
        predictor = NCFPredictor(model, train_val_splits[split_name], use_features=True)
        rmse[split_name] = rmse_of(predictor, train_val_splits[split_name])
    print(f"RMSE -> train: {rmse['train']:.4f} | val: {rmse['val']:.4f}")

    rows = evaluate_train_val_splits(
        f"{dataset_label}_{stage_name.upper()}_{label}",
        lambda s: NCFPredictor(model, train_val_splits[s], use_features=True),
        train_val_splits, meta, stats,
    )
    extra_meta = {
        'tuning_stage': stage_name,
        'embed_dim': config['embed_dim'],
        'mlp_layers': config['mlp_layers'],
        'num_layers': config['num_layers'],
        'num_neurons': config['num_neurons'],
        'mlp_structure': config['mlp_structure'],
        'activation': config['activation'],
        'optimizer': config['optimizer'],
        'momentum': config.get('momentum', DEFAULT_MOMENTUM) if config['optimizer'] == 'sgd_momentum' else '',
        'batch_size': config['batch_size'],
        'dropout': config['dropout'],
        'lr': config['lr'],
        'weight_decay': config['weight_decay'],
        **train_params,
    }
    save_ncf_tuning_results(rows, extra_meta)

    del model
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return rmse


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_stage_results(results, x_labels, best_key, title, out_path):
    """Line chart of train and val RMSE for a single-parameter tuning stage,
    marking the winner. Unchanged from the original script."""
    keys = list(results.keys())
    splits = ('train', 'val')
    styles = {
        'train': {'color': '#4C72B0', 'marker': 'o', 'offset': -8},
        'val':   {'color': '#DD8452', 'marker': 's', 'offset': 8},
    }

    x = list(range(len(keys)))

    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(keys)), 5))
    for split_name in splits:
        values = [results[k][split_name] for k in keys]
        st = styles[split_name]
        ax.plot(
            x, values,
            label=split_name,
            color=st['color'],
            marker=st['marker'],
            linewidth=2,
            markersize=7,
        )
        for xi, val in zip(x, values):
            ax.annotate(
                f"{val:.4f}",
                (xi, val),
                textcoords="offset points",
                xytext=(0, st['offset']),
                ha='center',
                fontsize=8,
                color=st['color'],
            )

    best_idx = keys.index(best_key)
    ax.axvline(best_idx, color='#55A868', linestyle='--', alpha=0.6, linewidth=1.2, label='best (val)')

    formatted_labels = [
        f"{lbl}\n(best)" if k == best_key else lbl
        for k, lbl in zip(keys, x_labels)
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(formatted_labels)
    ax.set_ylabel('RMSE')
    ax.set_title(title)
    ax.legend(title='split')
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_path}")


def plot_candidate_summary(candidates, values, best_cand, title, out_path, xlabel):
    """Bar chart of each candidate's best val RMSE (used for the 'param + LR'
    joint stages, after each candidate's own LR sweep has picked its best)."""
    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(candidates)), 5))
    x = list(range(len(candidates)))
    colors = ['#55A868' if c == best_cand else '#4C72B0' for c in candidates]
    ax.bar(x, values, color=colors)
    for xi, v in zip(x, values):
        ax.annotate(f"{v:.4f}", (xi, v), textcoords="offset points", xytext=(0, 4),
                    ha='center', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in candidates])
    ax.set_ylabel('Best val RMSE')
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {out_path}")


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------
def tune_single_param(
    stage_name, candidates, config_updater, label_fn,
    splits, meta, info, device, dataset, best_config,
    plot_title, plot_filename,
    ncf_params, ncf_has_kwargs, tm_params, tm_has_kwargs,
):
    """One sequential single-parameter tuning stage (same mechanics as the
    original script's `tune_stage`): try each candidate, keep the one with
    lowest val RMSE, update `best_config` in place, save a line-chart plot."""
    results = {}
    x_labels = []
    for cand in candidates:
        trial_config = dict(best_config)
        config_updater(trial_config, cand)
        lbl = label_fn(cand, trial_config)
        x_labels.append(lbl)
        results[cand] = run_trial(
            stage_name, str(cand), trial_config, splits, meta, info, device, dataset.upper(),
            ncf_params, ncf_has_kwargs, tm_params, tm_has_kwargs,
        )

    best_cand = min(results, key=lambda c: results[c]['val'])
    config_updater(best_config, best_cand)

    print(f"\n{'-' * 60}")
    print(f"Stage '{stage_name}' summary ({dataset} dataset)")
    print('-' * 60)
    print(f"{'candidate':<22}{'train':>10}{'val':>10}")
    for cand in candidates:
        r = results[cand]
        marker = " *" if cand == best_cand else ""
        print(f"{str(cand):<22}{r['train']:>10.4f}{r['val']:>10.4f}{marker}")
    print(f"Best {stage_name}: {best_cand} (val RMSE = {results[best_cand]['val']:.4f})")

    plot_path = os.path.join(OUTPUT_DIR, plot_filename)
    plot_stage_results(results, x_labels, best_cand, plot_title, plot_path)
    return results, best_cand


def tune_param_with_lr(
    stage_name, candidates, config_key, lr_candidates,
    splits, meta, info, device, dataset, best_config,
    plot_title_prefix, plot_filename_prefix,
    ncf_params, ncf_has_kwargs, tm_params, tm_has_kwargs,
):
    """Joint 'param + LR' stage used for optimizer and batch_size:

    For each candidate value of `config_key`, sweep the *entire* lr_candidates
    grid and keep that candidate's own best (lowest val RMSE) LR -- no scaling
    rule is assumed between the candidate and the LR. Candidates are then
    compared against each other on their own best LR, and the winner updates
    both best_config[config_key] and best_config['lr'].

    Produces one RMSE-vs-LR line chart per candidate, plus one summary bar
    chart comparing each candidate's best val RMSE.
    """
    per_candidate_lr_results = {}
    per_candidate_best_lr = {}
    per_candidate_best_rmse = {}

    for cand in candidates:
        trial_config = dict(best_config)
        trial_config[config_key] = cand

        lr_results = {}
        for lr in lr_candidates:
            lr_config = dict(trial_config)
            lr_config['lr'] = lr
            label = f"{config_key}={cand}_lr={lr:g}"
            lr_results[lr] = run_trial(
                stage_name, label, lr_config, splits, meta, info, device, dataset.upper(),
                ncf_params, ncf_has_kwargs, tm_params, tm_has_kwargs,
            )

        best_lr = min(lr_results, key=lambda l: lr_results[l]['val'])
        per_candidate_lr_results[cand] = lr_results
        per_candidate_best_lr[cand] = best_lr
        per_candidate_best_rmse[cand] = lr_results[best_lr]

        x_labels = [f"lr={lr:g}" for lr in lr_candidates]
        plot_path = os.path.join(OUTPUT_DIR, f"{plot_filename_prefix}_{cand}_lr_{dataset}.png")
        plot_stage_results(
            lr_results, x_labels, best_lr,
            f"{plot_title_prefix}: LR sweep for {config_key}={cand} ({dataset})",
            plot_path,
        )

    best_cand = min(per_candidate_best_rmse, key=lambda c: per_candidate_best_rmse[c]['val'])
    best_config[config_key] = best_cand
    best_config['lr'] = per_candidate_best_lr[best_cand]

    print(f"\n{'-' * 60}")
    print(f"Stage '{stage_name}' summary ({dataset} dataset)")
    print('-' * 60)
    print(f"{config_key:<16}{'best_lr':>12}{'train':>10}{'val':>10}")
    for cand in candidates:
        r = per_candidate_best_rmse[cand]
        marker = " *" if cand == best_cand else ""
        print(f"{str(cand):<16}{per_candidate_best_lr[cand]:>12g}{r['train']:>10.4f}{r['val']:>10.4f}{marker}")
    print(
        f"Best {config_key}: {best_cand} @ lr={per_candidate_best_lr[best_cand]:g} "
        f"(val RMSE = {per_candidate_best_rmse[best_cand]['val']:.4f})"
    )

    summary_path = os.path.join(OUTPUT_DIR, f"{plot_filename_prefix}_summary_{dataset}.png")
    plot_candidate_summary(
        candidates,
        [per_candidate_best_rmse[c]['val'] for c in candidates],
        best_cand,
        f"{plot_title_prefix}: best val RMSE per {config_key} ({dataset})",
        summary_path,
        xlabel=config_key,
    )
    return per_candidate_lr_results, best_cand


def main(dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    ncf_params, ncf_has_kwargs, tm_params, tm_has_kwargs = _check_hooks()

    df = DATASET_LOADERS[dataset]()
    splits, meta, info = prepare_data(df, device)
    # Keep only train and val splits so tuning never touches the test split.
    splits = {'train': splits['train'], 'val': splits['val']}
    del df
    gc.collect()

    default_lr, default_wd = _get_train_defaults()

    best_config = {
        # architecture -- mechanics unchanged from the original script
        'mlp_structure': 'constant',
        'num_layers': 3,
        'num_neurons': 192,
        'mlp_layers': build_mlp_layers('constant', 3, 192),
        'embed_dim': EMBED_DIM,          # fixed, not tuned here
        'weight_decay': default_wd,       # fixed, not tuned here
        # newly-tuned knobs
        'activation': 'relu',
        'optimizer': 'adam',
        'momentum': DEFAULT_MOMENTUM,
        'batch_size': BATCH_SIZE,
        # carried-over, re-tuned at several points
        'lr': default_lr,
        'dropout': DROPOUT,
    }

    # ------------------------------------------------------------------
    # 1. Optimizer + learning rate
    # ------------------------------------------------------------------
    tune_param_with_lr(
        stage_name='optimizer',
        candidates=OPTIMIZER_CANDIDATES,
        config_key='optimizer',
        lr_candidates=LR_CANDIDATES,
        splits=splits, meta=meta, info=info, device=device, dataset=dataset,
        best_config=best_config,
        plot_title_prefix=f"NeuMF optimizer ({dataset})",
        plot_filename_prefix="optimizer",
        ncf_params=ncf_params, ncf_has_kwargs=ncf_has_kwargs,
        tm_params=tm_params, tm_has_kwargs=tm_has_kwargs,
    )

    # ------------------------------------------------------------------
    # 2. Activation function (optimizer + lr held fixed)
    # ------------------------------------------------------------------
    tune_single_param(
        stage_name='activation',
        candidates=ACTIVATION_CANDIDATES,
        config_updater=lambda cfg, a: cfg.update({'activation': a}),
        label_fn=lambda a, cfg: f"act={a}",
        splits=splits, meta=meta, info=info, device=device, dataset=dataset,
        best_config=best_config,
        plot_title=(
            f"NeuMF activation function ({dataset}, "
            f"optimizer={best_config['optimizer']}, lr={best_config['lr']:g})"
        ),
        plot_filename=f"activation_rmse_{dataset}.png",
        ncf_params=ncf_params, ncf_has_kwargs=ncf_has_kwargs,
        tm_params=tm_params, tm_has_kwargs=tm_has_kwargs,
    )

    # ------------------------------------------------------------------
    # 3. MLP structure (layers & neurons) + LR re-sweep
    #    -- structure/num_layers/num_neurons mechanics are unchanged from
    #    the original script; a LR re-sweep is appended since a deeper or
    #    wider network can destabilise optimisation at the old LR.
    # ------------------------------------------------------------------
    tune_single_param(
        stage_name='mlp_structure',
        candidates=MLP_STRUCTURES,
        config_updater=lambda cfg, s: cfg.update({
            'mlp_structure': s,
            'mlp_layers': build_mlp_layers(s, cfg['num_layers'], cfg['num_neurons']),
        }),
        label_fn=lambda s, cfg: f"{s}\n{cfg['mlp_layers']}",
        splits=splits, meta=meta, info=info, device=device, dataset=dataset,
        best_config=best_config,
        plot_title=f"NeuMF MLP structure comparison ({dataset}, batch size {best_config['batch_size']})",
        plot_filename=f"mlp_structure_rmse_{dataset}.png",
        ncf_params=ncf_params, ncf_has_kwargs=ncf_has_kwargs,
        tm_params=tm_params, tm_has_kwargs=tm_has_kwargs,
    )

    tune_single_param(
        stage_name='num_layers',
        candidates=NUM_LAYERS_CANDIDATES,
        config_updater=lambda cfg, n: cfg.update({
            'num_layers': n,
            'mlp_layers': build_mlp_layers(cfg['mlp_structure'], n, cfg['num_neurons']),
        }),
        label_fn=lambda n, cfg: f"{n}L\n{cfg['mlp_layers']}",
        splits=splits, meta=meta, info=info, device=device, dataset=dataset,
        best_config=best_config,
        plot_title=(
            f"NeuMF number of layers ({dataset}, {best_config['mlp_structure']}, "
            f"{best_config['num_neurons']} neurons)"
        ),
        plot_filename=f"num_layers_rmse_{dataset}.png",
        ncf_params=ncf_params, ncf_has_kwargs=ncf_has_kwargs,
        tm_params=tm_params, tm_has_kwargs=tm_has_kwargs,
    )

    tune_single_param(
        stage_name='num_neurons',
        candidates=NUM_NEURONS_CANDIDATES,
        config_updater=lambda cfg, total: cfg.update({
            'num_neurons': total,
            'mlp_layers': build_mlp_layers(cfg['mlp_structure'], cfg['num_layers'], total),
        }),
        label_fn=lambda total, cfg: f"{total}\n{cfg['mlp_layers']}",
        splits=splits, meta=meta, info=info, device=device, dataset=dataset,
        best_config=best_config,
        plot_title=(
            f"NeuMF total neurons ({dataset}, {best_config['mlp_structure']}, "
            f"{best_config['num_layers']} layers)"
        ),
        plot_filename=f"num_neurons_rmse_{dataset}.png",
        ncf_params=ncf_params, ncf_has_kwargs=ncf_has_kwargs,
        tm_params=tm_params, tm_has_kwargs=tm_has_kwargs,
    )

    tune_single_param(
        stage_name='lr_refit_after_architecture',
        candidates=LR_CANDIDATES,
        config_updater=lambda cfg, lr: cfg.update({'lr': lr}),
        label_fn=lambda lr, cfg: f"lr={lr:g}",
        splits=splits, meta=meta, info=info, device=device, dataset=dataset,
        best_config=best_config,
        plot_title=(
            f"NeuMF LR re-sweep after architecture change "
            f"({dataset}, layers={best_config['mlp_layers']})"
        ),
        plot_filename=f"lr_refit_after_architecture_rmse_{dataset}.png",
        ncf_params=ncf_params, ncf_has_kwargs=ncf_has_kwargs,
        tm_params=tm_params, tm_has_kwargs=tm_has_kwargs,
    )

    # ------------------------------------------------------------------
    # 4. Batch size + LR (no assumed linear-scaling rule)
    # ------------------------------------------------------------------
    tune_param_with_lr(
        stage_name='batch_size',
        candidates=BATCH_SIZE_CANDIDATES,
        config_key='batch_size',
        lr_candidates=LR_CANDIDATES,
        splits=splits, meta=meta, info=info, device=device, dataset=dataset,
        best_config=best_config,
        plot_title_prefix=f"NeuMF batch size ({dataset})",
        plot_filename_prefix="batch_size",
        ncf_params=ncf_params, ncf_has_kwargs=ncf_has_kwargs,
        tm_params=tm_params, tm_has_kwargs=tm_has_kwargs,
    )

    # ------------------------------------------------------------------
    # 5. Dropout rate
    # ------------------------------------------------------------------
    tune_single_param(
        stage_name='dropout',
        candidates=DROPOUT_CANDIDATES,
        config_updater=lambda cfg, d: cfg.update({'dropout': d}),
        label_fn=lambda d, cfg: f"drop={d:.1f}",
        splits=splits, meta=meta, info=info, device=device, dataset=dataset,
        best_config=best_config,
        plot_title=(
            f"NeuMF dropout ({dataset}, optimizer={best_config['optimizer']}, "
            f"batch_size={best_config['batch_size']})"
        ),
        plot_filename=f"dropout_rmse_{dataset}.png",
        ncf_params=ncf_params, ncf_has_kwargs=ncf_has_kwargs,
        tm_params=tm_params, tm_has_kwargs=tm_has_kwargs,
    )

    print(f"\n{'=' * 60}")
    print(f"Final best_config ({dataset} dataset)")
    print('=' * 60)
    for k, v in best_config.items():
        print(f"  {k:<15}: {v}")

    best_config_path = os.path.join(OUTPUT_DIR, f'best_config_{dataset}.json')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"\nBest configuration saved to {best_config_path}")
    print(f"Tuning history saved to {HISTORY_CSV_PATH}")
    return best_config


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Sequential hyperparameter tuning for NCF")
    p.add_argument('--dataset', choices=sorted(DATASET_LOADERS), default='preprocessed')
    args = p.parse_args()
    main(args.dataset)