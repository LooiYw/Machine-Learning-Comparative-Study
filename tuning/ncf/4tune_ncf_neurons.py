"""
Hyperparameter Tuning for NeuMF (4-Layer MLP Neuron Width Scaling).

Locks in the winning configuration so far:
- Optimizer : RMSprop
- Activation: PReLU
- Depth     : 4 hidden layers
- Batch Size: 1024

Includes:
1. Original 5-config sweep (128 to 512) [commented out in main()]
2. Full-spectrum combined sweep & plotting (128 up to 1536) [commented out in main()]
3. Multi-seed robustness evaluation of the Top 4 (MLP, LR) configurations across 5 seeds [active in main()]

Evaluates on MovieLens 1M using ONLY train and validation splits
(no test split leakage).

Outputs (saved in the same directory as this .py file):
- ncf_neurons_multiseed_history.csv            : 5-seed epoch & trial history for top 4 configs.
- ncf_scaled_neurons_multiseed_summary.png     : Mean Train vs. Val RMSE (±1 SD) + per-seed scatter.
- ncf_scaled_neurons_multiseed_convergence.png : Mean validation convergence curves (±1 SD band).

Requires: torch, numpy, pandas, matplotlib, pyarrow, scikit-learn, psutil
"""
import gc
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.ncf import (
    BATCH_SIZE,
    DROPOUT,
    EMBED_DIM,
    MAX_EPOCHS,
    NCF,
    NCFPredictor,
    PATIENCE,
    SEED,
    WEIGHT_DECAY,
    prepare_data,
    set_seed,
)
from preprocessing import load_ratings_preprocessed, load_ratings_scaled
from utils import track_resources

# =====================================================================
# Tuning Configuration
# =====================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_CSV = os.path.join(SCRIPT_DIR, "ncf_neurons_tune_history.csv")
FULL_HISTORY_CSV = os.path.join(SCRIPT_DIR, "ncf_neurons_full_spectrum_history.csv")
MULTISEED_HISTORY_CSV = os.path.join(SCRIPT_DIR, "ncf_neurons_multiseed_history.csv")

WINNING_OPTIMIZER = "RMSprop"
WINNING_ACTIVATION = "PReLU"

# Original 4-layer tapered MLP structures (128 -> 512)
NEURON_CONFIGS = {
    "4L_128 (128-64-32-16)": (128, 64, 32, 16),
    "4L_192 (192-96-48-24)": (192, 96, 48, 24),
    "4L_256 (256-128-64-32)": (256, 128, 64, 32),
    "4L_384 (384-192-96-48)": (384, 192, 96, 48),
    "4L_512 (512-256-128-64)": (512, 256, 128, 64),
}

# Full-spectrum 4-layer structures spanning narrow (128) to ultra-wide (1536)
FULL_SPECTRUM_CONFIGS = {
    "4L_128 (128-64-32-16)": (128, 64, 32, 16),
    "4L_192 (192-96-48-24)": (192, 96, 48, 24),
    "4L_256 (256-128-64-32)": (256, 128, 64, 32),
    "4L_384 (384-192-96-48)": (384, 192, 96, 48),
    "4L_512 (512-256-128-64)": (512, 256, 128, 64),
    "4L_768 (768-384-192-96)": (768, 384, 192, 96),
    "4L_1024 (1024-512-256-128)": (1024, 512, 256, 128),
    "4L_1536 (1536-768-384-192)": (1536, 768, 384, 192),
}

# Top 4 best-performing (MLP structure, LR) configurations to validate across 5 seeds
# Update these 4 entries if your leaderboard differs
TOP_4_NEURON_CONFIGS = [
    ("4L_512 (512-256-128-64)", (512, 256, 128, 64), 1e-3),
    ("4L_768 (768-384-192-96)", (768, 384, 192, 96), 1e-3),
    ("4L_1024 (1024-512-256-128)", (1024, 512, 256, 128), 1e-3),
    ("4L_1536 (1536-768-384-192)", (1536, 768, 384, 192), 5e-4),
]

MULTISEED_LIST = [42, 100, 2024, 7, 99]

LR_GRID = [5e-4, 1e-3]

TUNE_DATASETS = [
    ("NCF_SCALED", load_ratings_scaled),
    # ("NCF_PREPROCESSED", load_ratings_preprocessed),
]


def build_ncf_with_neurons(info, mlp_layers, use_features=True):
    """Builds NeuMF with the specified 4-layer neuron widths and PReLU activations."""
    model = NCF(
        info["n_users"],
        info["n_movies"],
        n_features=info["n_features"] if use_features else 0,
        embed_dim=EMBED_DIM,
        mlp_layers=mlp_layers,
        dropout=DROPOUT,
        global_mean=info["global_mean"],
    )
    for idx, layer in enumerate(model.mlp):
        if isinstance(layer, nn.ReLU):
            model.mlp[idx] = nn.PReLU()
    return model


def build_rmsprop_optimizer(model, lr, weight_decay=WEIGHT_DECAY):
    """Builds RMSprop while exempting PReLU learnable slope weights from L2 decay."""
    prelu_params = []
    other_params = []
    for module in model.modules():
        if isinstance(module, nn.PReLU):
            prelu_params.extend(module.parameters())
        else:
            other_params.extend(module.parameters(recurse=False))

    param_groups = [{"params": other_params, "weight_decay": weight_decay}]
    if prelu_params:
        param_groups.append({"params": prelu_params, "weight_decay": 0.0})

    return torch.optim.RMSprop(param_groups, lr=lr)


# =====================================================================
# Single Trial Training (Train + Val Only)
# =====================================================================

def run_trial(
    dataset_label,
    config_name,
    mlp_layers,
    lr,
    splits,
    info,
    device,
    use_features=True,
    seed=SEED,
):
    """Trains one (neuron_config, lr) combination on train and validation splits."""
    set_seed(seed)
    train, val = splits["train"], splits["val"]
    n_train = len(train["rating"])

    best = {"val_rmse": float("inf"), "train_rmse": float("inf"), "epoch": 0}
    stale, epochs_run = 0, 0
    epoch_records = []

    with track_resources(device) as res:
        model = build_ncf_with_neurons(info, mlp_layers, use_features=use_features).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        optimizer = build_rmsprop_optimizer(model, lr=lr, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.MSELoss()
        val_pred_model = NCFPredictor(model, val, use_features)

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            perm = torch.randperm(n_train, device=device)
            running_loss = 0.0

            for start in range(0, n_train, BATCH_SIZE):
                idx = perm[start : start + BATCH_SIZE]
                feats = train["features"][idx] if use_features else None
                pred = model(train["user"][idx], train["movie"][idx], feats)
                loss = loss_fn(pred, train["rating"][idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(idx)

            epochs_run = epoch
            train_rmse = (running_loss / n_train) ** 0.5
            val_p = torch.from_numpy(
                val_pred_model.predict(np.arange(len(val["rating"])))
            ).to(device)
            val_rmse = float(torch.sqrt(((val_p - val["rating"]) ** 2).mean()))

            print(
                f"  [{config_name} | lr={lr:g} | seed={seed}] Epoch {epoch:02d}/{MAX_EPOCHS:02d} | "
                f"train RMSE {train_rmse:.4f} | val RMSE {val_rmse:.4f}"
            )

            epoch_records.append({
                "dataset": dataset_label,
                "seed": seed,
                "optimizer": WINNING_OPTIMIZER,
                "activation": WINNING_ACTIVATION,
                "config_name": config_name,
                "mlp_layers": str(mlp_layers),
                "first_layer_width": mlp_layers[0],
                "lr": lr,
                "n_parameters": n_params,
                "epoch": epoch,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
            })

            if np.isnan(val_rmse) or np.isinf(val_rmse):
                print(f"  [Diverged] {config_name} lr={lr} (seed={seed}) at epoch {epoch}")
                break

            if val_rmse < best["val_rmse"] - 1e-4:
                best.update(val_rmse=val_rmse, train_rmse=train_rmse, epoch=epoch)
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    print(f"  [Early Stop] No val improvement for {PATIENCE} epochs")
                    break

    gen_gap = best["val_rmse"] - best["train_rmse"] if np.isfinite(best["val_rmse"]) else float("nan")

    trial_summary = {
        "dataset": dataset_label,
        "seed": seed,
        "optimizer": WINNING_OPTIMIZER,
        "activation": WINNING_ACTIVATION,
        "config_name": config_name,
        "mlp_layers": str(mlp_layers),
        "first_layer_width": mlp_layers[0],
        "lr": lr,
        "n_parameters": n_params,
        "best_epoch": best["epoch"],
        "epochs_run": epochs_run,
        "best_train_rmse": best["train_rmse"],
        "best_val_rmse": best["val_rmse"],
        "gen_gap": gen_gap,
        "train_time_s": res["time_s"],
        "train_mem_mb": res["mem_mb"],
    }

    for rec in epoch_records:
        rec["best_epoch"] = best["epoch"]
        rec["best_train_rmse"] = best["train_rmse"]
        rec["best_val_rmse"] = best["val_rmse"]
        rec["gen_gap"] = gen_gap
        rec["train_time_s"] = res["time_s"]
        rec["train_mem_mb"] = res["mem_mb"]

    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return trial_summary, epoch_records


# =====================================================================
# Visualization: Single-Seed Sweep Plots
# =====================================================================

def generate_neuron_plots(history_df, out_dir, prefix="neurons"):
    """Generates diagnostic plots comparing 4-layer neuron counts."""
    plt.style.use("seaborn-v0_8-whitegrid")

    for dataset_label, df_ds in history_df.groupby("dataset"):
        trials_df = (
            df_ds.drop_duplicates(subset=["config_name", "lr"])
            .sort_values(["first_layer_width", "lr"])
            .copy()
        )

        best_per_cfg = (
            trials_df[np.isfinite(trials_df["best_val_rmse"])]
            .loc[lambda d: d.groupby("config_name")["best_val_rmse"].idxmin()]
            .sort_values("first_layer_width")
        )

        winner_row = best_per_cfg.loc[best_per_cfg["best_val_rmse"].idxmin()]
        winner_label = f"{winner_row['mlp_layers']} (lr={winner_row['lr']:g})"
        winner_val_rmse = winner_row["best_val_rmse"]

        fig, ax = plt.subplots(figsize=(10.5, 6.0))
        for _, best_row in best_per_cfg.iterrows():
            cfg_name = best_row["config_name"]
            best_lr = best_row["lr"]
            curve = df_ds[(df_ds["config_name"] == cfg_name) & (df_ds["lr"] == best_lr)]
            ax.plot(
                curve["epoch"],
                curve["val_rmse"],
                marker="o",
                markersize=4,
                linewidth=2,
                label=(
                    f"{best_row['mlp_layers']} (lr={best_lr:g}, "
                    f"best={best_row['best_val_rmse']:.4f} @ ep {int(best_row['best_epoch'])})"
                ),
            )

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Validation RMSE (lower is better)", fontsize=11)
        ax.set_title(
            f"Validation Convergence by 4-Layer Neuron Width ({dataset_label})\n"
            f"Optimizer: {WINNING_OPTIMIZER} | Activation: {WINNING_ACTIVATION}",
            fontsize=11.5,
            fontweight="bold",
        )
        ax.legend(title="MLP Neuron Structure", frameon=True, fontsize=8.5)
        plt.tight_layout()
        p1 = os.path.join(out_dir, f"{dataset_label.lower()}_{prefix}_convergence.png")
        plt.savefig(p1, dpi=300)
        plt.close(fig)

        fig_width = max(13.5, len(best_per_cfg) * 2.0)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, 5.6))

        labels = [
            f"{r['mlp_layers']}\n(lr={r['lr']:g})" for _, r in best_per_cfg.iterrows()
        ]
        x = np.arange(len(labels))
        width = 0.36

        rects1 = ax1.bar(
            x - width / 2,
            best_per_cfg["best_train_rmse"],
            width,
            label="Train RMSE (at best val)",
            color="#4C72B0",
        )
        rects2 = ax1.bar(
            x + width / 2,
            best_per_cfg["best_val_rmse"],
            width,
            label="Best Val RMSE",
            color="#DD8452",
        )

        ax1.axhline(
            y=winner_val_rmse,
            color="#C44E52",
            linestyle="--",
            linewidth=1.6,
            zorder=4,
            label=f"Best Val ({winner_label}: {winner_val_rmse:.4f})",
        )

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=8.5, rotation=20, ha="right")
        ax1.set_ylabel("RMSE", fontsize=11)
        ax1.set_title("Train vs. Validation RMSE (Generalization Gap)", fontsize=11, fontweight="bold")
        min_rmse = min(best_per_cfg["best_train_rmse"].min(), best_per_cfg["best_val_rmse"].min())
        max_rmse = max(best_per_cfg["best_train_rmse"].max(), best_per_cfg["best_val_rmse"].max())
        ax1.set_ylim(max(0, min_rmse - 0.02), max_rmse + 0.03)
        ax1.legend(loc="upper left", frameon=True, fontsize=8.5)

        for rect in list(rects1) + list(rects2):
            h = rect.get_height()
            ax1.annotate(
                f"{h:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.2,
            )

        bars_time = ax2.bar(x, best_per_cfg["train_time_s"], width=0.5, color="#55A868")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=8.5, rotation=20, ha="right")
        ax2.set_ylabel("Training Time (seconds)", fontsize=11)
        ax2.set_title("Wall-Clock Training Time (until Early Stop)", fontsize=11, fontweight="bold")

        for rect in bars_time:
            h = rect.get_height()
            ax2.annotate(
                f"{h:.1f}s",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.0,
            )

        plt.suptitle(
            f"4-Layer MLP Neuron Scaling Summary ({dataset_label} | {WINNING_OPTIMIZER}, {WINNING_ACTIVATION})",
            fontsize=12.5,
            fontweight="bold",
        )
        plt.tight_layout()
        p2 = os.path.join(out_dir, f"{dataset_label.lower()}_{prefix}_summary.png")
        plt.savefig(p2, dpi=300)
        plt.close(fig)

        print(f"Saved neuron scaling plots ({prefix}) for {dataset_label} in '{out_dir}'")


# =====================================================================
# Full-Spectrum Sweep (Reuses Cached CSV + Runs Missing Wider Configs)
# =====================================================================

def run_full_spectrum_neuron_study(
    device,
    full_configs=FULL_SPECTRUM_CONFIGS,
    lr_grid=LR_GRID,
    datasets=TUNE_DATASETS,
    cached_csv=HISTORY_CSV,
    out_csv=FULL_HISTORY_CSV,
    out_dir=SCRIPT_DIR,
):
    """
    Loads any already-completed trials from cached_csv (e.g., 128-512), runs only
    the missing wider configurations (e.g., 768-1536), merges both into out_csv,
    and plots the full spectrum of MLP neuron combinations.
    """
    existing_df = pd.DataFrame()
    completed_pairs = set()

    if os.path.exists(cached_csv):
        existing_df = pd.read_csv(cached_csv)
        for _, r in existing_df[["dataset", "mlp_layers", "lr"]].drop_duplicates().iterrows():
            completed_pairs.add((r["dataset"], str(r["mlp_layers"]), float(r["lr"])))
        print(f"Loaded {len(completed_pairs)} existing (dataset, mlp_layers, lr) trials from '{cached_csv}'")

    new_epoch_records = []

    for dataset_label, loader in datasets:
        pending = []
        for cfg_name, mlp_layers in full_configs.items():
            for lr in lr_grid:
                if (dataset_label, str(mlp_layers), float(lr)) not in completed_pairs:
                    pending.append((cfg_name, mlp_layers, lr))

        if not pending:
            print(f"All full-spectrum configurations for {dataset_label} are already cached!")
            continue

        print(
            f"\n{'=' * 70}\n"
            f"=== RUNNING {len(pending)} NEW WIDE NEURON TRIALS ON {dataset_label} ===\n"
            f"{'=' * 70}"
        )
        df = loader()
        splits, meta, info = prepare_data(df, device)
        del df, meta, splits["test"]
        gc.collect()

        for cfg_name, mlp_layers, lr in pending:
            print(f"\n--- Neuron Config: {cfg_name} | LR: {lr:g} ---")
            summary, epoch_records = run_trial(
                dataset_label, cfg_name, mlp_layers, lr, splits, info, device, use_features=True
            )
            new_epoch_records.extend(epoch_records)

            print(
                f"Config={cfg_name:<27} | LR={lr:<7g} | Best Ep={summary['best_epoch']:02d} | "
                f"Train={summary['best_train_rmse']:.4f} | Val={summary['best_val_rmse']:.4f} | "
                f"Gap={summary['gen_gap']:.4f} | Time={summary['train_time_s']:.1f}s"
            )

        del splits
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    new_df = pd.DataFrame(new_epoch_records)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)

    valid_layers = {str(v) for v in full_configs.values()}
    combined_df = combined_df[combined_df["mlp_layers"].isin(valid_layers)].copy()
    combined_df.to_csv(out_csv, index=False)
    print(f"\nSaved full-spectrum epoch history ({len(combined_df)} rows) to '{out_csv}'")

    trials_df = combined_df.drop_duplicates(subset=["dataset", "mlp_layers", "lr"]).copy()
    best_leaderboard = (
        trials_df[np.isfinite(trials_df["best_val_rmse"])]
        .loc[lambda d: d.groupby(["dataset", "mlp_layers"])["best_val_rmse"].idxmin()]
        .sort_values(["dataset", "first_layer_width"])
    )
    print("\n=== FULL-SPECTRUM 4-LAYER NEURON LEADERBOARD (ORDERED NARROW -> WIDE) ===")
    cols = [
        "dataset", "config_name", "lr", "n_parameters", "best_epoch",
        "best_train_rmse", "best_val_rmse", "gen_gap", "train_time_s"
    ]
    print(best_leaderboard[cols].to_string(index=False))

    generate_neuron_plots(combined_df, out_dir, prefix="neurons_full")
    return combined_df, best_leaderboard


# =====================================================================
# Multi-Seed Evaluation & Visualization for Top 4 Neuron Configs (Added)
# =====================================================================

def run_multiseed_neuron_study(
    device,
    top_configs=TOP_4_NEURON_CONFIGS,
    seeds=MULTISEED_LIST,
    datasets=TUNE_DATASETS,
    out_dir=SCRIPT_DIR,
    csv_path=MULTISEED_HISTORY_CSV,
):
    """
    Evaluates the Top 4 best-performing (MLP structure, LR) configurations across
    5 random seeds, computes mean/std RMSE and generalization gap across seeds,
    saves full epoch history to CSV, and generates comparison plots.
    """
    print(
        f"\nUsing device: {device} | Optimizer: {WINNING_OPTIMIZER} | "
        f"Activation: {WINNING_ACTIVATION} | Seeds: {seeds}"
    )

    all_epoch_history = []
    all_trial_summaries = []

    for dataset_label, loader in datasets:
        print(
            f"\n{'=' * 72}\n"
            f"=== MULTI-SEED TOP-4 NEURON CONFIG TUNING ({len(seeds)} SEEDS): {dataset_label} ===\n"
            f"{'=' * 72}"
        )
        df = loader()
        splits, meta, info = prepare_data(df, device)
        del df, meta, splits["test"]
        gc.collect()

        for cfg_name, mlp_layers, lr in top_configs:
            print(f"\n--- Config: {cfg_name} | LR={lr:g} (across {len(seeds)} seeds) ---")

            for seed in seeds:
                summary, epoch_records = run_trial(
                    dataset_label,
                    cfg_name,
                    mlp_layers,
                    lr,
                    splits,
                    info,
                    device,
                    use_features=True,
                    seed=seed,
                )
                all_trial_summaries.append(summary)
                all_epoch_history.extend(epoch_records)

                print(
                    f"Config={cfg_name:<27} | LR={lr:<7g} | Seed={seed:<4} | "
                    f"Best Ep={summary['best_epoch']:02d} | Train={summary['best_train_rmse']:.4f} | "
                    f"Val={summary['best_val_rmse']:.4f} | Gap={summary['gen_gap']:.4f} | "
                    f"Time={summary['train_time_s']:.1f}s"
                )

        del splits
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    history_df = pd.DataFrame(all_epoch_history)
    history_df.to_csv(csv_path, index=False)
    print(f"\nSaved full multi-seed neuron epoch history to '{csv_path}'")

    summary_df = pd.DataFrame(all_trial_summaries)
    agg_df = (
        summary_df.groupby(
            ["dataset", "config_name", "mlp_layers", "first_layer_width", "lr", "n_parameters"],
            as_index=False,
        )
        .agg(
            n_seeds=("seed", "count"),
            mean_best_epoch=("best_epoch", "mean"),
            mean_train_rmse=("best_train_rmse", "mean"),
            std_train_rmse=("best_train_rmse", "std"),
            mean_val_rmse=("best_val_rmse", "mean"),
            std_val_rmse=("best_val_rmse", "std"),
            min_val_rmse=("best_val_rmse", "min"),
            max_val_rmse=("best_val_rmse", "max"),
            mean_gen_gap=("gen_gap", "mean"),
            mean_train_time_s=("train_time_s", "mean"),
        )
        .sort_values(["dataset", "mean_val_rmse"])
    )

    print("\n=== MULTI-SEED TOP-4 NEURON CONFIG LEADERBOARD (SORTED BY MEAN VAL RMSE) ===")
    print(agg_df.to_string(index=False))

    generate_multiseed_neuron_plots(history_df, summary_df, agg_df, out_dir)
    return summary_df, agg_df


def generate_multiseed_neuron_plots(history_df, summary_df, agg_df, out_dir):
    """
    Generates multi-seed diagnostic plots for Top 4 MLP configurations:
    1. Mean Train vs. Mean Val RMSE (±1 SD error bars) with best-mean horizontal line
       + per-seed scatter distribution across the 5 seeds.
    2. Mean Validation Convergence Curves (±1 SD shaded band across seeds).
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    for dataset_label, agg_ds in agg_df.groupby("dataset"):
        agg_ds = agg_ds.sort_values("mean_val_rmse").reset_index(drop=True)
        sum_ds = summary_df[summary_df["dataset"] == dataset_label]
        hist_ds = history_df[history_df["dataset"] == dataset_label]

        winner_row = agg_ds.iloc[0]
        winner_label = f"{winner_row['mlp_layers']} (lr={winner_row['lr']:g})"
        winner_mean_val = winner_row["mean_val_rmse"]
        n_seeds = int(winner_row["n_seeds"])

        # -------------------------------------------------------------
        # Plot 1: Mean Train vs. Val RMSE (Error Bars) + Seed Scatter
        # -------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.0, 5.6))

        labels = [
            f"{r['mlp_layers']}\n(lr={r['lr']:g})" for _, r in agg_ds.iterrows()
        ]
        x = np.arange(len(labels))
        width = 0.36

        rects1 = ax1.bar(
            x - width / 2,
            agg_ds["mean_train_rmse"],
            width,
            yerr=agg_ds["std_train_rmse"].fillna(0),
            capsize=4,
            label="Mean Train RMSE (±1 SD)",
            color="#4C72B0",
            error_kw={"elinewidth": 1.2, "ecolor": "#1f2d46"},
        )
        rects2 = ax1.bar(
            x + width / 2,
            agg_ds["mean_val_rmse"],
            width,
            yerr=agg_ds["std_val_rmse"].fillna(0),
            capsize=4,
            label="Mean Val RMSE (±1 SD)",
            color="#DD8452",
            error_kw={"elinewidth": 1.2, "ecolor": "#5c3317"},
        )

        # Horizontal dashed benchmark line at the best mean Val RMSE
        ax1.axhline(
            y=winner_mean_val,
            color="#C44E52",
            linestyle="--",
            linewidth=1.6,
            zorder=4,
            label=f"Best Mean Val ({winner_label}: {winner_mean_val:.4f})",
        )

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9, rotation=12)
        ax1.set_ylabel("RMSE", fontsize=11)
        ax1.set_title(
            f"Mean Train vs. Val RMSE Across {n_seeds} Seeds",
            fontsize=11,
            fontweight="bold",
        )
        min_rmse = min(
            (agg_ds["mean_train_rmse"] - agg_ds["std_train_rmse"].fillna(0)).min(),
            (agg_ds["mean_val_rmse"] - agg_ds["std_val_rmse"].fillna(0)).min(),
        )
        max_rmse = max(
            (agg_ds["mean_train_rmse"] + agg_ds["std_train_rmse"].fillna(0)).max(),
            (agg_ds["mean_val_rmse"] + agg_ds["std_val_rmse"].fillna(0)).max(),
        )
        ax1.set_ylim(max(0, min_rmse - 0.015), max_rmse + 0.025)
        ax1.legend(loc="upper left", frameon=True, fontsize=8.2)

        for rect, (_, r) in zip(rects1, agg_ds.iterrows()):
            h = rect.get_height()
            offset = r["std_train_rmse"] if pd.notna(r["std_train_rmse"]) else 0.0
            ax1.annotate(
                f"{h:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, h + offset),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.8,
            )

        for rect, (_, r) in zip(rects2, agg_ds.iterrows()):
            h = rect.get_height()
            offset = r["std_val_rmse"] if pd.notna(r["std_val_rmse"]) else 0.0
            ax1.annotate(
                f"{h:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, h + offset),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.8,
            )

        # Right subplot: Per-seed Val RMSE distribution + Mean diamond
        for idx_cfg, (_, row) in enumerate(agg_ds.iterrows()):
            cfg_name = row["config_name"]
            lr_val = row["lr"]
            sub = sum_ds[(sum_ds["config_name"] == cfg_name) & (sum_ds["lr"] == lr_val)]
            jitter = np.linspace(-0.12, 0.12, len(sub))
            ax2.scatter(
                np.full(len(sub), idx_cfg) + jitter,
                sub["best_val_rmse"],
                color="#4C72B0",
                alpha=0.85,
                s=55,
                zorder=4,
                label="Individual Seed" if idx_cfg == 0 else "",
            )
            ax2.scatter(
                [idx_cfg],
                [row["mean_val_rmse"]],
                color="#C44E52",
                marker="D",
                s=85,
                zorder=5,
                edgecolors="black",
                linewidths=0.8,
                label="Mean Val RMSE" if idx_cfg == 0 else "",
            )

        ax2.axhline(
            y=winner_mean_val,
            color="#C44E52",
            linestyle="--",
            linewidth=1.4,
            alpha=0.85,
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=9, rotation=12)
        ax2.set_ylabel("Best Validation RMSE per Seed", fontsize=11)
        ax2.set_title(
            f"Seed-to-Seed Stability ({n_seeds} Seeds)",
            fontsize=11,
            fontweight="bold",
        )
        ax2.legend(loc="upper right", frameon=True, fontsize=8.5)

        plt.suptitle(
            f"Multi-Seed Top-4 MLP Configurations ({dataset_label} | {WINNING_OPTIMIZER}, {WINNING_ACTIVATION})",
            fontsize=12.5,
            fontweight="bold",
        )
        plt.tight_layout()
        p1 = os.path.join(out_dir, f"{dataset_label.lower()}_neurons_multiseed_summary.png")
        plt.savefig(p1, dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # Plot 2: Mean Validation Convergence (± 1 SD Band)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10.0, 5.6))
        for _, row in agg_ds.iterrows():
            cfg_name = row["config_name"]
            lr_val = row["lr"]
            curve_stats = (
                hist_ds[(hist_ds["config_name"] == cfg_name) & (hist_ds["lr"] == lr_val)]
                .groupby("epoch", as_index=False)
                .agg(mean_val=("val_rmse", "mean"), std_val=("val_rmse", "std"))
            )
            mean_line, = ax.plot(
                curve_stats["epoch"],
                curve_stats["mean_val"],
                marker="o",
                markersize=3.5,
                linewidth=2,
                label=(
                    f"{row['mlp_layers']} (lr={lr_val:g}, "
                    f"mean={row['mean_val_rmse']:.4f} ± {row['std_val_rmse']:.4f})"
                ),
            )
            std_vals = curve_stats["std_val"].fillna(0)
            ax.fill_between(
                curve_stats["epoch"],
                curve_stats["mean_val"] - std_vals,
                curve_stats["mean_val"] + std_vals,
                color=mean_line.get_color(),
                alpha=0.15,
            )

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Validation RMSE (Mean ± 1 SD)", fontsize=11)
        ax.set_title(
            f"Multi-Seed Validation Convergence for Top-4 MLP Configs ({dataset_label} | {n_seeds} Seeds)\n"
            f"Optimizer: {WINNING_OPTIMIZER} | Activation: {WINNING_ACTIVATION}",
            fontsize=11.5,
            fontweight="bold",
        )
        ax.legend(title="MLP Structure & LR", frameon=True, fontsize=8.5)
        plt.tight_layout()
        p2 = os.path.join(out_dir, f"{dataset_label.lower()}_neurons_multiseed_convergence.png")
        plt.savefig(p2, dpi=300)
        plt.close(fig)

        print(f"Saved multi-seed neuron plots for {dataset_label} in '{out_dir}'")


# =====================================================================
# Main Tuning Loop
# =====================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------
    # [COMMENTED OUT] Original 5-config sweep (128 to 512)
    # -----------------------------------------------------------------
    # print(
    #     f"Using device: {device} | Optimizer: {WINNING_OPTIMIZER} | "
    #     f"Activation: {WINNING_ACTIVATION} | Batch Size: {BATCH_SIZE}"
    # )
    #
    # all_epoch_history = []
    # all_trial_summaries = []
    #
    # for dataset_label, loader in TUNE_DATASETS:
    #     print(f"\n{'=' * 68}\n=== 4-LAYER NEURON WIDTH TUNING ON DATASET: {dataset_label} ===\n{'=' * 68}")
    #     df = loader()
    #     splits, meta, info = prepare_data(df, device)
    #     del df, meta, splits["test"]
    #     gc.collect()
    #
    #     for config_name, mlp_layers in NEURON_CONFIGS.items():
    #         print(f"\n--- Neuron Config: {config_name} ---")
    #         for lr in LR_GRID:
    #             summary, epoch_records = run_trial(
    #                 dataset_label, config_name, mlp_layers, lr, splits, info, device, use_features=True
    #             )
    #             all_trial_summaries.append(summary)
    #             all_epoch_history.extend(epoch_records)
    #
    #             print(
    #                 f"Config={config_name:<25} | LR={lr:<7g} | Best Ep={summary['best_epoch']:02d} | "
    #                 f"Train={summary['best_train_rmse']:.4f} | Val={summary['best_val_rmse']:.4f} | "
    #                 f"Gap={summary['gen_gap']:.4f} | Time={summary['train_time_s']:.1f}s"
    #             )
    #
    #     del splits
    #     gc.collect()
    #     if device.type == "cuda":
    #         torch.cuda.empty_cache()
    #
    # history_df = pd.DataFrame(all_epoch_history)
    # history_df.to_csv(HISTORY_CSV, index=False)
    # print(f"\nSaved full epoch-by-epoch tuning history to '{HISTORY_CSV}'")
    #
    # summary_df = pd.DataFrame(all_trial_summaries)
    # best_leaderboard = (
    #     summary_df[np.isfinite(summary_df["best_val_rmse"])]
    #     .loc[lambda d: d.groupby(["dataset", "config_name"])["best_val_rmse"].idxmin()]
    #     .sort_values(["dataset", "best_val_rmse"])
    # )
    # print("\n=== BEST LR PER 4-LAYER NEURON CONFIG (VALIDATION ONLY) ===")
    # print(best_leaderboard.to_string(index=False))
    #
    # generate_neuron_plots(history_df, SCRIPT_DIR)

    # -----------------------------------------------------------------
    # [COMMENTED OUT] Full-spectrum sweep (128 -> 1536)
    # -----------------------------------------------------------------
    # run_full_spectrum_neuron_study(
    #     device=device,
    #     full_configs=FULL_SPECTRUM_CONFIGS,
    #     lr_grid=LR_GRID,
    #     datasets=TUNE_DATASETS,
    #     cached_csv=HISTORY_CSV,
    #     out_csv=FULL_HISTORY_CSV,
    #     out_dir=SCRIPT_DIR,
    # )

    # -----------------------------------------------------------------
    # [ACTIVE] Multi-seed evaluation of Top 4 (MLP, LR) configs across 5 seeds
    # -----------------------------------------------------------------
    run_multiseed_neuron_study(
        device=device,
        top_configs=TOP_4_NEURON_CONFIGS,
        seeds=MULTISEED_LIST,
        datasets=TUNE_DATASETS,
        out_dir=SCRIPT_DIR,
        csv_path=MULTISEED_HISTORY_CSV,
    )


if __name__ == "__main__":
    main()