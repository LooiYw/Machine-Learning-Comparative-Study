"""
Hyperparameter Tuning for NeuMF (Activation Function Comparison).

Fixes the winning optimizer (RMSprop, lr=0.0005) and compares:
1. ReLU
2. LeakyReLU
3. PReLU
4. GELU
5. ELU
6. SiLU
7. Mish

Also includes multi-seed robustness evaluation for top activation candidates
(PReLU, ReLU, LeakyReLU, GELU) across 5 seeds.

Evaluates on MovieLens 1M using ONLY train and validation splits
(no test split leakage).

Outputs (saved in the same directory as this .py file):
- ncf_activation_tune_history.csv      : Single-seed full sweep history.
- ncf_activation_multiseed_history.csv : Multi-seed evaluation history.
- *.png                                : Comparison plots across activation functions.

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
    MLP_LAYERS,
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
HISTORY_CSV = os.path.join(SCRIPT_DIR, "ncf_activation_tune_history.csv")
MULTISEED_HISTORY_CSV = os.path.join(SCRIPT_DIR, "ncf_activation_multiseed_history.csv")

WINNING_OPTIMIZER = "RMSprop"
WINNING_LR = 0.0005

ACTIVATION_CANDIDATES = {
    "ReLU": lambda: nn.ReLU(),
    "LeakyReLU": lambda: nn.LeakyReLU(negative_slope=0.01),
    "PReLU": lambda: nn.PReLU(),
    "GELU": lambda: nn.GELU(),
    "ELU": lambda: nn.ELU(alpha=1.0),
    "SiLU": lambda: nn.SiLU(),
    "Mish": lambda: nn.Mish(),
}

# Multi-seed follow-up configuration
MULTISEED_ACTIVATIONS = ["PReLU", "ReLU", "LeakyReLU", "GELU"]
MULTISEED_LIST = [42, 100, 2024, 7, 99]

TUNE_DATASETS = [
    ("NCF_SCALED", load_ratings_scaled),
    # ("NCF_PREPROCESSED", load_ratings_preprocessed),
]


def build_ncf_with_activation(info, act_factory, use_features=True):
    """Builds NCF and swaps MLP ReLU layers with the chosen activation function."""
    model = NCF(
        info["n_users"],
        info["n_movies"],
        n_features=info["n_features"] if use_features else 0,
        embed_dim=EMBED_DIM,
        mlp_layers=MLP_LAYERS,
        dropout=DROPOUT,
        global_mean=info["global_mean"],
    )
    for idx, layer in enumerate(model.mlp):
        if isinstance(layer, nn.ReLU):
            model.mlp[idx] = act_factory()
    return model


def build_rmsprop_optimizer(model, lr=WINNING_LR, weight_decay=WEIGHT_DECAY):
    """
    Builds RMSprop while exempting PReLU learnable slope parameters from weight decay
    (PyTorch best practice so L2 regularization does not shrink PReLU slopes to 0).
    """
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
    act_name,
    act_factory,
    splits,
    info,
    device,
    use_features=True,
    seed=SEED,
):
    """Trains one activation configuration with RMSprop (lr=0.0005) on train/val."""
    set_seed(seed)
    train, val = splits["train"], splits["val"]
    n_train = len(train["rating"])

    best = {"val_rmse": float("inf"), "train_rmse": float("inf"), "epoch": 0}
    stale, epochs_run = 0, 0
    epoch_records = []

    with track_resources(device) as res:
        model = build_ncf_with_activation(info, act_factory, use_features=use_features).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        optimizer = build_rmsprop_optimizer(model, lr=WINNING_LR, weight_decay=WEIGHT_DECAY)
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
                f"  [{act_name} | seed={seed}] Epoch {epoch:02d}/{MAX_EPOCHS:02d} | "
                f"train RMSE {train_rmse:.4f} | val RMSE {val_rmse:.4f}"
            )

            epoch_records.append({
                "dataset": dataset_label,
                "seed": seed,
                "optimizer": WINNING_OPTIMIZER,
                "lr": WINNING_LR,
                "activation": act_name,
                "n_parameters": n_params,
                "epoch": epoch,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
            })

            if np.isnan(val_rmse) or np.isinf(val_rmse):
                print(f"  [Diverged] {act_name} (seed={seed}) at epoch {epoch}")
                break

            if val_rmse < best["val_rmse"] - 1e-4:
                best.update(val_rmse=val_rmse, train_rmse=train_rmse, epoch=epoch)
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    print(f"  [Early Stop] No val improvement for {PATIENCE} epochs")
                    break

    trial_summary = {
        "dataset": dataset_label,
        "seed": seed,
        "optimizer": WINNING_OPTIMIZER,
        "lr": WINNING_LR,
        "activation": act_name,
        "n_parameters": n_params,
        "best_epoch": best["epoch"],
        "epochs_run": epochs_run,
        "best_train_rmse": best["train_rmse"],
        "best_val_rmse": best["val_rmse"],
        "train_time_s": res["time_s"],
        "train_mem_mb": res["mem_mb"],
    }

    for rec in epoch_records:
        rec["best_epoch"] = best["epoch"]
        rec["best_train_rmse"] = best["train_rmse"]
        rec["best_val_rmse"] = best["val_rmse"]
        rec["train_time_s"] = res["time_s"]
        rec["train_mem_mb"] = res["mem_mb"]

    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return trial_summary, epoch_records


# =====================================================================
# Visualization: Single-Seed Activation Sweep (Original)
# =====================================================================

def generate_activation_plots(history_df, out_dir):
    """Generates and saves diagnostic charts comparing activation functions."""
    plt.style.use("seaborn-v0_8-whitegrid")

    for dataset_label, df_ds in history_df.groupby("dataset"):
        trials_df = (
            df_ds.drop_duplicates(subset=["activation"])
            .sort_values("best_val_rmse")
            .copy()
        )

        winner_row = trials_df.iloc[0]
        winner_name = winner_row["activation"]
        winner_val_rmse = winner_row["best_val_rmse"]

        # -------------------------------------------------------------
        # Plot 1: Epoch-by-Epoch Validation Convergence per Activation
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        for _, row in trials_df.iterrows():
            act_name = row["activation"]
            curve = df_ds[df_ds["activation"] == act_name]
            ax.plot(
                curve["epoch"],
                curve["val_rmse"],
                marker="o",
                markersize=4,
                linewidth=2,
                label=f"{act_name} (best={row['best_val_rmse']:.4f} @ ep {int(row['best_epoch'])})",
            )

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Validation RMSE (lower is better)", fontsize=11)
        ax.set_title(
            f"Activation Function Convergence ({dataset_label})\n"
            f"Optimizer: {WINNING_OPTIMIZER} (lr={WINNING_LR:g})",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(title="Activation Function", frameon=True)
        plt.tight_layout()
        p1 = os.path.join(out_dir, f"{dataset_label.lower()}_activation_convergence.png")
        plt.savefig(p1, dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # Plot 2: Train vs. Val RMSE & Training Time Comparison
        # -------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2))

        labels = trials_df["activation"].tolist()
        x = np.arange(len(labels))
        width = 0.36

        rects1 = ax1.bar(
            x - width / 2,
            trials_df["best_train_rmse"],
            width,
            label="Train RMSE (at best val)",
            color="#4C72B0",
        )
        rects2 = ax1.bar(
            x + width / 2,
            trials_df["best_val_rmse"],
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
            label=f"Best Val ({winner_name}: {winner_val_rmse:.4f})",
        )

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9.5, rotation=15)
        ax1.set_ylabel("RMSE", fontsize=11)
        ax1.set_title("Train vs. Validation RMSE (Generalization Gap)", fontsize=11, fontweight="bold")
        min_rmse = min(trials_df["best_train_rmse"].min(), trials_df["best_val_rmse"].min())
        max_rmse = max(trials_df["best_train_rmse"].max(), trials_df["best_val_rmse"].max())
        ax1.set_ylim(max(0, min_rmse - 0.02), max_rmse + 0.025)
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
                fontsize=7.5,
            )

        bars_time = ax2.bar(x, trials_df["train_time_s"], width=0.5, color="#55A868")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=9.5, rotation=15)
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
                fontsize=8.5,
            )

        plt.suptitle(
            f"Activation Function Comparison ({dataset_label} | {WINNING_OPTIMIZER}, lr={WINNING_LR:g})",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        p2 = os.path.join(out_dir, f"{dataset_label.lower()}_activation_summary.png")
        plt.savefig(p2, dpi=300)
        plt.close(fig)

        print(f"Saved activation plots for {dataset_label} in '{out_dir}'")


# =====================================================================
# Multi-Seed Evaluation & Visualization (Added)
# =====================================================================

def run_multiseed_activation_study(
    device,
    activations=MULTISEED_ACTIVATIONS,
    seeds=MULTISEED_LIST,
    datasets=TUNE_DATASETS,
    out_dir=SCRIPT_DIR,
    csv_path=MULTISEED_HISTORY_CSV,
):
    """
    Evaluates selected activation functions across multiple random seeds,
    computes mean/std RMSE across seeds, saves CSV logs, and generates plots.
    """
    print(
        f"\nUsing device: {device} | Optimizer: {WINNING_OPTIMIZER} (lr={WINNING_LR}) | "
        f"Seeds: {seeds}"
    )

    all_epoch_history = []
    all_trial_summaries = []

    for dataset_label, loader in datasets:
        print(
            f"\n{'=' * 70}\n"
            f"=== MULTI-SEED ACTIVATION TUNING ({len(seeds)} SEEDS): {dataset_label} ===\n"
            f"{'=' * 70}"
        )
        df = loader()
        splits, meta, info = prepare_data(df, device)
        del df, meta, splits["test"]
        gc.collect()

        for act_name in activations:
            act_factory = ACTIVATION_CANDIDATES[act_name]
            print(f"\n--- Activation: {act_name} (across {len(seeds)} seeds) ---")

            for seed in seeds:
                summary, epoch_records = run_trial(
                    dataset_label,
                    act_name,
                    act_factory,
                    splits,
                    info,
                    device,
                    use_features=True,
                    seed=seed,
                )
                all_trial_summaries.append(summary)
                all_epoch_history.extend(epoch_records)

                print(
                    f"Activation={act_name:<10} | Seed={seed:<4} | "
                    f"Best Ep={summary['best_epoch']:02d} | "
                    f"Train={summary['best_train_rmse']:.4f} | "
                    f"Val={summary['best_val_rmse']:.4f} | "
                    f"Time={summary['train_time_s']:.1f}s"
                )

        del splits
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    history_df = pd.DataFrame(all_epoch_history)
    history_df.to_csv(csv_path, index=False)
    print(f"\nSaved full multi-seed epoch history to '{csv_path}'")

    summary_df = pd.DataFrame(all_trial_summaries)
    agg_df = (
        summary_df.groupby(["dataset", "activation"], as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            mean_best_epoch=("best_epoch", "mean"),
            mean_train_rmse=("best_train_rmse", "mean"),
            std_train_rmse=("best_train_rmse", "std"),
            mean_val_rmse=("best_val_rmse", "mean"),
            std_val_rmse=("best_val_rmse", "std"),
            min_val_rmse=("best_val_rmse", "min"),
            max_val_rmse=("best_val_rmse", "max"),
            mean_train_time_s=("train_time_s", "mean"),
        )
        .sort_values(["dataset", "mean_val_rmse"])
    )

    print("\n=== MULTI-SEED ACTIVATION LEADERBOARD (SORTED BY MEAN VAL RMSE) ===")
    print(agg_df.to_string(index=False))

    generate_multiseed_activation_plots(history_df, summary_df, agg_df, out_dir)
    return summary_df, agg_df


def generate_multiseed_activation_plots(history_df, summary_df, agg_df, out_dir):
    """
    Generates multi-seed diagnostic plots:
    1. Mean Train vs. Mean Val RMSE (+/- 1 SD error bars) with best-mean horizontal line
       + per-seed scatter distribution across the 5 seeds.
    2. Mean Validation Convergence Curves (+/- 1 SD shaded band across seeds).
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    for dataset_label, agg_ds in agg_df.groupby("dataset"):
        agg_ds = agg_ds.sort_values("mean_val_rmse").reset_index(drop=True)
        sum_ds = summary_df[summary_df["dataset"] == dataset_label]
        hist_ds = history_df[history_df["dataset"] == dataset_label]

        winner_row = agg_ds.iloc[0]
        winner_name = winner_row["activation"]
        winner_mean_val = winner_row["mean_val_rmse"]
        n_seeds = int(winner_row["n_seeds"])

        # -------------------------------------------------------------
        # Plot 1: Mean Train vs. Val RMSE (Error Bars) + Seed Scatter
        # -------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

        labels = agg_ds["activation"].tolist()
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

        # Horizontal reference line marking best mean Val RMSE
        ax1.axhline(
            y=winner_mean_val,
            color="#C44E52",
            linestyle="--",
            linewidth=1.6,
            zorder=4,
            label=f"Best Mean Val ({winner_name}: {winner_mean_val:.4f})",
        )

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=10)
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
        ax1.set_ylim(max(0, min_rmse - 0.015), max_rmse + 0.022)
        ax1.legend(loc="upper left", frameon=True, fontsize=8.5)

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
                fontsize=8,
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
                fontsize=8,
            )

        # Right subplot: Per-seed Val RMSE distribution + Mean diamond
        for idx_act, act_name in enumerate(labels):
            sub = sum_ds[sum_ds["activation"] == act_name]
            jitter = np.linspace(-0.12, 0.12, len(sub))
            ax2.scatter(
                np.full(len(sub), idx_act) + jitter,
                sub["best_val_rmse"],
                color="#4C72B0",
                alpha=0.85,
                s=55,
                zorder=4,
                label="Individual Seed" if idx_act == 0 else "",
            )
            mean_val = agg_ds.loc[idx_act, "mean_val_rmse"]
            ax2.scatter(
                [idx_act],
                [mean_val],
                color="#C44E52",
                marker="D",
                s=85,
                zorder=5,
                edgecolors="black",
                linewidths=0.8,
                label="Mean Val RMSE" if idx_act == 0 else "",
            )

        ax2.axhline(
            y=winner_mean_val,
            color="#C44E52",
            linestyle="--",
            linewidth=1.4,
            alpha=0.85,
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=10)
        ax2.set_ylabel("Best Validation RMSE per Seed", fontsize=11)
        ax2.set_title(
            f"Seed-to-Seed Stability ({n_seeds} Seeds)",
            fontsize=11,
            fontweight="bold",
        )
        ax2.legend(loc="upper right", frameon=True, fontsize=8.5)

        plt.suptitle(
            f"Multi-Seed Activation Function Comparison ({dataset_label} | {WINNING_OPTIMIZER}, lr={WINNING_LR:g})",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        p1 = os.path.join(out_dir, f"{dataset_label.lower()}_activation_multiseed_summary.png")
        plt.savefig(p1, dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # Plot 2: Mean Validation Convergence (+/- 1 SD Band)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        for _, row in agg_ds.iterrows():
            act_name = row["activation"]
            curve_stats = (
                hist_ds[hist_ds["activation"] == act_name]
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
                    f"{act_name} (mean={row['mean_val_rmse']:.4f} "
                    f"± {row['std_val_rmse']:.4f})"
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
            f"Multi-Seed Validation Convergence ({dataset_label} | {n_seeds} Seeds)\n"
            f"Optimizer: {WINNING_OPTIMIZER} (lr={WINNING_LR:g})",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(title="Activation Function", frameon=True)
        plt.tight_layout()
        p2 = os.path.join(out_dir, f"{dataset_label.lower()}_activation_multiseed_convergence.png")
        plt.savefig(p2, dpi=300)
        plt.close(fig)

        print(f"Saved multi-seed activation plots for {dataset_label} in '{out_dir}'")


# =====================================================================
# Main Tuning Loop
# =====================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------
    # [COMMENTED OUT] Original single-seed 7-activation sweep
    # -----------------------------------------------------------------
    # print(f"Using device: {device} | Optimizer: {WINNING_OPTIMIZER} (lr={WINNING_LR})")
    #
    # all_epoch_history = []
    # all_trial_summaries = []
    #
    # for dataset_label, loader in TUNE_DATASETS:
    #     print(f"\n{'=' * 65}\n=== ACTIVATION TUNING ON DATASET: {dataset_label} ===\n{'=' * 65}")
    #     df = loader()
    #     splits, meta, info = prepare_data(df, device)
    #     del df, meta, splits["test"]
    #     gc.collect()
    #
    #     for act_name, act_factory in ACTIVATION_CANDIDATES.items():
    #         print(f"\n--- Activation: {act_name} ---")
    #         summary, epoch_records = run_trial(
    #             dataset_label, act_name, act_factory, splits, info, device, use_features=True
    #         )
    #         all_trial_summaries.append(summary)
    #         all_epoch_history.extend(epoch_records)
    #
    #         print(
    #             f"Activation={act_name:<12} | Best Epoch={summary['best_epoch']:02d} | "
    #             f"Train RMSE={summary['best_train_rmse']:.4f} | "
    #             f"Val RMSE={summary['best_val_rmse']:.4f} | "
    #             f"Time={summary['train_time_s']:.1f}s"
    #         )
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
    # summary_df = pd.DataFrame(all_trial_summaries).sort_values(["dataset", "best_val_rmse"])
    # print("\n=== ACTIVATION FUNCTION LEADERBOARD (VALIDATION ONLY) ===")
    # print(summary_df.to_string(index=False))
    #
    # generate_activation_plots(history_df, SCRIPT_DIR)

    # -----------------------------------------------------------------
    # [ACTIVE] Multi-seed evaluation on PReLU, ReLU, LeakyReLU, GELU
    # -----------------------------------------------------------------
    run_multiseed_activation_study(
        device=device,
        activations=MULTISEED_ACTIVATIONS,
        seeds=MULTISEED_LIST,
        datasets=TUNE_DATASETS,
        out_dir=SCRIPT_DIR,
        csv_path=MULTISEED_HISTORY_CSV,
    )


if __name__ == "__main__":
    main()