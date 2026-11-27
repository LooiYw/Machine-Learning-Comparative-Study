"""
Hyperparameter Tuning for NeuMF (Optimizer + Learning Rate Grid Search).

Compares Adam, AdamW, SGD (with Momentum), RMSprop, and Adagrad, tuning the
learning rate independently for each optimizer on MovieLens 1M using ONLY
the train and validation splits (no test split leakage).

Outputs:
- ncf_tune_history.csv : Full trial & epoch-level tuning history (train/val only).
- tuning>ncf/*.png     : Comparison plots across optimizers and learning rates.

Requires: torch, numpy, pandas, matplotlib, pyarrow, scikit-learn, psutil
"""
import copy
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

HISTORY_CSV = "ncf_tune_history.csv"
PLOT_DIR = "tuning>ncf"  # Safely falls back to "tuning_ncf" on Windows

OPTIMIZER_SEARCH_SPACE = {
    "Adam": [1e-4, 5e-4, 1e-3, 3e-3, 1e-2],
    "AdamW": [1e-4, 5e-4, 1e-3, 3e-3, 1e-2],
    "RMSprop": [1e-4, 5e-4, 1e-3, 3e-3, 1e-2],
    "Adagrad": [5e-4, 1e-3, 5e-3, 1e-2, 5e-2],
    "SGD_Momentum": [1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
}

TUNE_DATASETS = [
    ("NCF_SCALED", load_ratings_scaled),
    # ("NCF_PREPROCESSED", load_ratings_preprocessed),
]


def ensure_output_dir(directory):
    """Creates the target directory, falling back if '>' is disallowed by the OS."""
    try:
        os.makedirs(directory, exist_ok=True)
        return directory
    except OSError:
        fallback = directory.replace(">", "_")
        print(f"Warning: OS rejected '{directory}'. Saving plots to '{fallback}' instead.")
        os.makedirs(fallback, exist_ok=True)
        return fallback


def build_optimizer(opt_name, model_params, lr, weight_decay=WEIGHT_DECAY):
    """Instantiates the requested PyTorch optimizer."""
    if opt_name == "Adam":
        return torch.optim.Adam(model_params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "AdamW":
        return torch.optim.AdamW(model_params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "RMSprop":
        return torch.optim.RMSprop(model_params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "Adagrad":
        return torch.optim.Adagrad(model_params, lr=lr, weight_decay=weight_decay)
    elif opt_name == "SGD_Momentum":
        return torch.optim.SGD(
            model_params, lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")


# =====================================================================
# Single Trial Training (Train + Val Only)
# =====================================================================

def run_trial(dataset_label, opt_name, lr, splits, info, device, use_features=True):
    """Trains one (optimizer, lr) configuration using only train and val splits."""
    set_seed(SEED)
    train, val = splits["train"], splits["val"]
    n_train = len(train["rating"])

    best = {"val_rmse": float("inf"), "train_rmse": float("inf"), "epoch": 0}
    stale, epochs_run = 0, 0
    epoch_records = []

    with track_resources(device) as res:
        model = NCF(
            info["n_users"],
            info["n_movies"],
            n_features=info["n_features"] if use_features else 0,
            embed_dim=EMBED_DIM,
            mlp_layers=MLP_LAYERS,
            dropout=DROPOUT,
            global_mean=info["global_mean"],
        ).to(device)

        optimizer = build_optimizer(opt_name, model.parameters(), lr=lr)
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
                f"  [{opt_name} | lr={lr:g}] Epoch {epoch:02d}/{MAX_EPOCHS:02d} | "
                f"train RMSE {train_rmse:.4f} | val RMSE {val_rmse:.4f}"
            )

            epoch_records.append({
                "dataset": dataset_label,
                "optimizer": opt_name,
                "lr": lr,
                "epoch": epoch,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
            })

            if np.isnan(val_rmse) or np.isinf(val_rmse):
                print(f"  [Diverged] {opt_name} lr={lr} at epoch {epoch}")
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
        "optimizer": opt_name,
        "lr": lr,
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
# Visualization (Validation-Only)
# =====================================================================

def generate_comparison_plots(history_df, out_dir):
    """Generates and saves 3 diagnostic charts comparing optimizers and LRs on val."""
    plt.style.use("seaborn-v0_8-whitegrid")

    for dataset_label, df_ds in history_df.groupby("dataset"):
        trials_df = (
            df_ds.drop_duplicates(subset=["optimizer", "lr"])
            .sort_values(["optimizer", "lr"])
            .copy()
        )

        best_per_opt = (
            trials_df[np.isfinite(trials_df["best_val_rmse"])]
            .loc[lambda d: d.groupby("optimizer")["best_val_rmse"].idxmin()]
            .sort_values("best_val_rmse")
        )

        # -------------------------------------------------------------
        # Plot 1: Learning Rate Sensitivity Curves (Best Val RMSE vs LR)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for opt_name, grp in trials_df.groupby("optimizer"):
            valid = grp[np.isfinite(grp["best_val_rmse"])]
            if valid.empty:
                continue
            line, = ax.plot(
                valid["lr"], valid["best_val_rmse"], marker="o", linewidth=2, label=opt_name
            )
            best_row = valid.loc[valid["best_val_rmse"].idxmin()]
            ax.scatter(
                [best_row["lr"]],
                [best_row["best_val_rmse"]],
                color=line.get_color(),
                s=160,
                marker="*",
                zorder=5,
                edgecolors="black",
                linewidths=0.8,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Learning Rate (log scale)", fontsize=11)
        ax.set_ylabel("Best Validation RMSE (lower is better)", fontsize=11)
        ax.set_title(
            f"Optimizer Learning Rate Sensitivity ({dataset_label})\n"
            "Stars indicate optimal LR per optimizer",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(title="Optimizer", frameon=True)
        plt.tight_layout()
        p1 = os.path.join(out_dir, f"{dataset_label.lower()}_lr_sensitivity.png")
        plt.savefig(p1, dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # Plot 2: Validation Convergence Curves for Each Optimizer's Best LR
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for _, best_row in best_per_opt.iterrows():
            opt_name = best_row["optimizer"]
            best_lr = best_row["lr"]
            curve = df_ds[(df_ds["optimizer"] == opt_name) & (df_ds["lr"] == best_lr)]
            ax.plot(
                curve["epoch"],
                curve["val_rmse"],
                marker="o",
                markersize=4,
                linewidth=2,
                label=f"{opt_name} (lr={best_lr:g}, best={best_row['best_val_rmse']:.4f})",
            )

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Validation RMSE", fontsize=11)
        ax.set_title(
            f"Validation Convergence at Optimal Learning Rate ({dataset_label})",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(title="Best Configuration", frameon=True)
        plt.tight_layout()
        p2 = os.path.join(out_dir, f"{dataset_label.lower()}_best_convergence.png")
        plt.savefig(p2, dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # Plot 3: Best Train vs. Val RMSE & Training Time Comparison
        # -------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        labels = [
            f"{r['optimizer']}\n(lr={r['lr']:g})" for _, r in best_per_opt.iterrows()
        ]
        x = np.arange(len(labels))
        width = 0.36

        rects1 = ax1.bar(
            x - width / 2, best_per_opt["best_train_rmse"], width, label="Train RMSE (at best val)", color="#4C72B0"
        )
        rects2 = ax1.bar(
            x + width / 2, best_per_opt["best_val_rmse"], width, label="Best Val RMSE", color="#DD8452"
        )

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9.5)
        ax1.set_ylabel("RMSE", fontsize=11)
        ax1.set_title("Train vs. Validation RMSE (Generalization Gap)", fontsize=11, fontweight="bold")
        min_rmse = min(best_per_opt["best_train_rmse"].min(), best_per_opt["best_val_rmse"].min())
        max_rmse = max(best_per_opt["best_train_rmse"].max(), best_per_opt["best_val_rmse"].max())
        ax1.set_ylim(max(0, min_rmse - 0.03), max_rmse + 0.03)
        ax1.legend()

        for rect in list(rects1) + list(rects2):
            h = rect.get_height()
            ax1.annotate(
                f"{h:.4f}",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        bars_time = ax2.bar(x, best_per_opt["train_time_s"], width=0.5, color="#55A868")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=9.5)
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
            f"Top Optimizer Configurations Summary ({dataset_label})",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        p3 = os.path.join(out_dir, f"{dataset_label.lower()}_optimizer_summary.png")
        plt.savefig(p3, dpi=300)
        plt.close(fig)

        print(f"Saved plots for {dataset_label} in '{out_dir}/'")


# =====================================================================
# Main Tuning Loop
# =====================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    out_dir = ensure_output_dir(PLOT_DIR)
    all_epoch_history = []
    all_trial_summaries = []

    for dataset_label, loader in TUNE_DATASETS:
        print(f"\n{'=' * 65}\n=== TUNING ON DATASET: {dataset_label} ===\n{'=' * 65}")
        df = loader()
        splits, meta, info = prepare_data(df, device)
        # Drop test tensor and meta immediately so it cannot be touched during tuning
        del df, meta, splits["test"]
        gc.collect()

        for opt_name, lr_grid in OPTIMIZER_SEARCH_SPACE.items():
            print(f"\n--- Optimizer: {opt_name} ---")
            for lr in lr_grid:
                summary, epoch_records = run_trial(
                    dataset_label, opt_name, lr, splits, info, device, use_features=True
                )
                all_trial_summaries.append(summary)
                all_epoch_history.extend(epoch_records)

                print(
                    f"LR={lr:<8g} | Best Epoch={summary['best_epoch']:02d} | "
                    f"Train RMSE={summary['best_train_rmse']:.4f} | "
                    f"Val RMSE={summary['best_val_rmse']:.4f} | "
                    f"Time={summary['train_time_s']:.1f}s"
                )

        del splits
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    history_df = pd.DataFrame(all_epoch_history)
    history_df.to_csv(HISTORY_CSV, index=False)
    print(f"\nSaved full epoch-by-epoch tuning history to '{HISTORY_CSV}'")

    summary_df = pd.DataFrame(all_trial_summaries)
    best_leaderboard = (
        summary_df[np.isfinite(summary_df["best_val_rmse"])]
        .loc[lambda d: d.groupby(["dataset", "optimizer"])["best_val_rmse"].idxmin()]
        .sort_values(["dataset", "best_val_rmse"])
    )
    print("\n=== BEST CONFIGURATION PER OPTIMIZER (VALIDATION ONLY) ===")
    print(best_leaderboard.to_string(index=False))

    generate_comparison_plots(history_df, out_dir)


if __name__ == "__main__":
    main()