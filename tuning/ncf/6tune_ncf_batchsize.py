"""
Hyperparameter Tuning for NeuMF (Batch Size + Learning Rate).

Locks in the winning configuration so far:
- Optimizer : RMSprop
- Activation: PReLU
- MLP Layers: (256, 128, 64, 32)

Compares different batch sizes while independently reassessing the learning rate
for each batch size (without assuming linear LR scaling). Evaluates strictly on
train and validation splits (no test split leakage).

Outputs (saved in the same directory as this .py file):
- ncf_batch_lr_tune_history.csv : Full trial & epoch-level tuning history.
- *.png                         : Comparison plots across batch sizes and LRs.

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
# Tuning Configuration (Locked Winners + Search Grid)
# =====================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_CSV = os.path.join(SCRIPT_DIR, "ncf_batch_lr_tune_history.csv")

WINNING_OPTIMIZER = "RMSprop"
WINNING_ACTIVATION = "PReLU"
WINNING_MLP_LAYERS = (256, 128, 64, 32)
BASELINE_LR = 0.001

# Candidate batch sizes spanning small/noisy to large/smooth gradient estimates
BATCH_SIZES = [256, 512, 1024, 2048, 4096]

# Reassess LR independently around 0.001 without assuming linear scaling
LR_GRID = [2e-4, 5e-4, 1e-3, 2e-3, 3e-3]

TUNE_DATASETS = [
    ("NCF_SCALED", load_ratings_scaled),
    # ("NCF_PREPROCESSED", load_ratings_preprocessed),
]


def build_winning_ncf(info, use_features=True):
    """Builds NeuMF with the winning (256, 128, 64, 32) MLP and PReLU activations."""
    model = NCF(
        info["n_users"],
        info["n_movies"],
        n_features=info["n_features"] if use_features else 0,
        embed_dim=EMBED_DIM,
        mlp_layers=WINNING_MLP_LAYERS,
        dropout=DROPOUT,
        global_mean=info["global_mean"],
    )
    for idx, layer in enumerate(model.mlp):
        if isinstance(layer, nn.ReLU):
            model.mlp[idx] = nn.PReLU()
    return model


def build_rmsprop_optimizer(model, lr, weight_decay=WEIGHT_DECAY):
    """Builds RMSprop while exempting PReLU slope weights from L2 decay."""
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

def run_trial(dataset_label, batch_size, lr, splits, info, device, use_features=True):
    """Trains one (batch_size, lr) configuration on train and validation splits."""
    set_seed(SEED)
    train, val = splits["train"], splits["val"]
    n_train = len(train["rating"])
    steps_per_epoch = int(np.ceil(n_train / batch_size))

    best = {"val_rmse": float("inf"), "train_rmse": float("inf"), "epoch": 0}
    stale, epochs_run = 0, 0
    epoch_records = []

    with track_resources(device) as res:
        model = build_winning_ncf(info, use_features=use_features).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        optimizer = build_rmsprop_optimizer(model, lr=lr, weight_decay=WEIGHT_DECAY)
        loss_fn = nn.MSELoss()
        val_pred_model = NCFPredictor(model, val, use_features)

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            perm = torch.randperm(n_train, device=device)
            running_loss = 0.0

            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
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
                f"  [BS={batch_size:<4} | lr={lr:g}] Epoch {epoch:02d}/{MAX_EPOCHS:02d} | "
                f"train RMSE {train_rmse:.4f} | val RMSE {val_rmse:.4f}"
            )

            epoch_records.append({
                "dataset": dataset_label,
                "optimizer": WINNING_OPTIMIZER,
                "activation": WINNING_ACTIVATION,
                "mlp_layers": str(WINNING_MLP_LAYERS),
                "batch_size": batch_size,
                "steps_per_epoch": steps_per_epoch,
                "lr": lr,
                "n_parameters": n_params,
                "epoch": epoch,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
            })

            if np.isnan(val_rmse) or np.isinf(val_rmse):
                print(f"  [Diverged] BS={batch_size} lr={lr} at epoch {epoch}")
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
        "optimizer": WINNING_OPTIMIZER,
        "activation": WINNING_ACTIVATION,
        "mlp_layers": str(WINNING_MLP_LAYERS),
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "lr": lr,
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
# Visualization (Saved Next to Script)
# =====================================================================

def generate_batch_lr_plots(history_df, out_dir):
    """Generates and saves 3 diagnostic charts comparing batch sizes and LRs."""
    plt.style.use("seaborn-v0_8-whitegrid")

    for dataset_label, df_ds in history_df.groupby("dataset"):
        trials_df = (
            df_ds.drop_duplicates(subset=["batch_size", "lr"])
            .sort_values(["batch_size", "lr"])
            .copy()
        )

        # Best LR per batch size
        best_per_bs = (
            trials_df[np.isfinite(trials_df["best_val_rmse"])]
            .loc[lambda d: d.groupby("batch_size")["best_val_rmse"].idxmin()]
            .sort_values("batch_size")
        )

        winner_row = best_per_bs.loc[best_per_bs["best_val_rmse"].idxmin()]
        winner_label = f"BS={int(winner_row['batch_size'])}, lr={winner_row['lr']:g}"
        winner_val_rmse = winner_row["best_val_rmse"]

        # -------------------------------------------------------------
        # Plot 1: LR Sensitivity Curves per Batch Size
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        for bs, grp in trials_df.groupby("batch_size"):
            valid = grp[np.isfinite(grp["best_val_rmse"])]
            if valid.empty:
                continue
            line, = ax.plot(
                valid["lr"],
                valid["best_val_rmse"],
                marker="o",
                linewidth=2,
                label=f"Batch Size = {int(bs)}",
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
            f"Batch Size × Learning Rate Sensitivity ({dataset_label})\n"
            f"{WINNING_OPTIMIZER} | {WINNING_ACTIVATION} | MLP {WINNING_MLP_LAYERS} (Stars = optimal LR)",
            fontsize=11.5,
            fontweight="bold",
        )
        ax.legend(title="Batch Size", frameon=True)
        plt.tight_layout()
        p1 = os.path.join(out_dir, f"{dataset_label.lower()}_batch_lr_sensitivity.png")
        plt.savefig(p1, dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # Plot 2: Validation Convergence at Optimal LR per Batch Size
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        for _, best_row in best_per_bs.iterrows():
            bs = int(best_row["batch_size"])
            best_lr = best_row["lr"]
            curve = df_ds[(df_ds["batch_size"] == bs) & (df_ds["lr"] == best_lr)]
            ax.plot(
                curve["epoch"],
                curve["val_rmse"],
                marker="o",
                markersize=4,
                linewidth=2,
                label=f"BS={bs} (lr={best_lr:g}, best={best_row['best_val_rmse']:.4f})",
            )

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Validation RMSE", fontsize=11)
        ax.set_title(
            f"Validation Convergence by Batch Size at Optimal LR ({dataset_label})",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(title="Best Configuration", frameon=True)
        plt.tight_layout()
        p2 = os.path.join(out_dir, f"{dataset_label.lower()}_batch_best_convergence.png")
        plt.savefig(p2, dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # Plot 3: Generalization Gap (with Best Val Line) & Training Time
        # -------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

        labels = [
            f"BS={int(r['batch_size'])}\n(lr={r['lr']:g})" for _, r in best_per_bs.iterrows()
        ]
        x = np.arange(len(labels))
        width = 0.36

        rects1 = ax1.bar(
            x - width / 2,
            best_per_bs["best_train_rmse"],
            width,
            label="Train RMSE (at best val)",
            color="#4C72B0",
        )
        rects2 = ax1.bar(
            x + width / 2,
            best_per_bs["best_val_rmse"],
            width,
            label="Best Val RMSE",
            color="#DD8452",
        )

        # Horizontal reference line marking the best overall (Batch Size, LR) configuration
        ax1.axhline(
            y=winner_val_rmse,
            color="#C44E52",
            linestyle="--",
            linewidth=1.6,
            zorder=4,
            label=f"Best Val ({winner_label}: {winner_val_rmse:.4f})",
        )

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9.5)
        ax1.set_ylabel("RMSE", fontsize=11)
        ax1.set_title("Train vs. Validation RMSE (Generalization Gap)", fontsize=11, fontweight="bold")
        min_rmse = min(best_per_bs["best_train_rmse"].min(), best_per_bs["best_val_rmse"].min())
        max_rmse = max(best_per_bs["best_train_rmse"].max(), best_per_bs["best_val_rmse"].max())
        ax1.set_ylim(max(0, min_rmse - 0.02), max_rmse + 0.028)
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

        bars_time = ax2.bar(x, best_per_bs["train_time_s"], width=0.5, color="#55A868")
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
            f"Batch Size & LR Summary ({dataset_label} | {WINNING_OPTIMIZER}, {WINNING_ACTIVATION}, {WINNING_MLP_LAYERS})",
            fontsize=12.5,
            fontweight="bold",
        )
        plt.tight_layout()
        p3 = os.path.join(out_dir, f"{dataset_label.lower()}_batch_summary.png")
        plt.savefig(p3, dpi=300)
        plt.close(fig)

        print(f"Saved batch size & LR plots for {dataset_label} in '{out_dir}'")


# =====================================================================
# Main Tuning Loop
# =====================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using device: {device} | Optimizer: {WINNING_OPTIMIZER} | "
        f"Activation: {WINNING_ACTIVATION} | MLP: {WINNING_MLP_LAYERS}"
    )

    all_epoch_history = []
    all_trial_summaries = []

    for dataset_label, loader in TUNE_DATASETS:
        print(f"\n{'=' * 68}\n=== BATCH SIZE + LR TUNING ON DATASET: {dataset_label} ===\n{'=' * 68}")
        df = loader()
        splits, meta, info = prepare_data(df, device)
        del df, meta, splits["test"]
        gc.collect()

        for batch_size in BATCH_SIZES:
            print(f"\n--- Batch Size: {batch_size} ---")
            for lr in LR_GRID:
                summary, epoch_records = run_trial(
                    dataset_label, batch_size, lr, splits, info, device, use_features=True
                )
                all_trial_summaries.append(summary)
                all_epoch_history.extend(epoch_records)

                print(
                    f"BS={batch_size:<5} | LR={lr:<7g} | Best Ep={summary['best_epoch']:02d} | "
                    f"Train={summary['best_train_rmse']:.4f} | Val={summary['best_val_rmse']:.4f} | "
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
        .loc[lambda d: d.groupby(["dataset", "batch_size"])["best_val_rmse"].idxmin()]
        .sort_values(["dataset", "best_val_rmse"])
    )
    print("\n=== BEST LR PER BATCH SIZE (VALIDATION ONLY) ===")
    print(best_leaderboard.to_string(index=False))

    generate_batch_lr_plots(history_df, SCRIPT_DIR)


if __name__ == "__main__":
    main()