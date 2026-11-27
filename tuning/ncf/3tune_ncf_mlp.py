"""
Hyperparameter Tuning for NeuMF (MLP Structure [Layers & Neurons] + Learning Rate).

Experiments with different MLP depths and hidden-layer widths while reassessing
the learning rate with RMSprop to handle optimization instability in deeper or
wider architectures. Uses ONLY train and validation splits (no test leakage).

Outputs (saved in the same directory as this .py file):
- ncf_mlp_tune_history.csv : Full trial & epoch-level tuning history.
- *.png                    : Comparison plots across MLP architectures and LRs.

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
HISTORY_CSV = os.path.join(SCRIPT_DIR, "ncf_mlp_tune_history.csv")

WINNING_OPTIMIZER = "RMSprop"
# Set this to the winning activation from your activation tuning run
WINNING_ACTIVATION = "PReLU"

ACTIVATION_MAP = {
    "ReLU": lambda: nn.ReLU(),
    "LeakyReLU": lambda: nn.LeakyReLU(negative_slope=0.01),
    "PReLU": lambda: nn.PReLU(),
    "GELU": lambda: nn.GELU(),
    "ELU": lambda: nn.ELU(alpha=1.0),
    "SiLU": lambda: nn.SiLU(),
    "Mish": lambda: nn.Mish(),
}

# Candidate MLP architectures spanning shallow/narrow to deep/wide towers
MLP_STRUCTURES = {
    "Shallow_32_16": (32, 16),
    "Base_64_32_16": (64, 32, 16),
    "Wide_128_64_32": (128, 64, 32),
    "Deep_128_64_32_16": (128, 64, 32, 16),
    "DeepWide_256_128_64_32": (256, 128, 64, 32),
}

# Reassess LR around 5e-4 (deeper/wider towers often prefer smaller LRs)
LR_GRID = [1e-4, 3e-4, 5e-4, 1e-3]

TUNE_DATASETS = [
    ("NCF_SCALED", load_ratings_scaled),
    # ("NCF_PREPROCESSED", load_ratings_preprocessed),
]


def build_ncf_with_mlp(info, mlp_layers, act_name=WINNING_ACTIVATION, use_features=True):
    """Instantiates NCF with custom MLP layer widths/depths and winning activation."""
    model = NCF(
        info["n_users"],
        info["n_movies"],
        n_features=info["n_features"] if use_features else 0,
        embed_dim=EMBED_DIM,
        mlp_layers=mlp_layers,
        dropout=DROPOUT,
        global_mean=info["global_mean"],
    )
    act_factory = ACTIVATION_MAP[act_name]
    for idx, layer in enumerate(model.mlp):
        if isinstance(layer, nn.ReLU):
            model.mlp[idx] = act_factory()
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

def run_trial(dataset_label, arch_name, mlp_layers, lr, splits, info, device, use_features=True):
    """Trains one (mlp_layers, lr) configuration on train and validation splits."""
    set_seed(SEED)
    train, val = splits["train"], splits["val"]
    n_train = len(train["rating"])

    best = {"val_rmse": float("inf"), "train_rmse": float("inf"), "epoch": 0}
    stale, epochs_run = 0, 0
    epoch_records = []

    with track_resources(device) as res:
        model = build_ncf_with_mlp(
            info, mlp_layers, act_name=WINNING_ACTIVATION, use_features=use_features
        ).to(device)
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
                f"  [{arch_name} | lr={lr:g}] Epoch {epoch:02d}/{MAX_EPOCHS:02d} | "
                f"train RMSE {train_rmse:.4f} | val RMSE {val_rmse:.4f}"
            )

            epoch_records.append({
                "dataset": dataset_label,
                "optimizer": WINNING_OPTIMIZER,
                "activation": WINNING_ACTIVATION,
                "arch_name": arch_name,
                "mlp_layers": str(mlp_layers),
                "depth": len(mlp_layers),
                "lr": lr,
                "n_parameters": n_params,
                "epoch": epoch,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
            })

            if np.isnan(val_rmse) or np.isinf(val_rmse):
                print(f"  [Diverged] {arch_name} lr={lr} at epoch {epoch}")
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
        "arch_name": arch_name,
        "mlp_layers": str(mlp_layers),
        "depth": len(mlp_layers),
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

def generate_mlp_plots(history_df, out_dir):
    """Generates and saves 3 diagnostic charts comparing MLP structures and LRs."""
    plt.style.use("seaborn-v0_8-whitegrid")

    for dataset_label, df_ds in history_df.groupby("dataset"):
        trials_df = (
            df_ds.drop_duplicates(subset=["arch_name", "lr"])
            .sort_values(["arch_name", "lr"])
            .copy()
        )

        # Best LR per MLP structure
        best_per_arch = (
            trials_df[np.isfinite(trials_df["best_val_rmse"])]
            .loc[lambda d: d.groupby("arch_name")["best_val_rmse"].idxmin()]
            .sort_values("best_val_rmse")
        )

        winner_row = best_per_arch.iloc[0]
        winner_label = f"{winner_row['arch_name']} (lr={winner_row['lr']:g})"
        winner_val_rmse = winner_row["best_val_rmse"]

        # -------------------------------------------------------------
        # Plot 1: LR Sensitivity per MLP Architecture
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        for arch_name, grp in trials_df.groupby("arch_name"):
            valid = grp[np.isfinite(grp["best_val_rmse"])]
            if valid.empty:
                continue
            line, = ax.plot(
                valid["lr"], valid["best_val_rmse"], marker="o", linewidth=2, label=arch_name
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
            f"MLP Structure Learning Rate Sensitivity ({dataset_label})\n"
            f"Optimizer: {WINNING_OPTIMIZER} | Activation: {WINNING_ACTIVATION} (Stars = optimal LR)",
            fontsize=11.5,
            fontweight="bold",
        )
        ax.legend(title="MLP Architecture", frameon=True)
        plt.tight_layout()
        p1 = os.path.join(out_dir, f"{dataset_label.lower()}_mlp_lr_sensitivity.png")
        plt.savefig(p1, dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # Plot 2: Validation Convergence at Optimal LR per Architecture
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        for _, best_row in best_per_arch.iterrows():
            arch_name = best_row["arch_name"]
            best_lr = best_row["lr"]
            curve = df_ds[(df_ds["arch_name"] == arch_name) & (df_ds["lr"] == best_lr)]
            ax.plot(
                curve["epoch"],
                curve["val_rmse"],
                marker="o",
                markersize=4,
                linewidth=2,
                label=f"{arch_name} (lr={best_lr:g}, best={best_row['best_val_rmse']:.4f})",
            )

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Validation RMSE", fontsize=11)
        ax.set_title(
            f"Validation Convergence by MLP Structure at Optimal LR ({dataset_label})",
            fontsize=12,
            fontweight="bold",
        )
        ax.legend(title="Best Configuration", frameon=True)
        plt.tight_layout()
        p2 = os.path.join(out_dir, f"{dataset_label.lower()}_mlp_best_convergence.png")
        plt.savefig(p2, dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # Plot 3: Generalization Gap (with Best Val Line) & Training Time
        # -------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

        labels = [
            f"{r['arch_name']}\n(lr={r['lr']:g})" for _, r in best_per_arch.iterrows()
        ]
        x = np.arange(len(labels))
        width = 0.36

        rects1 = ax1.bar(
            x - width / 2,
            best_per_arch["best_train_rmse"],
            width,
            label="Train RMSE (at best val)",
            color="#4C72B0",
        )
        rects2 = ax1.bar(
            x + width / 2,
            best_per_arch["best_val_rmse"],
            width,
            label="Best Val RMSE",
            color="#DD8452",
        )

        # Horizontal reference line marking the best overall MLP configuration
        ax1.axhline(
            y=winner_val_rmse,
            color="#C44E52",
            linestyle="--",
            linewidth=1.6,
            zorder=4,
            label=f"Best Val ({winner_label}: {winner_val_rmse:.4f})",
        )

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9, rotation=15)
        ax1.set_ylabel("RMSE", fontsize=11)
        ax1.set_title("Train vs. Validation RMSE (Generalization Gap)", fontsize=11, fontweight="bold")
        min_rmse = min(best_per_arch["best_train_rmse"].min(), best_per_arch["best_val_rmse"].min())
        max_rmse = max(best_per_arch["best_train_rmse"].max(), best_per_arch["best_val_rmse"].max())
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

        bars_time = ax2.bar(x, best_per_arch["train_time_s"], width=0.5, color="#55A868")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=9, rotation=15)
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
            f"MLP Structure & LR Summary ({dataset_label} | {WINNING_OPTIMIZER}, {WINNING_ACTIVATION})",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        p3 = os.path.join(out_dir, f"{dataset_label.lower()}_mlp_summary.png")
        plt.savefig(p3, dpi=300)
        plt.close(fig)

        print(f"Saved MLP structure plots for {dataset_label} in '{out_dir}'")


# =====================================================================
# Main Tuning Loop
# =====================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using device: {device} | Optimizer: {WINNING_OPTIMIZER} | "
        f"Activation: {WINNING_ACTIVATION}"
    )

    all_epoch_history = []
    all_trial_summaries = []

    for dataset_label, loader in TUNE_DATASETS:
        print(f"\n{'=' * 65}\n=== MLP STRUCTURE TUNING ON DATASET: {dataset_label} ===\n{'=' * 65}")
        df = loader()
        splits, meta, info = prepare_data(df, device)
        del df, meta, splits["test"]
        gc.collect()

        for arch_name, mlp_layers in MLP_STRUCTURES.items():
            print(f"\n--- MLP Architecture: {arch_name} {mlp_layers} ---")
            for lr in LR_GRID:
                summary, epoch_records = run_trial(
                    dataset_label, arch_name, mlp_layers, lr, splits, info, device, use_features=True
                )
                all_trial_summaries.append(summary)
                all_epoch_history.extend(epoch_records)

                print(
                    f"Arch={arch_name:<22} | LR={lr:<7g} | Best Ep={summary['best_epoch']:02d} | "
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
        .loc[lambda d: d.groupby(["dataset", "arch_name"])["best_val_rmse"].idxmin()]
        .sort_values(["dataset", "best_val_rmse"])
    )
    print("\n=== BEST LR PER MLP STRUCTURE (VALIDATION ONLY) ===")
    print(best_leaderboard.to_string(index=False))

    generate_mlp_plots(history_df, SCRIPT_DIR)


if __name__ == "__main__":
    main()