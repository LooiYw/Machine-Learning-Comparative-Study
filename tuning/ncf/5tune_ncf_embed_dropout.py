"""
Sequential Hyperparameter Tuning for NeuMF:
Stage 1: Embedding Dimension (16, 32, 50, 64, 128)
Stage 2: Dropout Rate (0.0, 0.1, 0.2, 0.3, 0.4, 0.5) using the Stage 1 winning embed_dim.

Locks in the winning configuration so far:
- Optimizer : RMSprop (lr=0.001)
- Activation: PReLU
- MLP Layers: (512, 256, 128, 64)
- Batch Size: 1024

Evaluates strictly on train and validation splits (no test split leakage).

Outputs (saved in the same directory as this .py file):
- ncf_embed_dim_tune_history.csv       : Stage 1 epoch & trial history.
- ncf_scaled_embed_dim_convergence.png : Stage 1 validation convergence plot.
- ncf_scaled_embed_dim_summary.png     : Stage 1 generalization gap & time bar chart.
- ncf_dropout_tune_history.csv         : Stage 2 epoch & trial history.
- ncf_scaled_dropout_convergence.png   : Stage 2 validation convergence plot.
- ncf_scaled_dropout_summary.png       : Stage 2 generalization gap & time bar chart.

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
# Tuning Configuration (Locked Winners + Search Grids)
# =====================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMBED_HISTORY_CSV = os.path.join(SCRIPT_DIR, "ncf_embed_dim_tune_history.csv")
DROPOUT_HISTORY_CSV = os.path.join(SCRIPT_DIR, "ncf_dropout_tune_history.csv")

WINNING_OPTIMIZER = "RMSprop"
WINNING_LR = 0.001
WINNING_ACTIVATION = "PReLU"
WINNING_MLP_LAYERS = (512, 256, 128, 64)

# Stage 1: Embedding dimensions to compare (using baseline DROPOUT=0.2)
EMBED_DIM_CANDIDATES = [16, 32, 50, 64, 128]

# Stage 2: Dropout rates to compare (0.0 = no dropout, up to 0.5)
DROPOUT_CANDIDATES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

TUNE_DATASETS = [
    ("NCF_SCALED", load_ratings_scaled),
    # ("NCF_PREPROCESSED", load_ratings_preprocessed),
]


def build_ncf_model(info, embed_dim, dropout, use_features=True):
    """Builds NeuMF with (512, 256, 128, 64) MLP, PReLU, and custom embed_dim/dropout."""
    model = NCF(
        info["n_users"],
        info["n_movies"],
        n_features=info["n_features"] if use_features else 0,
        embed_dim=embed_dim,
        mlp_layers=WINNING_MLP_LAYERS,
        dropout=dropout,
        global_mean=info["global_mean"],
    )
    for idx, layer in enumerate(model.mlp):
        if isinstance(layer, nn.ReLU):
            model.mlp[idx] = nn.PReLU()
    return model


def build_rmsprop_optimizer(model, lr=WINNING_LR, weight_decay=WEIGHT_DECAY):
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
# Generic Single Trial Runner (Train + Val Only)
# =====================================================================

def run_trial(
    dataset_label,
    stage_name,
    embed_dim,
    dropout,
    splits,
    info,
    device,
    use_features=True,
    seed=SEED,
):
    """Trains one (embed_dim, dropout) configuration on train and validation splits."""
    set_seed(seed)
    train, val = splits["train"], splits["val"]
    n_train = len(train["rating"])

    best = {"val_rmse": float("inf"), "train_rmse": float("inf"), "epoch": 0}
    stale, epochs_run = 0, 0
    epoch_records = []

    with track_resources(device) as res:
        model = build_ncf_model(
            info, embed_dim=embed_dim, dropout=dropout, use_features=use_features
        ).to(device)
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
                f"  [{stage_name} | emb={embed_dim} | drop={dropout:g}] "
                f"Epoch {epoch:02d}/{MAX_EPOCHS:02d} | train RMSE {train_rmse:.4f} | val RMSE {val_rmse:.4f}"
            )

            epoch_records.append({
                "dataset": dataset_label,
                "stage": stage_name,
                "optimizer": WINNING_OPTIMIZER,
                "lr": WINNING_LR,
                "activation": WINNING_ACTIVATION,
                "mlp_layers": str(WINNING_MLP_LAYERS),
                "embed_dim": embed_dim,
                "dropout": dropout,
                "n_parameters": n_params,
                "epoch": epoch,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
            })

            if np.isnan(val_rmse) or np.isinf(val_rmse):
                print(f"  [Diverged] emb={embed_dim}, drop={dropout} at epoch {epoch}")
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
        "stage": stage_name,
        "optimizer": WINNING_OPTIMIZER,
        "lr": WINNING_LR,
        "activation": WINNING_ACTIVATION,
        "mlp_layers": str(WINNING_MLP_LAYERS),
        "embed_dim": embed_dim,
        "dropout": dropout,
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
# Separate Visualization Helper for Each Stage
# =====================================================================

def generate_stage_plots(
    history_df,
    param_col,
    param_title,
    file_prefix,
    fixed_context_str,
    out_dir,
):
    """
    Generates two separate diagnostic plots for a single tuning stage:
    1. *_convergence.png : Validation RMSE across epochs per candidate value.
    2. *_summary.png     : Train vs. Val RMSE (with best horizontal line) + Training Time.
    """
    plt.style.use("seaborn-v0_8-whitegrid")

    for dataset_label, df_ds in history_df.groupby("dataset"):
        trials_df = (
            df_ds.drop_duplicates(subset=[param_col])
            .sort_values(param_col)
            .copy()
        )

        valid_trials = trials_df[np.isfinite(trials_df["best_val_rmse"])]
        winner_row = valid_trials.loc[valid_trials["best_val_rmse"].idxmin()]
        winner_val = winner_row[param_col]
        winner_rmse = winner_row["best_val_rmse"]

        # -------------------------------------------------------------
        # Plot 1: Validation Convergence Curves
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        for _, row in trials_df.iterrows():
            val = row[param_col]
            curve = df_ds[df_ds[param_col] == val]
            val_fmt = f"{int(val)}" if param_col == "embed_dim" else f"{val:g}"
            ax.plot(
                curve["epoch"],
                curve["val_rmse"],
                marker="o",
                markersize=4,
                linewidth=2,
                label=(
                    f"{param_title}={val_fmt} "
                    f"(best={row['best_val_rmse']:.4f} @ ep {int(row['best_epoch'])})"
                ),
            )

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Validation RMSE (lower is better)", fontsize=11)
        ax.set_title(
            f"{param_title} Convergence ({dataset_label})\n{fixed_context_str}",
            fontsize=11.5,
            fontweight="bold",
        )
        ax.legend(title=param_title, frameon=True, fontsize=9)
        plt.tight_layout()
        p1 = os.path.join(out_dir, f"{dataset_label.lower()}_{file_prefix}_convergence.png")
        plt.savefig(p1, dpi=300)
        plt.close(fig)

        # -------------------------------------------------------------
        # Plot 2: Generalization Gap (with Best Val Line) & Training Time
        # -------------------------------------------------------------
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

        labels = [
            f"{param_title}={int(r[param_col])}" if param_col == "embed_dim"
            else f"{param_title}={r[param_col]:g}"
            for _, r in trials_df.iterrows()
        ]
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

        winner_fmt = f"{int(winner_val)}" if param_col == "embed_dim" else f"{winner_val:g}"
        ax1.axhline(
            y=winner_rmse,
            color="#C44E52",
            linestyle="--",
            linewidth=1.6,
            zorder=4,
            label=f"Best Val ({param_title}={winner_fmt}: {winner_rmse:.4f})",
        )

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=9, rotation=12)
        ax1.set_ylabel("RMSE", fontsize=11)
        ax1.set_title("Train vs. Validation RMSE (Generalization Gap)", fontsize=11, fontweight="bold")
        min_rmse = min(trials_df["best_train_rmse"].min(), trials_df["best_val_rmse"].min())
        max_rmse = max(trials_df["best_train_rmse"].max(), trials_df["best_val_rmse"].max())
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

        bars_time = ax2.bar(x, trials_df["train_time_s"], width=0.5, color="#55A868")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=9, rotation=12)
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
            f"{param_title} Tuning Summary ({dataset_label} | {fixed_context_str})",
            fontsize=12.5,
            fontweight="bold",
        )
        plt.tight_layout()
        p2 = os.path.join(out_dir, f"{dataset_label.lower()}_{file_prefix}_summary.png")
        plt.savefig(p2, dpi=300)
        plt.close(fig)

        print(f"Saved {file_prefix} plots for {dataset_label} in '{out_dir}'")


# =====================================================================
# Main Sequential Tuning Pipeline (Stage 1: Embed Dim -> Stage 2: Dropout)
# =====================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using device: {device} | Optimizer: {WINNING_OPTIMIZER} (lr={WINNING_LR}) | "
        f"Activation: {WINNING_ACTIVATION} | MLP: {WINNING_MLP_LAYERS} | Batch: {BATCH_SIZE}"
    )

    all_embed_history = []
    all_embed_summaries = []

    all_dropout_history = []
    all_dropout_summaries = []

    for dataset_label, loader in TUNE_DATASETS:
        print(f"\n{'=' * 72}\n=== LOADING DATASET FOR SEQUENTIAL TUNING: {dataset_label} ===\n{'=' * 72}")
        df = loader()
        splits, meta, info = prepare_data(df, device)
        del df, meta, splits["test"]
        gc.collect()

        # -------------------------------------------------------------
        # STAGE 1: Embedding Dimension Tuning (with default DROPOUT=0.2)
        # -------------------------------------------------------------
        print(f"\n{'-' * 72}\n>>> STAGE 1: EMBEDDING DIMENSION TUNING (Dropout={DROPOUT}) <<<\n{'-' * 72}")
        ds_embed_summaries = []

        for emb_dim in EMBED_DIM_CANDIDATES:
            print(f"\n--- Embedding Dimension: {emb_dim} ---")
            summary, epoch_records = run_trial(
                dataset_label=dataset_label,
                stage_name="embed_dim",
                embed_dim=emb_dim,
                dropout=DROPOUT,
                splits=splits,
                info=info,
                device=device,
                use_features=True,
            )
            ds_embed_summaries.append(summary)
            all_embed_summaries.append(summary)
            all_embed_history.extend(epoch_records)

            print(
                f"EmbedDim={emb_dim:<4} | Params={summary['n_parameters']:,} | "
                f"Best Ep={summary['best_epoch']:02d} | Train={summary['best_train_rmse']:.4f} | "
                f"Val={summary['best_val_rmse']:.4f} | Gap={summary['gen_gap']:.4f} | "
                f"Time={summary['train_time_s']:.1f}s"
            )

        # Pick winning embedding dimension (lowest best_val_rmse) for Stage 2
        ds_embed_df = pd.DataFrame(ds_embed_summaries)
        valid_embed_df = ds_embed_df[np.isfinite(ds_embed_df["best_val_rmse"])]
        best_embed_row = valid_embed_df.loc[valid_embed_df["best_val_rmse"].idxmin()]
        winning_embed_dim = int(best_embed_row["embed_dim"])

        print(
            f"\n*** STAGE 1 WINNER ({dataset_label}): embed_dim = {winning_embed_dim} "
            f"(Val RMSE = {best_embed_row['best_val_rmse']:.4f}) ***"
        )

        # -------------------------------------------------------------
        # STAGE 2: Dropout Rate Tuning (using winning_embed_dim)
        # -------------------------------------------------------------
        print(
            f"\n{'-' * 72}\n"
            f">>> STAGE 2: DROPOUT RATE TUNING (Using Winning embed_dim={winning_embed_dim}) <<<\n"
            f"{'-' * 72}"
        )

        for drop_rate in DROPOUT_CANDIDATES:
            print(f"\n--- Dropout Rate: {drop_rate:g} (embed_dim={winning_embed_dim}) ---")
            summary, epoch_records = run_trial(
                dataset_label=dataset_label,
                stage_name="dropout",
                embed_dim=winning_embed_dim,
                dropout=drop_rate,
                splits=splits,
                info=info,
                device=device,
                use_features=True,
            )
            all_dropout_summaries.append(summary)
            all_dropout_history.extend(epoch_records)

            print(
                f"Dropout={drop_rate:<5g} | EmbedDim={winning_embed_dim} | "
                f"Best Ep={summary['best_epoch']:02d} | Train={summary['best_train_rmse']:.4f} | "
                f"Val={summary['best_val_rmse']:.4f} | Gap={summary['gen_gap']:.4f} | "
                f"Time={summary['train_time_s']:.1f}s"
            )

        del splits
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Save Stage 1 CSV, Print Leaderboard & Generate Stage 1 Plots
    # -----------------------------------------------------------------
    embed_history_df = pd.DataFrame(all_embed_history)
    embed_history_df.to_csv(EMBED_HISTORY_CSV, index=False)
    print(f"\nSaved Stage 1 embedding history to '{EMBED_HISTORY_CSV}'")

    embed_summary_df = pd.DataFrame(all_embed_summaries).sort_values(["dataset", "best_val_rmse"])
    print("\n=== STAGE 1: EMBEDDING DIMENSION LEADERBOARD (VALIDATION ONLY) ===")
    print(embed_summary_df.to_string(index=False))

    generate_stage_plots(
        history_df=embed_history_df,
        param_col="embed_dim",
        param_title="Embed Dim",
        file_prefix="embed_dim",
        fixed_context_str=f"{WINNING_OPTIMIZER} (lr={WINNING_LR}) | {WINNING_ACTIVATION} | MLP {WINNING_MLP_LAYERS} | Dropout={DROPOUT}",
        out_dir=SCRIPT_DIR,
    )

    # -----------------------------------------------------------------
    # Save Stage 2 CSV, Print Leaderboard & Generate Stage 2 Plots
    # -----------------------------------------------------------------
    dropout_history_df = pd.DataFrame(all_dropout_history)
    dropout_history_df.to_csv(DROPOUT_HISTORY_CSV, index=False)
    print(f"\nSaved Stage 2 dropout history to '{DROPOUT_HISTORY_CSV}'")

    dropout_summary_df = pd.DataFrame(all_dropout_summaries).sort_values(["dataset", "best_val_rmse"])
    print("\n=== STAGE 2: DROPOUT RATE LEADERBOARD (VALIDATION ONLY) ===")
    print(dropout_summary_df.to_string(index=False))

    used_embed_dim = int(dropout_summary_df.iloc[0]["embed_dim"])
    generate_stage_plots(
        history_df=dropout_history_df,
        param_col="dropout",
        param_title="Dropout",
        file_prefix="dropout",
        fixed_context_str=f"{WINNING_OPTIMIZER} (lr={WINNING_LR}) | {WINNING_ACTIVATION} | MLP {WINNING_MLP_LAYERS} | EmbedDim={used_embed_dim}",
        out_dir=SCRIPT_DIR,
    )


if __name__ == "__main__":
    main()