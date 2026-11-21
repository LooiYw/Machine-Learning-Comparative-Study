"""
Multi-task learning (MTL) model for rating prediction on MovieLens 1M.

Reads data/processed/ratings_*.parquet (built by preprocessing.py) through the same
shared pipeline as ncf.py: trains on the 'train' rows, picks the best epoch on the
'val' rows (early stopping on RMSE), then evaluates train / val / test with
utils.evaluate_nn and appends the rows to results/model_evaluation_results.csv
via utils.save_results.

Model: user / movie ID embeddings + the scaled side features go through a shared MLP
with two heads:
  - main task:      rating regression, squashed into [1, 5]
  - auxiliary task: "liked" classification (rating >= 4), BCE loss
Loss = MSE(rating) + LIKE_LOSS_WEIGHT * BCE(like). Only the rating head is used at
prediction time, so evaluation goes through the same NCFPredictor as the other models.

The training loop below mirrors ncf.train_model (same seed, optimizer, batching,
early stopping on val RMSE and time / memory tracking); the only difference is the
extra auxiliary loss, which the shared trainer has no hook for.

Requires: torch, numpy, pandas, pyarrow, scikit-learn, psutil
"""
import copy
import gc
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ncf import (BATCH_SIZE, LEARNING_RATE, MAX_EPOCHS, PATIENCE, WEIGHT_DECAY,
                 NCFPredictor, evaluate_all_splits, prepare_data, set_seed)
from preprocessing import load_ratings_scaled, load_ratings_preprocessed
from utils import save_results, track_resources

HIDDEN_DIM = 128
LATENT_DIM = 32
LIKE_LOSS_WEIGHT = 0.5
LIKE_THRESHOLD = 4.0

# Label is split on the first '_' by save_results: Algorithm 'MTL', Dataset 'preprocessed'/'scaled'.
DATASETS = [
    ('MTL_PREPROCESSED', load_ratings_preprocessed),
    ('MTL_SCALED', load_ratings_scaled),
]


class MTL(nn.Module):
    """ID embeddings + side features -> shared MLP -> (rating head, like head)."""

    def __init__(self, n_users, n_movies, n_features, global_mean=3.5,
                 hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, latent_dim)
        self.movie_embedding = nn.Embedding(n_movies, latent_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.movie_embedding.weight, std=0.05)

        self.shared = nn.Sequential(
            nn.Linear(2 * latent_dim + n_features, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim), nn.ReLU())
        self.rating_head = nn.Linear(latent_dim, 1)
        self.like_head = nn.Linear(latent_dim, 1)

        # start the rating output at the train mean: 1 + 4 * sigmoid(b) = global_mean
        p = min(max((global_mean - 1.0) / 4.0, 1e-3), 1 - 1e-3)
        nn.init.constant_(self.rating_head.bias, math.log(p / (1 - p)))

    def forward_multi(self, user, movie, features):
        x = torch.cat([self.user_embedding(user), self.movie_embedding(movie), features], dim=1)
        h = self.shared(x)
        rating = 1.0 + 4.0 * torch.sigmoid(self.rating_head(h).squeeze(1))
        like_logits = self.like_head(h).squeeze(1)
        return rating, like_logits

    def forward(self, user, movie, features=None):
        # rating only: this is what NCFPredictor / utils.evaluate_nn call
        return self.forward_multi(user, movie, features)[0]


def train_mtl(build_model, splits, device):
    """
    Same recipe as ncf.train_model (early stopping on val RMSE, time and memory
    tracked around everything training needs) plus the auxiliary like-loss.
    Returns (best model, stats dict, training-params dict).
    """
    set_seed()
    train, val = splits['train'], splits['val']
    n_train = len(train['rating'])

    best = {'rmse': float('inf'), 'state': None, 'epoch': 0}
    stale, epochs_run = 0, 0

    with track_resources(device) as res:
        model = build_model().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                     weight_decay=WEIGHT_DECAY)
        val_pred_model = NCFPredictor(model, val, use_features=True)

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            perm = torch.randperm(n_train, device=device)
            running_mse = 0.0

            for start in range(0, n_train, BATCH_SIZE):
                idx = perm[start:start + BATCH_SIZE]
                truth = train['rating'][idx]
                rating, like_logits = model.forward_multi(
                    train['user'][idx], train['movie'][idx], train['features'][idx])
                mse = F.mse_loss(rating, truth)
                like = F.binary_cross_entropy_with_logits(
                    like_logits, (truth >= LIKE_THRESHOLD).float())
                loss = mse + LIKE_LOSS_WEIGHT * like

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_mse += mse.item() * len(idx)

            epochs_run = epoch
            train_rmse = (running_mse / n_train) ** 0.5
            val_p = torch.from_numpy(
                val_pred_model.predict(np.arange(len(val['rating'])))).to(device)
            val_rmse = float(torch.sqrt(((val_p - val['rating']) ** 2).mean()))
            print(f"Epoch {epoch:02d} | train RMSE {train_rmse:.4f} | val RMSE {val_rmse:.4f}")

            if val_rmse < best['rmse'] - 1e-4:
                best.update(rmse=val_rmse, state=copy.deepcopy(model.state_dict()), epoch=epoch)
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    print(f"Early stopping (no val improvement for {PATIENCE} epochs)")
                    break

    model.load_state_dict(best['state'])
    print(f"Best epoch: {best['epoch']} (val RMSE {best['rmse']:.4f})")
    print(f"Training cost: {res['time_s']:.1f}s | {res['mem_mb']:.1f} MB peak extra memory")

    stats = {'train_time_s': res['time_s'], 'train_mem_mb': res['mem_mb']}
    params = {
        'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY, 'batch_size': BATCH_SIZE,
        'best_epoch': best['epoch'], 'epochs_run': epochs_run,
        'n_parameters': n_params, 'use_side_features': True,
    }
    return model, stats, params


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model_params = {'hidden_dim': HIDDEN_DIM, 'latent_dim': LATENT_DIM,
                    'like_loss_weight': LIKE_LOSS_WEIGHT}

    for label, loader in DATASETS:
        print(f"\n{'=' * 60}\n=== MODEL: {label} ===\n{'=' * 60}")
        df = loader()
        splits, meta, info = prepare_data(df, device)
        del df
        gc.collect()

        def build():
            return MTL(info['n_users'], info['n_movies'], info['n_features'],
                       global_mean=info['global_mean'])

        model, stats, train_params = train_mtl(build, splits, device)
        rows = evaluate_all_splits(
            label, lambda s: NCFPredictor(model, splits[s], True),
            splits, meta, stats)
        save_results(rows, {**model_params, **train_params})

        # free memory before the next dataset so it doesn't skew its measurements
        del model, splits, meta
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print("\nResults appended to results/model_evaluation_results.csv")


if __name__ == "__main__":
    main()