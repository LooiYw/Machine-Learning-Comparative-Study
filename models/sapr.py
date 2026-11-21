"""
SAPR (item-side cascaded autoencoder) for rating prediction on MovieLens 1M.

Reads data/processed/ratings_*.parquet (built by preprocessing.py) through the same
shared pipeline as ncf.py: trains on the 'train' rows, picks the best epoch on the
'val' rows (early stopping on RMSE), then evaluates train / val / test with
utils.evaluate_nn and appends the rows to results/model_evaluation_results.csv
via utils.save_results.

Model: movies are the samples. AE1 takes [the movie's train rating column, the movie's
attributes (release_year + genres)] and reconstructs both; its reconstructed ratings are
fed into AE2, which reconstructs the ratings again. The prediction for (user, movie) is
AE2's output for that movie at that user's position.
Loss = MSE(AE2 ratings) + FIRST_LOSS_WEIGHT * MSE(AE1 ratings) + SIDE_LOSS_WEIGHT * MSE(attributes),
rating losses computed on observed train entries only.

The rating matrix is built from TRAIN rows only. Movie attributes are static metadata
(no rating information), so they are taken from every row. movie_popularity, user
features and the leaky user_avg / movie_avg are not used.

The training loop below mirrors ncf.train_model (same seed, optimizer, early stopping on
val RMSE, time / memory tracking). It is separate because SAPR trains on movies (not
rating rows) and has three loss terms, which the shared trainer has no hook for.

Requires: torch, numpy, pandas, pyarrow, scikit-learn, psutil
"""
import copy
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ncf import (LEARNING_RATE, MAX_EPOCHS, PATIENCE, WEIGHT_DECAY, NCFPredictor,
                 evaluate_all_splits, prepare_data, set_seed, split_user_item_features)
from preprocessing import load_ratings_scaled, load_ratings_preprocessed
from utils import save_results, track_resources

HIDDEN_DIM = 128
FIRST_LOSS_WEIGHT = 0.5
SIDE_LOSS_WEIGHT = 0.1
ITEM_BATCH_SIZE = 128          # movies per step (the autoencoder's samples are movies)

# Label is split on the first '_' by save_results: Algorithm 'SAPR', Dataset 'preprocessed'/'scaled'.
DATASETS = [
    ('SAPR_PREPROCESSED', load_ratings_preprocessed),
    ('SAPR_SCALED', load_ratings_scaled),
]


def item_attribute_columns(feature_cols):
    """Movie-side attributes only: release_year + genres (popularity is a rating-derived count)."""
    _, item_idx = split_user_item_features(feature_cols)
    cols = [i for i in item_idx if feature_cols[i] != 'movie_popularity']
    if not cols:
        raise ValueError("No movie attribute columns found in the feature matrix")
    return cols


def build_item_tensors(splits, info, attr_cols, device):
    """Item x user train rating matrix (scaled to [0, 1]) and item x attribute matrix."""
    train = splits['train']
    ratings = torch.zeros(info['n_movies'], info['n_users'], device=device)
    ratings[train['movie'], train['user']] = train['rating'] / 5.0

    attr_idx = torch.tensor(attr_cols, dtype=torch.long, device=device)
    side = torch.zeros(info['n_movies'], len(attr_cols), device=device)
    for part in splits.values():       # static metadata: identical for every row of a movie
        side[part['movie']] = part['features'][:, attr_idx]
    return ratings, side


class SAPR(nn.Module):
    """Cascaded item autoencoders: AE1 reconstructs [ratings, attributes], AE2 reconstructs ratings."""

    def __init__(self, item_ratings, side, hidden_dim=HIDDEN_DIM):
        super().__init__()
        # persistent=False keeps the big matrices out of state_dict (small best-epoch snapshots)
        self.register_buffer('item_ratings', item_ratings, persistent=False)
        self.register_buffer('side', side, persistent=False)
        self.n_users = item_ratings.shape[1]
        total_dim = self.n_users + side.shape[1]
        self.ae1 = nn.Sequential(nn.Linear(total_dim, hidden_dim), nn.Tanh(),
                                 nn.Linear(hidden_dim, total_dim))
        self.ae2 = nn.Sequential(nn.Linear(self.n_users, hidden_dim), nn.Tanh(),
                                 nn.Linear(hidden_dim, self.n_users))

    def forward_items(self, items):
        combined = torch.cat([self.item_ratings[items], self.side[items]], dim=1)
        decoded = self.ae1(combined)
        first = 1.0 + 4.0 * torch.sigmoid(decoded[:, :self.n_users])
        # AE1's reconstruction goes straight into AE2, so gradients flow through both
        final = 1.0 + 4.0 * torch.sigmoid(self.ae2(first / 5.0))
        return first, decoded[:, self.n_users:], final

    def forward(self, user, movie, features=None):
        # rating for each (user, movie) pair; this is what NCFPredictor / utils.evaluate_nn call.
        # each distinct movie in the batch is encoded once.
        uniq, inv = torch.unique(movie, return_inverse=True)
        final = self.forward_items(uniq)[2]
        return final[inv, user]


def train_sapr(build_model, splits, device, n_attributes):
    """
    Same recipe as ncf.train_model (early stopping on val RMSE, time and memory
    tracked around everything training needs), but batches are movies and the
    loss has three terms. Returns (best model, stats dict, training-params dict).
    """
    set_seed()
    train, val = splits['train'], splits['val']
    warm_items = torch.unique(train['movie'])          # movies with >= 1 train rating
    n_warm = len(warm_items)

    best = {'rmse': float('inf'), 'state': None, 'epoch': 0}
    stale, epochs_run = 0, 0

    with track_resources(device) as res:
        model = build_model().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                     weight_decay=WEIGHT_DECAY)
        val_pred_model = NCFPredictor(model, val, use_features=False)

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            perm = warm_items[torch.randperm(n_warm, device=device)]
            sq_err, n_obs = 0.0, 0

            for start in range(0, n_warm, ITEM_BATCH_SIZE):
                items = perm[start:start + ITEM_BATCH_SIZE]
                truth = model.item_ratings[items] * 5.0
                observed = truth > 0              # missing entries are not labels
                first, attributes, final = model.forward_items(items)

                rating_loss = F.mse_loss(final[observed], truth[observed])
                first_loss = F.mse_loss(first[observed], truth[observed])
                side_loss = F.mse_loss(attributes, model.side[items])
                loss = (rating_loss + FIRST_LOSS_WEIGHT * first_loss
                        + SIDE_LOSS_WEIGHT * side_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                n_batch = int(observed.sum())
                sq_err += rating_loss.item() * n_batch
                n_obs += n_batch

            epochs_run = epoch
            train_rmse = (sq_err / n_obs) ** 0.5
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
        'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY, 'batch_size': ITEM_BATCH_SIZE,
        'best_epoch': best['epoch'], 'epochs_run': epochs_run,
        'n_parameters': n_params, 'use_side_features': True,
        'n_item_attributes': n_attributes,
    }
    return model, stats, params


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model_params = {'hidden_dim': HIDDEN_DIM, 'first_loss_weight': FIRST_LOSS_WEIGHT,
                    'side_loss_weight': SIDE_LOSS_WEIGHT}

    for label, loader in DATASETS:
        print(f"\n{'=' * 60}\n=== MODEL: {label} ===\n{'=' * 60}")
        df = loader()
        splits, meta, info = prepare_data(df, device)
        del df
        gc.collect()
        attr_cols = item_attribute_columns(info['feature_cols'])
        print(f"Item attributes used: {len(attr_cols)}")

        # built inside train_sapr's resource tracking, so the matrices' memory is counted
        def build():
            ratings, side = build_item_tensors(splits, info, attr_cols, device)
            return SAPR(ratings, side)

        model, stats, train_params = train_sapr(build, splits, device, len(attr_cols))
        rows = evaluate_all_splits(
            label, lambda s: NCFPredictor(model, splits[s], False),
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