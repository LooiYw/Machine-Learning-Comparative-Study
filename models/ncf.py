"""
Neural Collaborative Filtering (NeuMF) for rating prediction on MovieLens 1M.

Reads data/processed/ratings_*.parquet (built by preprocessing.py), trains on the
'train' rows, picks the best epoch on the 'val' rows, then evaluates train / val /
test with utils.evaluate_nn (RMSE, MAE, R2, accuracy, P/R/F1@20, time, memory) and
appends the rows to results/model_evaluation_results.csv via utils.save_results.

This file also holds the shared neural-network pipeline (data preparation, generic
trainer, predictor bridge, dataset loop) that c_nge.py and nmetricf.py reuse, so all
three models are trained and measured in exactly the same way.

Model: NeuMF with user / movie embeddings, plus scaled user / movie side
features fed into the MLP branch.

Requires: torch, numpy, pandas, pyarrow, scikit-learn, psutil
"""
import copy
import gc

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from preprocessing import load_ratings_scaled, load_ratings_preprocessed
from utils import NON_FEATURE_COLS, evaluate_nn, save_results, track_resources

SEED = 42
EMBED_DIM = 32
MLP_LAYERS = (64, 32, 16)
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 1024
MAX_EPOCHS = 100
PATIENCE = 3
RATING_MIN, RATING_MAX = 1.0, 5.0

# user_avg / movie_avg include the row's own rating on train rows (leakage)
LEAKY_COLS = ['user_avg', 'movie_avg']

# Side features that describe the USER (everything else describes the movie)
USER_FEATURE_COLS = ('is_male', 'age', 'user_activity')
USER_FEATURE_PREFIXES = ('occ_', 'zip_')

# The two datasets we compare: (name, loader function), same as knn.py.
# The label is split on the first '_' by save_results: Algorithm 'NCF', Dataset 'preprocessed'/'scaled'.
DATASETS = [
    ('NCF_PREPROCESSED', load_ratings_preprocessed),
    ('NCF_SCALED', load_ratings_scaled),
]


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# Data
# =====================================================================

def prepare_data(df, device):
    """Maps ids to contiguous indices; builds tensors and eval metadata per split."""
    user_index = {u: i for i, u in enumerate(np.sort(df['userId'].unique()))}
    movie_index = {m: i for i, m in enumerate(np.sort(df['movieId'].unique()))}

    feature_cols = [c for c in df.columns
                    if c not in NON_FEATURE_COLS and c not in LEAKY_COLS]
    features = df[feature_cols].fillna(0).to_numpy(dtype=np.float32)

    users = df['userId'].map(user_index).to_numpy(dtype=np.int64)
    movies = df['movieId'].map(movie_index).to_numpy(dtype=np.int64)
    ratings = df['rating'].to_numpy(dtype=np.float32)

    splits, meta = {}, {}
    for name in ('train', 'val', 'test'):
        mask = (df['split'] == name).to_numpy()
        splits[name] = {
            'user': torch.from_numpy(users[mask]).to(device),
            'movie': torch.from_numpy(movies[mask]).to(device),
            'features': torch.from_numpy(features[mask]).to(device),
            'rating': torch.from_numpy(ratings[mask]).to(device),
        }
        # same row order as the tensors; used by utils for Top-K ranking metrics
        meta[name] = df.loc[mask, ['userId', 'movieId', 'rating']].copy()

    info = {
        'n_users': len(user_index),
        'n_movies': len(movie_index),
        'n_features': len(feature_cols),
        'feature_cols': feature_cols,
        'global_mean': float(ratings[(df['split'] == 'train').to_numpy()].mean()),
    }
    print(f"Users: {info['n_users']:,} | Movies: {info['n_movies']:,} | "
          f"Side features: {info['n_features']}")
    print({k: len(v['rating']) for k, v in splits.items()})
    return splits, meta, info


def split_user_item_features(feature_cols):
    """Column positions of the user-side and movie-side features within the feature matrix."""
    user_idx = [i for i, c in enumerate(feature_cols)
                if c in USER_FEATURE_COLS or c.startswith(USER_FEATURE_PREFIXES)]
    user_set = set(user_idx)
    item_idx = [i for i in range(len(feature_cols)) if i not in user_set]
    print(f"User-side features: {len(user_idx)} | Movie-side features: {len(item_idx)}")
    return user_idx, item_idx


# =====================================================================
# Model
# =====================================================================

class NCF(nn.Module):
    """NeuMF: a GMF branch (element-wise product) + an MLP branch, fused by a linear head."""

    def __init__(self, n_users, n_movies, n_features=0, embed_dim=EMBED_DIM,
                 mlp_layers=MLP_LAYERS, dropout=DROPOUT, global_mean=3.5):
        super().__init__()
        self.n_features = n_features

        self.user_gmf = nn.Embedding(n_users, embed_dim)
        self.movie_gmf = nn.Embedding(n_movies, embed_dim)
        self.user_mlp = nn.Embedding(n_users, embed_dim)
        self.movie_mlp = nn.Embedding(n_movies, embed_dim)

        layers, in_dim = [], 2 * embed_dim + n_features
        for out_dim in mlp_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)

        self.head = nn.Linear(embed_dim + mlp_layers[-1], 1)
        nn.init.constant_(self.head.bias, global_mean)

        for emb in (self.user_gmf, self.movie_gmf, self.user_mlp, self.movie_mlp):
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, user, movie, features=None):
        gmf = self.user_gmf(user) * self.movie_gmf(movie)
        mlp_in = [self.user_mlp(user), self.movie_mlp(movie)]
        if self.n_features > 0:
            mlp_in.append(features)
        mlp = self.mlp(torch.cat(mlp_in, dim=1))
        return self.head(torch.cat([gmf, mlp], dim=1)).squeeze(1)


# =====================================================================
# Bridge to utils.evaluate_nn
# =====================================================================

class NCFPredictor:
    """
    Gives a torch model the sklearn-style .predict(X) that utils.evaluate_nn expects.
    X is just the row positions (np.arange(n)) within one split, because the models
    need user / movie ids and side features that live in the split's tensors.
    Works for any model with forward(user, movie, features).
    """

    def __init__(self, model, data, use_features, batch_size=8192):
        self.model, self.data = model, data
        self.use_features, self.batch_size = use_features, batch_size

    @torch.no_grad()
    def predict(self, X):
        self.model.eval()
        idx_all = torch.as_tensor(np.asarray(X).ravel(), dtype=torch.long,
                                  device=self.data['user'].device)
        out = []
        for s in range(0, len(idx_all), self.batch_size):
            idx = idx_all[s:s + self.batch_size]
            feats = self.data['features'][idx] if self.use_features else None
            out.append(self.model(self.data['user'][idx], self.data['movie'][idx], feats))
        return torch.cat(out).clamp(RATING_MIN, RATING_MAX).cpu().numpy()


def evaluate_all_splits(label, model_factory, splits, meta, train_stats):
    """Runs utils.evaluate_nn on train/val/test and attaches the training cost columns."""
    rows = []
    for split_name in ('train', 'val', 'test'):
        n = len(splits[split_name]['rating'])
        X = np.arange(n)
        y = splits[split_name]['rating'].cpu().numpy()
        device = splits[split_name]['rating'].device
        res = evaluate_nn(split_name, model_factory(split_name), X, y,
                          meta[split_name], device=device)
        res['model'] = label
        res.update(train_stats)
        rows.append(res)
    return rows


# =====================================================================
# Train (generic: works for NCF, C-NGE, NMetricF, ...)
# =====================================================================

def train_model(build_model, splits, info, device, use_features=True):
    """
    Trains with early stopping on val RMSE.
    build_model() must return an nn.Module with forward(user, movie, features).
    Returns (best model, stats dict, training-params dict).
    """
    set_seed()
    train, val = splits['train'], splits['val']
    n_train = len(train['rating'])

    best = {'rmse': float('inf'), 'state': None, 'epoch': 0}
    stale, epochs_run = 0, 0

    # Time and memory cover everything training needs: model creation (weights),
    # optimizer state, every epoch and the per-epoch validation passes.
    with track_resources(device) as res:
        model = build_model().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                     weight_decay=WEIGHT_DECAY)
        loss_fn = nn.MSELoss()
        val_pred_model = NCFPredictor(model, val, use_features)

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            perm = torch.randperm(n_train, device=device)
            running_loss = 0.0

            for start in range(0, n_train, BATCH_SIZE):
                idx = perm[start:start + BATCH_SIZE]
                feats = train['features'][idx] if use_features else None
                pred = model(train['user'][idx], train['movie'][idx], feats)
                loss = loss_fn(pred, train['rating'][idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(idx)

            epochs_run = epoch
            train_rmse = (running_loss / n_train) ** 0.5
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
        'n_parameters': n_params, 'use_side_features': use_features,
    }
    return model, stats, params


# =====================================================================
# Dataset loop (same flow as knn.py)
# =====================================================================

def run_datasets(datasets, make_builder, model_params, use_features=True):
    """
    For each (label, loader): load the data, train, evaluate train/val/test and save.
    make_builder(info) returns the zero-argument function that builds a fresh model.
    model_params is the model-specific hyper-parameter dict logged to the CSV.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for label, loader in datasets:
        print(f"\n{'=' * 60}\n=== MODEL: {label} ===\n{'=' * 60}")
        df = loader()
        splits, meta, info = prepare_data(df, device)
        del df
        gc.collect()

        model, stats, train_params = train_model(
            make_builder(info), splits, info, device, use_features)
        rows = evaluate_all_splits(
            label, lambda s: NCFPredictor(model, splits[s], use_features),
            splits, meta, stats)
        save_results(rows, {**model_params, **train_params})

        # free memory before the next dataset so it doesn't skew its measurements
        del model, splits, meta
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print("\nResults appended to results/model_evaluation_results.csv")


# =====================================================================
# Main
# =====================================================================

def main():
    def make_builder(info):
        return lambda: NCF(
            info['n_users'], info['n_movies'], n_features=info['n_features'],
            global_mean=info['global_mean'])

    run_datasets(
        DATASETS, make_builder,
        {'embed_dim': EMBED_DIM, 'mlp_layers': MLP_LAYERS, 'dropout': DROPOUT},
        use_features=True)


if __name__ == "__main__":
    main()
