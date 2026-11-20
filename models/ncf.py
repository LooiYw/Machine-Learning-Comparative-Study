"""
Neural Collaborative Filtering (NeuMF) for rating prediction on MovieLens 1M.

Reads data/processed/ratings_scaled.parquet (built by preprocess_ml1m.py), trains
on the 'train' rows, picks the best epoch on the 'val' rows, and reports RMSE / MAE
on train, val and test. Results are written to model_evaluation_results.csv.

Two variants are trained and compared:
  - NCF (IDs only)         : classic NeuMF, user and movie embeddings only
  - NCF (IDs + side feats) : same, plus the scaled user / movie features fed into the MLP

Requires: torch, numpy, pandas, pyarrow
"""
import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from preprocessing import load_ratings_scaled

RESULTS_PATH = 'model_evaluation_results.csv'

SEED = 42
EMBED_DIM = 32
MLP_LAYERS = (64, 32, 16)
DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 1024
MAX_EPOCHS = 30
PATIENCE = 3                    # stop after this many epochs without val RMSE gain
RATING_MIN, RATING_MAX = 1.0, 5.0

# Columns that are not model inputs. user_avg / movie_avg are excluded from the
# side features because, on train rows, they include the row's own rating
# (target leakage), which makes the model look great in training and worse on
# val/test.
NON_FEATURE_COLS = ['userId', 'movieId', 'rating', 'split']
LEAKY_COLS = ['user_avg', 'movie_avg']


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# Data
# =====================================================================

def prepare_data(df, device):
    """Maps ids to contiguous indices and builds tensors for each split."""
    user_index = {u: i for i, u in enumerate(np.sort(df['userId'].unique()))}
    movie_index = {m: i for i, m in enumerate(np.sort(df['movieId'].unique()))}

    feature_cols = [c for c in df.columns
                    if c not in NON_FEATURE_COLS and c not in LEAKY_COLS]
    # NaN only appears for movies unseen in train (movie_popularity); 0 is the
    # scaled mean, so it reads as "average".
    features = df[feature_cols].fillna(0).to_numpy(dtype=np.float32)

    users = df['userId'].map(user_index).to_numpy(dtype=np.int64)
    movies = df['movieId'].map(movie_index).to_numpy(dtype=np.int64)
    ratings = df['rating'].to_numpy(dtype=np.float32)

    splits = {}
    for name in ('train', 'val', 'test'):
        mask = (df['split'] == name).to_numpy()
        splits[name] = {
            'user': torch.from_numpy(users[mask]).to(device),
            'movie': torch.from_numpy(movies[mask]).to(device),
            'features': torch.from_numpy(features[mask]).to(device),
            'rating': torch.from_numpy(ratings[mask]).to(device),
        }

    info = {
        'n_users': len(user_index),
        'n_movies': len(movie_index),
        'n_features': len(feature_cols),
        'global_mean': float(ratings[(df['split'] == 'train').to_numpy()].mean()),
    }
    print(f"Users: {info['n_users']:,} | Movies: {info['n_movies']:,} | "
          f"Side features: {info['n_features']}")
    print({k: len(v['rating']) for k, v in splits.items()})
    return splits, info


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
        # start predicting the global mean instead of 0
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
# Train / evaluate
# =====================================================================

@torch.no_grad()
def predict(model, data, use_features, batch_size=8192):
    model.eval()
    preds = []
    n = len(data['rating'])
    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        feats = data['features'][sl] if use_features else None
        preds.append(model(data['user'][sl], data['movie'][sl], feats))
    return torch.cat(preds).clamp(RATING_MIN, RATING_MAX)


def rmse_mae(pred, target):
    err = pred - target
    return float(torch.sqrt((err ** 2).mean())), float(err.abs().mean())


def train_and_evaluate(name, splits, info, use_features, device):
    set_seed()
    model = NCF(
        info['n_users'], info['n_movies'],
        n_features=info['n_features'] if use_features else 0,
        global_mean=info['global_mean'],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    train = splits['train']
    n_train = len(train['rating'])

    print(f"\n{'=' * 60}\nTraining: {name}  ({n_params:,} parameters, device={device})\n{'=' * 60}")
    start_time = time.time()
    best = {'rmse': float('inf'), 'state': None, 'epoch': 0}
    stale = 0
    epochs_run = 0

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
        val_rmse, val_mae = rmse_mae(predict(model, splits['val'], use_features),
                                     splits['val']['rating'])
        print(f"Epoch {epoch:02d} | train RMSE {train_rmse:.4f} | "
              f"val RMSE {val_rmse:.4f} | val MAE {val_mae:.4f}")

        if val_rmse < best['rmse'] - 1e-4:
            best.update(rmse=val_rmse, state=copy.deepcopy(model.state_dict()), epoch=epoch)
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"Early stopping (no val improvement for {PATIENCE} epochs)")
                break

    train_seconds = time.time() - start_time
    model.load_state_dict(best['state'])
    print(f"Best epoch: {best['epoch']} (val RMSE {best['rmse']:.4f})")

    rows = []
    for split_name in ('train', 'val', 'test'):
        rmse, mae = rmse_mae(predict(model, splits[split_name], use_features),
                             splits[split_name]['rating'])
        rows.append({
            'model': name, 'split': split_name,
            'rmse': round(rmse, 4), 'mae': round(mae, 4),
            'best_epoch': best['epoch'], 'epochs_run': epochs_run,
            'parameters': n_params, 'train_seconds': round(train_seconds, 1),
        })
        print(f"{split_name:<5} RMSE {rmse:.4f} | MAE {mae:.4f}")
    return rows


def baseline_rows(splits, info):
    """Predict the train-set mean for everything, as a reference point."""
    rows = []
    for split_name in ('train', 'val', 'test'):
        target = splits[split_name]['rating']
        pred = torch.full_like(target, info['global_mean'])
        rmse, mae = rmse_mae(pred, target)
        rows.append({'model': 'Global mean baseline', 'split': split_name,
                     'rmse': round(rmse, 4), 'mae': round(mae, 4),
                     'best_epoch': 0, 'epochs_run': 0, 'parameters': 0,
                     'train_seconds': 0.0})
    return rows


# =====================================================================
# Main
# =====================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    df = load_ratings_scaled()
    splits, info = prepare_data(df, device)
    del df

    results = baseline_rows(splits, info)
    results += train_and_evaluate('NCF (IDs only)', splits, info,
                                  use_features=False, device=device)
    results += train_and_evaluate('NCF (IDs + side features)', splits, info,
                                  use_features=True, device=device)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved: {RESULTS_PATH}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()