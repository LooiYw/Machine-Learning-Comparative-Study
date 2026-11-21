"""
DeepMF for rating prediction on MovieLens 1M.

Reads data/processed/ratings_*.parquet (built by preprocessing.py) through the same
shared pipeline as ncf.py: trains on the 'train' rows, picks the best epoch on the
'val' rows (early stopping on RMSE), then evaluates train / val / test with
utils.evaluate_nn and appends the rows to results/model_evaluation_results.csv
via utils.save_results.

Model: two-tower DeepMF. The user tower encodes the user's row of the train rating
matrix, the movie tower encodes the movie's column; the towers are combined with a
scaled inner product plus user / movie / global biases and squashed into [1, 5].
The rating matrix is built from TRAIN rows only (val/test ratings never enter it).
No side features are used (as in the original DeepMF), so the leaky user_avg /
movie_avg columns and all other side features are ignored.

Requires: torch, numpy, pandas, pyarrow, scikit-learn, psutil
"""
import gc
import math

import torch
import torch.nn as nn

from models.ncf import (NCFPredictor, evaluate_all_splits, prepare_data, train_model)
from preprocessing import load_ratings_scaled, load_ratings_preprocessed
from utils import save_results

HIDDEN_DIM = 128
LATENT_DIM = 32

# Label is split on the first '_' by save_results: Algorithm 'DEEPMF', Dataset 'preprocessed'/'scaled'.
DATASETS = [
    ('DEEPMF_PREPROCESSED', load_ratings_preprocessed),
    ('DEEPMF_SCALED', load_ratings_scaled),
]


class DeepMF(nn.Module):
    """Two-tower DeepMF with biases and a bounded [1, 5] rating head."""

    def __init__(self, train, n_users, n_movies, global_mean=3.5,
                 hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        # Dense train-only rating matrix scaled to [0, 1]; 0 means "not observed".
        # persistent=False keeps it out of state_dict, so best-epoch snapshots stay small.
        matrix = torch.zeros(n_users, n_movies, device=train['user'].device)
        matrix[train['user'], train['movie']] = train['rating'] / 5.0
        self.register_buffer('ratings', matrix, persistent=False)

        self.user_net = nn.Sequential(nn.Linear(n_movies, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, latent_dim))
        self.item_net = nn.Sequential(nn.Linear(n_users, hidden_dim), nn.ReLU(),
                                      nn.Linear(hidden_dim, latent_dim))
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_movies, 1)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

        # start the output at the train mean: 1 + 4 * sigmoid(b) = global_mean
        p = min(max((global_mean - 1.0) / 4.0, 1e-3), 1 - 1e-3)
        self.global_bias = nn.Parameter(torch.tensor(math.log(p / (1 - p))))
        self.scale = math.sqrt(latent_dim)

    def forward(self, user, movie, features=None):
        # encode each distinct id in the batch once (rating vectors are long)
        uu, ui = torch.unique(user, return_inverse=True)
        ii, ij = torch.unique(movie, return_inverse=True)
        uv = self.user_net(self.ratings[uu])[ui]
        iv = self.item_net(self.ratings[:, ii].T)[ij]
        score = (uv * iv).sum(-1) / self.scale
        score = (score + self.user_bias(user).squeeze(-1)
                 + self.item_bias(movie).squeeze(-1) + self.global_bias)
        return 1.0 + 4.0 * torch.sigmoid(score)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model_params = {'hidden_dim': HIDDEN_DIM, 'latent_dim': LATENT_DIM}

    for label, loader in DATASETS:
        print(f"\n{'=' * 60}\n=== MODEL: {label} ===\n{'=' * 60}")
        df = loader()
        splits, meta, info = prepare_data(df, device)
        del df
        gc.collect()

        # the builder closes over the train split: the rating matrix is created
        # inside train_model's resource tracking, so its memory is counted
        def build():
            return DeepMF(splits['train'], info['n_users'], info['n_movies'],
                          global_mean=info['global_mean'])

        model, stats, train_params = train_model(
            build, splits, info, device, use_features=False)
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