"""
Neural Metric Factorization (NMetricF) for rating prediction on MovieLens 1M.

Reference: Sun et al., "Neural Metric Factorization for Recommendation",
Mathematics 10(3), 503, 2022.

Idea: instead of a dot product between user and item vectors (matrix factorization),
users and items are mapped by neural networks into one Euclidean space and the model
predicts the *distance* between them. A rating is turned into a distance by
inverting it (distance = RATING_MAX - rating: a 5-star rating means "close"), and the
predicted distance is inverted back to give the rating:

    rating_hat = RATING_MAX - || f_user(u) - f_item(i) ||_2

Training on MSE of the rating is the same as MSE of the distance, because
(d_hat - d)^2 = (r - r_hat)^2.

Each tower is: [embedding (+ side features)] -> ReLU MLP -> linear output vector.
Set USE_SIDE_FEATURES = False for the paper's ratings-only setting.

Training, evaluation, timing and saving reuse the shared pipeline in ncf.py, so the
results are directly comparable with the other neural models in the results CSV.

Requires: torch, numpy, pandas, pyarrow, scikit-learn, psutil
"""
import torch
import torch.nn as nn

from preprocessing import load_ratings_scaled, load_ratings_preprocessed
from models.ncf import (EMBED_DIM, DROPOUT, RATING_MAX,
                 run_datasets, split_user_item_features)

USE_SIDE_FEATURES = True      # True: towers also see user / movie side features (like NCF)
TOWER_LAYERS = (64, 32)       # hidden layers of each tower
OUT_DIM = 16                  # dimension of the shared metric space

# Labels are split on the first '_' by save_results: Algorithm 'NMetricF', Dataset 'preprocessed'/'scaled'
DATASETS = [
    ('NMetricF_PREPROCESSED', load_ratings_preprocessed),
    ('NMetricF_SCALED', load_ratings_scaled),
]


def make_tower(in_dim, hidden, out_dim, dropout):
    layers = []
    for h in hidden:
        layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = h
    layers.append(nn.Linear(in_dim, out_dim))     # linear output: coordinates can be negative
    return nn.Sequential(*layers)


class NMetricF(nn.Module):
    def __init__(self, n_users, n_movies, user_idx, item_idx, use_features=True,
                 embed_dim=EMBED_DIM, hidden=TOWER_LAYERS, out_dim=OUT_DIM, dropout=DROPOUT):
        super().__init__()
        self.use_features = use_features
        self.register_buffer('user_idx', torch.tensor(user_idx, dtype=torch.long))
        self.register_buffer('item_idx', torch.tensor(item_idx, dtype=torch.long))

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.movie_emb = nn.Embedding(n_movies, embed_dim)
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.movie_emb.weight, std=0.1)

        user_in = embed_dim + (len(user_idx) if use_features else 0)
        item_in = embed_dim + (len(item_idx) if use_features else 0)
        self.user_tower = make_tower(user_in, hidden, out_dim, dropout)
        self.item_tower = make_tower(item_in, hidden, out_dim, dropout)

    def forward(self, user, movie, features=None):
        u, v = self.user_emb(user), self.movie_emb(movie)
        if self.use_features:
            u = torch.cat([u, features[:, self.user_idx]], dim=1)
            v = torch.cat([v, features[:, self.item_idx]], dim=1)

        p, q = self.user_tower(u), self.item_tower(v)
        # Euclidean distance (the small constant avoids an infinite gradient at distance 0)
        distance = torch.sqrt(((p - q) ** 2).sum(dim=1) + 1e-8)
        return RATING_MAX - distance          # invert the distance back into a rating


def main():
    def make_builder(info):
        user_idx, item_idx = split_user_item_features(info['feature_cols'])
        return lambda: NMetricF(info['n_users'], info['n_movies'], user_idx, item_idx,
                                use_features=USE_SIDE_FEATURES)

    run_datasets(
        DATASETS, make_builder,
        {'embed_dim': EMBED_DIM, 'tower_layers': TOWER_LAYERS, 'out_dim': OUT_DIM,
         'dropout': DROPOUT},
        use_features=USE_SIDE_FEATURES)


if __name__ == "__main__":
    main()