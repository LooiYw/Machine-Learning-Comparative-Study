"""
Collaborative Neural Generative Embedding (C-NGE) for rating prediction on MovieLens 1M.

Reference: Nahta et al., "A Deep Collaborative Neural Generative Embedding for Rating
Prediction in Movie Recommendation Systems", CMES 144(1), 2025.

Three modules, as in the paper:
  R-NGE : user / movie latent vectors are *generated* with the reparameterization
          trick, z = mu + sigma * eps, where eps comes from the user / movie metadata
          (metadata acts as the sampled noise). The pseudo-rating branch is the
          element-wise product z_u * z_i.
  M-NE  : user and movie metadata -> embedding layer -> 3-layer ReLU MLP
          (each layer half the size of the previous one).
  C-NGE : concatenation of the R-NGE and M-NE outputs -> output layer with a sigmoid,
          rescaled to the 1-5 rating range.

Training, evaluation, timing and saving reuse the shared pipeline in ncf.py, so the
results are directly comparable with the NCF rows in the results CSV.

Requires: torch, numpy, pandas, pyarrow, scikit-learn, psutil
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing import load_ratings_scaled, load_ratings_preprocessed
from models.ncf import (EMBED_DIM, DROPOUT, RATING_MIN, RATING_MAX,
                 run_datasets, split_user_item_features)

META_DIM = 32                 # size of the metadata embedding layer (M-NE)
MLP_LAYERS = (64, 32, 16)     # M-NE hidden layers, each half the previous one (as in the paper)

# Labels are split on the first '_' by save_results: Algorithm 'C-NGE', Dataset 'preprocessed'/'scaled'
DATASETS = [
    ('C-NGE_PREPROCESSED', load_ratings_preprocessed),
    ('C-NGE_SCALED', load_ratings_scaled),
]


class CNGE(nn.Module):
    def __init__(self, n_users, n_movies, user_idx, item_idx, embed_dim=EMBED_DIM,
                 meta_dim=META_DIM, mlp_layers=MLP_LAYERS, dropout=DROPOUT, global_mean=3.5):
        super().__init__()
        # which columns of the side-feature matrix are user metadata / movie metadata
        self.register_buffer('user_idx', torch.tensor(user_idx, dtype=torch.long))
        self.register_buffer('item_idx', torch.tensor(item_idx, dtype=torch.long))
        n_user_meta, n_item_meta = len(user_idx), len(item_idx)

        # ---- R-NGE: generated user / movie embeddings --------------------------
        self.user_mu = nn.Embedding(n_users, embed_dim)
        self.user_sigma = nn.Embedding(n_users, embed_dim)     # softplus -> sigma > 0
        self.movie_mu = nn.Embedding(n_movies, embed_dim)
        self.movie_sigma = nn.Embedding(n_movies, embed_dim)
        # metadata -> noise term eps (bounded with tanh so unscaled features can't blow it up)
        self.user_noise = nn.Linear(n_user_meta, embed_dim)
        self.movie_noise = nn.Linear(n_item_meta, embed_dim)

        # ---- M-NE: metadata embedding + deep MLP -------------------------------
        self.user_meta = nn.Linear(n_user_meta, meta_dim)      # embedding layer for user metadata
        self.movie_meta = nn.Linear(n_item_meta, meta_dim)     # embedding layer for movie metadata
        layers, in_dim = [], 2 * meta_dim
        for out_dim in mlp_layers:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        self.mne = nn.Sequential(*layers)

        # ---- C-NGE: fuse R-NGE and M-NE ----------------------------------------
        self.head = nn.Linear(embed_dim + mlp_layers[-1], 1)
        # start by predicting the global mean: sigmoid(bias) maps to the mean rating
        p = (global_mean - RATING_MIN) / (RATING_MAX - RATING_MIN)
        nn.init.constant_(self.head.bias, math.log(p / (1 - p)))

        for emb in (self.user_mu, self.user_sigma, self.movie_mu, self.movie_sigma):
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, user, movie, features):
        user_meta = features[:, self.user_idx]
        movie_meta = features[:, self.item_idx]

        # R-NGE: z = mu + sigma * eps  (reparameterization trick, eps from metadata)
        eps_u = torch.tanh(self.user_noise(user_meta))
        eps_i = torch.tanh(self.movie_noise(movie_meta))
        z_u = self.user_mu(user) + F.softplus(self.user_sigma(user)) * eps_u
        z_i = self.movie_mu(movie) + F.softplus(self.movie_sigma(movie)) * eps_i
        phi_r = z_u * z_i

        # M-NE
        m_in = torch.cat([self.user_meta(user_meta), self.movie_meta(movie_meta)], dim=1)
        phi_m = self.mne(m_in)

        # C-NGE: concatenate -> sigmoid output rescaled to 1-5
        logit = self.head(torch.cat([phi_r, phi_m], dim=1)).squeeze(1)
        return RATING_MIN + (RATING_MAX - RATING_MIN) * torch.sigmoid(logit)


def main():
    def make_builder(info):
        user_idx, item_idx = split_user_item_features(info['feature_cols'])
        return lambda: CNGE(info['n_users'], info['n_movies'], user_idx, item_idx,
                            global_mean=info['global_mean'])

    run_datasets(
        DATASETS, make_builder,
        {'embed_dim': EMBED_DIM, 'meta_dim': META_DIM, 'mlp_layers': MLP_LAYERS,
         'dropout': DROPOUT},
        use_features=True)       # C-NGE needs the metadata features


if __name__ == "__main__":
    main()
