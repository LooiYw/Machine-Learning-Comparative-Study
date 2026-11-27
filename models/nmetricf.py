"""
Neural Metric Factorization (NMetricF) for rating prediction on MovieLens 1M.

Reference: Sun et al., "Neural Metric Factorization for Recommendation",
Mathematics 10(3), 503, 2022. https://doi.org/10.3390/math10030503

This is a corrected reimplementation that follows Section 4 of the paper exactly:

  Input layer         - user id / item id, one-hot (Sec. 4.2)
  Embedding layer     - e_u = P_u^T u,  e_i = P_i^T i                    (Eq. 3)
  Metric interaction  - h_0 = e_u (-) e_i   <-- element-wise SUBTRACTION
  Nonlinear MLP        h_1 = ReLU(W_1^T h_0 + b_1)
                        ...
                        h_{L-1} = ReLU(W_{L-1}^T h_{L-2} + b_{L-1})
  Output layer        - y_hat = W_L^T h_{L-1} + b_L                     (Eq. 4)

The paper is explicit (Sec. 4.4) that if you collapse the MLP to a single step that
squares every dimension of h_0 and sums + sqrt's them, you recover plain (linear)
Metric Factorization (FML, Zhang et al.). NMetricF is the deep, nonlinear
generalization of that one operation. Concretely this means:

  - There is only ONE embedding per user/item (dimension k) and ONE shared MLP,
    applied to the elementwise DIFFERENCE of the two embeddings.
  - There are NOT two independent towers that each map user/item into a separate
    output space whose outputs are then compared with an explicit Euclidean norm.
  - The final y_hat is read directly off the last linear layer - there is no
    torch.sqrt/torch.pow at the output. The "metric" structure comes entirely from
    starting the MLP with a subtraction, not from an explicit distance formula at
    the end.

The previous version of this file used two independent towers (user_tower,
item_tower) and computed torch.sqrt(((p - q) ** 2).sum(dim=1) + eps) on their
outputs. That is a twin-network / metric-learning architecture, not the one
described in the paper, and has been replaced below.

y_hat here is still the model's *distance* prediction, exactly as before: the data
loader is assumed to hand the model targets y = RATING_MAX - rating (same convention
as the original file), and the rating is recovered as RATING_MAX - y_hat. Training on
MSE of the distance is equivalent to MSE of the rating, as the original docstring noted.

Paper defaults (Sec. 5.1.1, "Implementation Details"), reproduced here as module
constants:
  - weight init             : LeCun uniform ("lecun_uniform distribution")
  - MLP depth/width         : 6 layers total, e.g. 200-150-100-50-25-1 for their
                               embed_dim = 200 configuration (i.e. 4 hidden layers of
                               150-100-50-25 after the embedding, then 1 output)
  - normalization            : BatchNorm BETWEEN layers, used INSTEAD OF dropout
  - regularization           : L2 penalty on weights, lambda = 0.02              (Eq. 5)
  - optimizer                 : Adam

USE_SIDE_FEATURES is NOT part of the paper's base model - NMetricF's input layer is
ratings-only (one-hot user id / item id, Sec. 4.2). The flag is kept here only as an
optional extension (the paper's own "Summary and Prospect" section suggests folding in
auxiliary information as future work); set it to False to match the paper exactly.
When enabled, side features are linearly projected to embed_dim before being added to
the corresponding embedding, so the subtraction in the metric interaction layer stays
well-defined - this projection is this implementation's addition, not the paper's.

Training, evaluation, timing and saving still reuse the shared pipeline in ncf.py, so
results stay comparable with the other neural models in the results CSV. The model's
forward(user, movie, features=None) signature is unchanged so it keeps working with
that pipeline; REG_LAMBDA is exposed as a module constant so it can be wired into the
optimizer's weight_decay wherever run_datasets configures Adam.

Requires: torch, numpy, pandas, pyarrow, scikit-learn, psutil
"""
import math

import torch
import torch.nn as nn

from preprocessing import load_ratings_scaled, load_ratings_preprocessed
from models.ncf import (EMBED_DIM, DROPOUT, RATING_MAX,
                 run_datasets, split_user_item_features)

USE_SIDE_FEATURES = False        # paper's base model: ratings only (Sec. 4.2) - True is an extension
MLP_LAYERS = (150, 100, 50, 25)  # hidden sizes of the SHARED metric MLP (paper: 150-100-50-25 for embed_dim=200)
USE_BATCHNORM = True             # paper: BatchNorm between layers, used INSTEAD of dropout
USE_DROPOUT = False              # not used by the paper; kept only for ablation/comparability
REG_LAMBDA = 0.02                # L2 regularization coefficient (Eq. 5) - wire into optimizer weight_decay

# Labels are split on the first '_' by save_results: Algorithm 'NMetricF', Dataset 'preprocessed'/'scaled'
DATASETS = [
    ('NMetricF_PREPROCESSED', load_ratings_preprocessed),
    ('NMetricF_SCALED', load_ratings_scaled),
]


def lecun_uniform_(tensor):
    """LeCun uniform init: U(-limit, limit) with limit = sqrt(3 / fan_in).
    Matches the paper: "we use the lecun_uniform distribution to initialize the
    weight parameters of the model" (Sec. 5.1.1)."""
    fan_in = tensor.shape[1] if tensor.dim() > 1 else tensor.shape[0]
    limit = math.sqrt(3.0 / fan_in)
    with torch.no_grad():
        return tensor.uniform_(-limit, limit)


def make_metric_mlp(in_dim, hidden, use_batchnorm, use_dropout, dropout_p):
    """The shared nonlinear MLP applied to h_0 = e_u (-) e_i (Sec. 4.4, Eq. 4).
    Ends in a single linear output neuron with NO activation (rating prediction is
    a regression task; the paper reserves sigmoid for the separate ranking task,
    which this file does not implement)."""
    layers = []
    for h in hidden:
        linear = nn.Linear(in_dim, h)
        lecun_uniform_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(h))
        layers.append(nn.ReLU())
        if use_dropout:
            layers.append(nn.Dropout(dropout_p))
        in_dim = h
    out = nn.Linear(in_dim, 1)
    lecun_uniform_(out.weight)
    nn.init.zeros_(out.bias)
    layers.append(out)
    return nn.Sequential(*layers)


class NMetricF(nn.Module):
    def __init__(self, n_users, n_movies, user_idx, item_idx, use_features=USE_SIDE_FEATURES,
                 embed_dim=EMBED_DIM, hidden=MLP_LAYERS, dropout=DROPOUT,
                 use_batchnorm=USE_BATCHNORM, use_dropout=USE_DROPOUT):
        super().__init__()
        self.use_features = use_features
        self.register_buffer('user_idx', torch.tensor(user_idx, dtype=torch.long))
        self.register_buffer('item_idx', torch.tensor(item_idx, dtype=torch.long))

        self.user_emb = nn.Embedding(n_users, embed_dim)
        self.movie_emb = nn.Embedding(n_movies, embed_dim)
        lecun_uniform_(self.user_emb.weight)
        lecun_uniform_(self.movie_emb.weight)

        # Extension beyond the paper: if side features are used, project them down
        # to embed_dim so e_u and e_i stay the same shape and the subtraction below
        # (the paper's actual metric interaction) remains well-defined.
        if use_features:
            self.user_feat_proj = nn.Linear(len(user_idx), embed_dim)
            self.item_feat_proj = nn.Linear(len(item_idx), embed_dim)

        # ONE shared metric MLP over the element-wise difference of the two
        # embeddings (Sec. 4.4) - not two independent towers compared by a norm.
        self.metric_mlp = make_metric_mlp(embed_dim, hidden, use_batchnorm, use_dropout, dropout)

    def forward(self, user, movie, features=None):
        e_u = self.user_emb(user)
        e_i = self.movie_emb(movie)
        if self.use_features:
            e_u = e_u + self.user_feat_proj(features[:, self.user_idx])
            e_i = e_i + self.item_feat_proj(features[:, self.item_idx])

        h0 = e_u - e_i                        # element-wise subtraction (⊖), Sec. 4.4
        y_hat_distance = self.metric_mlp(h0).squeeze(-1)
        return RATING_MAX - y_hat_distance    # invert the predicted distance back into a rating


def main():
    def make_builder(info):
        user_idx, item_idx = split_user_item_features(info['feature_cols'])
        return lambda: NMetricF(info['n_users'], info['n_movies'], user_idx, item_idx,
                                use_features=USE_SIDE_FEATURES)

    run_datasets(
        DATASETS, make_builder,
        {'embed_dim': EMBED_DIM, 'mlp_layers': MLP_LAYERS, 'use_batchnorm': USE_BATCHNORM,
         'reg_lambda': REG_LAMBDA},
        use_features=USE_SIDE_FEATURES)


if __name__ == "__main__":
    main()