"""Biased matrix factorization trained with mini-batch SGD.

    r_hat(u, i) = mu + b_u + b_i + p_u . q_i

The original version of this did one rating per Python iteration. That was
painfully slow on the full ML-100K - moved to batched np.add.at updates and
the per-epoch time dropped by something like 3-4x on my laptop.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MFConfig:
    n_factors: int = 64
    n_epochs: int = 20
    lr: float = 0.01
    reg: float = 0.05
    batch_size: int = 4096
    seed: int = 42
    verbose: bool = True


class MatrixFactorization:

    def __init__(self, n_users, n_items, config=None):
        self.n_users = n_users
        self.n_items = n_items
        self.cfg = config or MFConfig()

        rng = np.random.default_rng(self.cfg.seed)
        # Small init so the bias terms dominate the first few epochs.
        scale = 0.1 / np.sqrt(self.cfg.n_factors)
        self.P = rng.normal(0.0, scale, (n_users, self.cfg.n_factors)).astype(np.float32)
        self.Q = rng.normal(0.0, scale, (n_items, self.cfg.n_factors)).astype(np.float32)
        self.bu = np.zeros(n_users, dtype=np.float32)
        self.bi = np.zeros(n_items, dtype=np.float32)
        self.mu = 0.0

    def fit(self, users, items, ratings):
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        ratings = np.asarray(ratings, dtype=np.float32)
        self.mu = float(ratings.mean())

        cfg = self.cfg
        n = len(ratings)
        rng = np.random.default_rng(cfg.seed)

        for epoch in range(cfg.n_epochs):
            perm = rng.permutation(n)
            u_all, i_all, r_all = users[perm], items[perm], ratings[perm]
            sse = 0.0

            for start in range(0, n, cfg.batch_size):
                end = start + cfg.batch_size
                u = u_all[start:end]
                i = i_all[start:end]
                r = r_all[start:end]

                pu = self.P[u]
                qi = self.Q[i]
                pred = self.mu + self.bu[u] + self.bi[i] + np.einsum("bf,bf->b", pu, qi)
                err = (r - pred).astype(np.float32)
                sse += float(np.dot(err, err))

                # np.add.at does unbuffered scatter-add - critical when the
                # same user/item shows up twice in a batch. Plain `+=` would
                # silently lose updates in that case.
                np.add.at(self.P, u, cfg.lr * (err[:, None] * qi - cfg.reg * pu))
                np.add.at(self.Q, i, cfg.lr * (err[:, None] * pu - cfg.reg * qi))
                np.add.at(self.bu, u, cfg.lr * (err - cfg.reg * self.bu[u]))
                np.add.at(self.bi, i, cfg.lr * (err - cfg.reg * self.bi[i]))

            if cfg.verbose:
                print(f"[MF] epoch {epoch + 1:02d}/{cfg.n_epochs}  RMSE={np.sqrt(sse / n):.4f}")
        return self

    def predict(self, users, items):
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        pred = (
            self.mu
            + self.bu[users]
            + self.bi[items]
            + np.einsum("bf,bf->b", self.P[users], self.Q[items])
        )
        return np.clip(pred, 1.0, 5.0)

    def predict_for_user(self, user):
        # Score every item for one user. Cheap - one matvec.
        scores = self.mu + self.bu[user] + self.bi + self.P[user] @ self.Q.T
        return np.clip(scores, 1.0, 5.0)

    def score_matrix(self, users=None):
        # Dense (|users|, n_items) score grid. Used by the ranking eval.
        if users is None:
            users = np.arange(self.n_users)
        users = np.asarray(users, dtype=np.int64)
        scores = (
            self.mu
            + self.bu[users][:, None]
            + self.bi[None, :]
            + self.P[users] @ self.Q.T
        )
        return np.clip(scores, 1.0, 5.0)
