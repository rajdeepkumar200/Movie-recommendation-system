"""SVD++.

Adds an implicit-feedback term on top of biased MF:

    r_hat(u, i) = mu + b_u + b_i + q_i . ( p_u + |N(u)|^(-1/2) * sum_{j in N(u)} y_j )

N(u) is the set of items the user has rated. Empirically this is what gets
ML-100K down from ~0.92 (plain MF) to ~0.88 RMSE for me.

The training loop groups updates by user so each user's |N(u)|-sized y_sum
is only built once per mini-batch instead of once per rating. Not the
fanciest impl but easily fits in numpy and is fast enough on 100K ratings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SVDppConfig:
    n_factors: int = 64
    n_epochs: int = 25
    lr: float = 0.01
    lr_decay: float = 0.95
    reg: float = 0.04
    user_minibatch: int = 16     # split each user's ratings into chunks of this size
    seed: int = 42
    verbose: bool = True


class SVDpp:
    def __init__(self, n_users, n_items, config=None):
        self.n_users = n_users
        self.n_items = n_items
        self.cfg = config or SVDppConfig()

        rng = np.random.default_rng(self.cfg.seed)
        scale = 0.1 / np.sqrt(self.cfg.n_factors)
        self.P = rng.normal(0.0, scale, (n_users, self.cfg.n_factors)).astype(np.float32)
        self.Q = rng.normal(0.0, scale, (n_items, self.cfg.n_factors)).astype(np.float32)
        self.Y = rng.normal(0.0, scale, (n_items, self.cfg.n_factors)).astype(np.float32)
        self.bu = np.zeros(n_users, dtype=np.float32)
        self.bi = np.zeros(n_items, dtype=np.float32)
        self.mu = 0.0
        self._user_items = None   # N(u) per user, filled in fit()

    def fit(self, users, items, ratings):
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        ratings = np.asarray(ratings, dtype=np.float32)
        cfg = self.cfg
        self.mu = float(ratings.mean())

        # Sort by user so each user's ratings are contiguous - lets us
        # slice with start:end instead of building a dict.
        order = np.argsort(users, kind="stable")
        u_s = users[order]
        i_s = items[order]
        r_s = ratings[order]
        bounds = np.concatenate(([0], np.flatnonzero(np.diff(u_s)) + 1, [len(u_s)]))
        unique_users = u_s[bounds[:-1]]

        # N(u) = every training item the user rated. (For ML-100K with explicit
        # 1-5 ratings this is a fine proxy for "implicit feedback".)
        self._user_items = [i_s[bounds[k]:bounds[k + 1]] for k in range(len(unique_users))]

        rng = np.random.default_rng(cfg.seed)
        lr = cfg.lr
        mb = cfg.user_minibatch

        for epoch in range(cfg.n_epochs):
            perm = rng.permutation(len(unique_users))
            sse = 0.0
            n_seen = 0

            for k in perm:
                u = int(unique_users[k])
                i_all = i_s[bounds[k]:bounds[k + 1]]
                r_all = r_s[bounds[k]:bounds[k + 1]]
                if len(i_all) == 0:
                    continue
                N_u = self._user_items[k]
                inv_sqrt = 1.0 / np.sqrt(float(len(N_u)))

                # Shuffle this user's ratings and process them in small chunks.
                # Recomputing y_sum once per chunk (not per rating) is the
                # whole point of grouping by user.
                order_u = rng.permutation(len(i_all))
                for cs in range(0, len(order_u), mb):
                    chunk = order_u[cs:cs + mb]
                    i_batch = i_all[chunk]
                    r_batch = r_all[chunk]

                    y_sum = self.Y[N_u].sum(axis=0) * inv_sqrt
                    p_implicit = self.P[u] + y_sum

                    q_batch = self.Q[i_batch]
                    pred = self.mu + self.bu[u] + self.bi[i_batch] + q_batch @ p_implicit
                    err = (r_batch - pred).astype(np.float32)
                    sse += float(np.dot(err, err))
                    n_seen += len(err)

                    # User params: one update for the whole chunk.
                    self.bu[u] += lr * (err.sum() - cfg.reg * self.bu[u])
                    grad_p = (err[:, None] * q_batch).sum(axis=0) - cfg.reg * self.P[u]
                    self.P[u] += lr * grad_p

                    # Item params: per-rating scatter (same duplicate-safety
                    # reasoning as in the MF model).
                    np.add.at(self.bi, i_batch, lr * (err - cfg.reg * self.bi[i_batch]))
                    grad_q = err[:, None] * p_implicit[None, :] - cfg.reg * q_batch
                    np.add.at(self.Q, i_batch, lr * grad_q)

                    # Y[N(u)] all share the same gradient direction.
                    shared = (err[:, None] * q_batch).sum(axis=0) * inv_sqrt
                    y_old = self.Y[N_u]
                    self.Y[N_u] = y_old + lr * (shared[None, :] - cfg.reg * y_old)

            lr *= cfg.lr_decay
            if cfg.verbose:
                rmse = np.sqrt(sse / n_seen)
                print(f"[SVD++] epoch {epoch + 1:02d}/{cfg.n_epochs}  RMSE={rmse:.4f}  lr={lr:.4f}")

        # Cache p_u + y_sum so predict() doesn't have to redo it every call.
        self._implicit_p = self.P.copy()
        for k, N_u in enumerate(self._user_items):
            u = int(unique_users[k])
            inv_sqrt = 1.0 / np.sqrt(float(len(N_u)))
            self._implicit_p[u] = self.P[u] + self.Y[N_u].sum(axis=0) * inv_sqrt
        return self

    def predict(self, users, items):
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        pred = (
            self.mu
            + self.bu[users]
            + self.bi[items]
            + np.einsum("bf,bf->b", self._implicit_p[users], self.Q[items])
        )
        return np.clip(pred, 1.0, 5.0)

    def score_matrix(self, users=None):
        if users is None:
            users = np.arange(self.n_users)
        users = np.asarray(users, dtype=np.int64)
        scores = (
            self.mu
            + self.bu[users][:, None]
            + self.bi[None, :]
            + self._implicit_p[users] @ self.Q.T
        )
        return np.clip(scores, 1.0, 5.0)
