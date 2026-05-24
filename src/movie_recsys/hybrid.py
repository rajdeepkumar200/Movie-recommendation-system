"""Hybrid = weighted blend of CF and supervised predictions."""
from __future__ import annotations

import numpy as np


class HybridRecommender:
    """alpha * CF + (1 - alpha) * supervised. Both clipped to [1, 5]."""

    def __init__(self, cf, sup, alpha=0.5):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        self.cf = cf
        self.sup = sup
        self.alpha = float(alpha)

    def predict(self, users, items):
        cf_pred = self.cf.predict(users, items)
        sup_pred = self.sup.predict(users, items, cf_pred)
        return np.clip(self.alpha * cf_pred + (1.0 - self.alpha) * sup_pred, 1.0, 5.0)

    def score_matrix(self, users):
        # Dense (|users|, n_items) hybrid scores. The CF gives us the matrix
        # directly; we flatten for the supervised model, then reshape back.
        users = np.asarray(users, dtype=np.int64)
        cf_scores = self.cf.score_matrix(users)
        n_u, n_i = cf_scores.shape

        u_grid = np.repeat(users, n_i)
        i_grid = np.tile(np.arange(n_i, dtype=np.int64), n_u)
        sup_flat = self.sup.predict(u_grid, i_grid, cf_scores.reshape(-1))
        sup_scores = sup_flat.reshape(n_u, n_i)

        return np.clip(self.alpha * cf_scores + (1.0 - self.alpha) * sup_scores, 1.0, 5.0)
