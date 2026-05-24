"""Supervised stacker on top of the collaborative model.

Builds a feature vector per (user, item) and trains a gradient-boosted
regressor to predict the rating. Features:
  - user demographics (age, gender, occupation one-hot)
  - item genre flags
  - smoothed user-mean / item-mean / log-popularity
  - the CF prediction itself  <-- the stacking signal that actually helps

I tried plain linear/ridge regression first; HGBT was noticeably better
on the residuals after CF, so we stuck with it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class SupervisedConfig:
    max_iter: int = 200
    max_depth: int | None = 8
    learning_rate: float = 0.06
    seed: int = 42


class SupervisedRecommender:

    def __init__(self, user_features, item_features, config=None):
        self.user_features = user_features.astype(np.float32)
        self.item_features = item_features.astype(np.float32)
        self.cfg = config or SupervisedConfig()
        self.model = HistGradientBoostingRegressor(
            max_iter=self.cfg.max_iter,
            max_depth=self.cfg.max_depth,
            learning_rate=self.cfg.learning_rate,
            random_state=self.cfg.seed,
        )
        # Filled in by _compute_priors.
        self.user_mean = None
        self.item_mean = None
        self.item_count = None
        self.global_mean = 0.0

    def _compute_priors(self, users, items, ratings):
        n_u = self.user_features.shape[0]
        n_i = self.item_features.shape[0]
        self.global_mean = float(ratings.mean())

        sum_u = np.bincount(users, weights=ratings, minlength=n_u)
        cnt_u = np.bincount(users, minlength=n_u).astype(np.float32)
        sum_i = np.bincount(items, weights=ratings, minlength=n_i)
        cnt_i = np.bincount(items, minlength=n_i).astype(np.float32)

        # Bayesian shrinkage toward the global mean (m = pseudo-count).
        # Without this, users/items with 1-2 ratings get wild means.
        m = 5.0
        self.user_mean = ((sum_u + m * self.global_mean) / (cnt_u + m)).astype(np.float32)
        self.item_mean = ((sum_i + m * self.global_mean) / (cnt_i + m)).astype(np.float32)
        self.item_count = np.log1p(cnt_i).astype(np.float32)

    def _features(self, users, items, cf_pred):
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        uf = self.user_features[users]
        itf = self.item_features[items]
        priors = np.stack(
            [self.user_mean[users], self.item_mean[items], self.item_count[items], cf_pred],
            axis=1,
        )
        return np.concatenate([uf, itf, priors], axis=1)

    def fit(self, users, items, ratings, cf_pred):
        users = np.asarray(users, dtype=np.int64)
        items = np.asarray(items, dtype=np.int64)
        ratings = np.asarray(ratings, dtype=np.float32)
        self._compute_priors(users, items, ratings)
        X = self._features(users, items, cf_pred)
        self.model.fit(X, ratings)
        return self

    def predict(self, users, items, cf_pred):
        X = self._features(users, items, cf_pred)
        pred = self.model.predict(X).astype(np.float32)
        return np.clip(pred, 1.0, 5.0)
