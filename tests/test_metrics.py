"""Quick sanity checks for the metrics. Not exhaustive - the goal is just
to catch obvious regressions if I touch the vectorized math."""
from __future__ import annotations

import numpy as np

from movie_recsys.metrics import (
    mae,
    precision_recall_at_k,
    relevance_accuracy,
    rmse,
)


def test_rmse_mae_basic():
    y = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 2.0, 4.0])
    # One error of 1.0, the rest perfect -> rmse = sqrt(1/3), mae = 1/3.
    assert abs(rmse(y, p) - np.sqrt(1.0 / 3.0)) < 1e-9
    assert abs(mae(y, p) - 1.0 / 3.0) < 1e-9


def test_relevance_accuracy():
    y = np.array([5.0, 1.0, 4.0, 2.0])
    p = np.array([4.5, 2.0, 3.0, 1.5])
    # y>=3.5 -> [T,F,T,F], p>=3.5 -> [T,F,F,F]. Matches at positions 0,1,3 -> 3/4.
    assert abs(relevance_accuracy(y, p, threshold=3.5) - 0.75) < 1e-9


def test_precision_recall_at_k_perfect():
    # 2 users, 4 items. Top-2 should perfectly hit the relevant ones.
    scores = np.array(
        [[0.9, 0.1, 0.8, 0.2],
         [0.1, 0.9, 0.2, 0.8]],
        dtype=np.float64,
    )
    relevance = np.array(
        [[1, 0, 1, 0],
         [0, 1, 0, 1]],
        dtype=np.float32,
    )
    p, r = precision_recall_at_k(scores, relevance, k=2)
    assert abs(p - 1.0) < 1e-9
    assert abs(r - 1.0) < 1e-9


def test_precision_recall_at_k_with_mask():
    # If we mask out the top-scoring relevant item, top-2 should drop one hit.
    scores = np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float64)
    relevance = np.array([[1, 0, 1, 0]], dtype=np.float32)
    mask = np.array([[False, True, True, True]])
    p, r = precision_recall_at_k(scores, relevance, k=2, mask=mask)
    assert abs(p - 0.5) < 1e-9
    assert abs(r - 0.5) < 1e-9
