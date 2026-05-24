"""RMSE / MAE / Precision@K / Recall@K - all numpy, no per-row Python loops."""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def relevance_accuracy(y_true, y_pred, threshold=3.5):
    """Did we predict the right "relevant vs not" label?

    With threshold=3.5 this is what gets quoted as the ~85-90% accuracy
    number in the README - it's a sanity check, not a ranking metric.
    """
    yt = np.asarray(y_true) >= threshold
    yp = np.asarray(y_pred) >= threshold
    return float(np.mean(yt == yp))


def precision_recall_at_k(scores, relevance, k=10, mask=None):
    """Precision@K and Recall@K, evaluated over a dense (U, I) grid.

    scores     : (U, I) predicted scores
    relevance  : (U, I) 1/0 - is item i relevant for user u in the test set?
    mask       : optional (U, I) bool. False positions are forced to -inf so
                 they can't sneak into the top-K (used to exclude train items).

    argpartition gives us the unordered top-K per row in O(I) - way faster
    than np.argsort on a 943 x 1682 matrix.
    """
    scores = np.asarray(scores, dtype=np.float64)
    relevance = np.asarray(relevance, dtype=np.float32)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)

    n_users, n_items = scores.shape
    k = min(k, n_items)
    topk_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(n_users)[:, None]
    hits = relevance[rows, topk_idx]            # (U, K)

    hit_counts = hits.sum(axis=1)
    total_relevant = relevance.sum(axis=1)

    # Skip users with no relevant items in the test set - they'd just NaN.
    valid = total_relevant > 0
    if not valid.any():
        return 0.0, 0.0
    precision = float(np.mean(hit_counts[valid] / k))
    recall = float(np.mean(hit_counts[valid] / total_relevant[valid]))
    return precision, recall
