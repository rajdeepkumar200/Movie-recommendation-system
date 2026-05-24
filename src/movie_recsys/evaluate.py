"""Ranking evaluation - one naive baseline + two faster variants.

The naive version is the obvious "loop over users, score every candidate
one at a time" implementation. It's mostly here to benchmark against -
the vectorized version is what gets used in train.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import precision_recall_at_k


@dataclass
class RankingResult:
    precision_at_k: float
    recall_at_k: float


def _build_relevance_and_mask(train, test, n_users, n_items, threshold):
    # relevance[u, i] = 1 if (u,i) in test and rating >= threshold
    # mask[u, i]      = True if (u,i) is NOT in train (eligible to recommend)
    relevance = np.zeros((n_users, n_items), dtype=np.float32)
    rel_rows = test["user"].to_numpy()
    rel_cols = test["item"].to_numpy()
    rel_vals = (test["rating"].to_numpy() >= threshold).astype(np.float32)
    relevance[rel_rows, rel_cols] = rel_vals

    mask = np.ones((n_users, n_items), dtype=bool)
    mask[train["user"].to_numpy(), train["item"].to_numpy()] = False
    return relevance, mask


def ranking_eval_vectorized(model, train, test, n_users, n_items, k=10, threshold=3.5):
    """Score every test user x every item in one matmul, then top-K."""
    test_users = np.array(sorted(test["user"].unique()), dtype=np.int64)
    scores = model.score_matrix(test_users)        # (U, I)

    relevance, mask = _build_relevance_and_mask(train, test, n_users, n_items, threshold)
    p, r = precision_recall_at_k(
        scores, relevance[test_users], k=k, mask=mask[test_users],
    )
    return RankingResult(p, r)


def ranking_eval_naive(model, train, test, n_users, n_items, k=10, threshold=3.5):
    """Per-user Python loop. Slow on purpose - this is the baseline."""
    train_by_user = {}
    for u, i in zip(train["user"].to_numpy(), train["item"].to_numpy()):
        train_by_user.setdefault(int(u), set()).add(int(i))

    test_by_user = {}
    for u, i, r in zip(test["user"].to_numpy(),
                       test["item"].to_numpy(),
                       test["rating"].to_numpy()):
        test_by_user.setdefault(int(u), []).append((int(i), float(r)))

    all_items = np.arange(n_items, dtype=np.int64)
    precisions, recalls = [], []

    for u, pairs in test_by_user.items():
        relevant = {i for i, r in pairs if r >= threshold}
        if not relevant:
            continue
        seen = train_by_user.get(u, set())
        candidates = [i for i in all_items.tolist() if i not in seen]

        # Score every candidate individually. This is the slow part.
        scores = [
            float(model.predict(np.array([u]), np.array([i]))[0])
            for i in candidates
        ]
        order = np.argsort(scores)[::-1][:k]
        topk = [candidates[j] for j in order]
        hits = sum(1 for i in topk if i in relevant)
        precisions.append(hits / k)
        recalls.append(hits / len(relevant))

    return RankingResult(
        precision_at_k=float(np.mean(precisions)) if precisions else 0.0,
        recall_at_k=float(np.mean(recalls)) if recalls else 0.0,
    )


def ranking_eval_sampled(
    model, train, test, n_users, n_items,
    k=10, threshold=4.0, n_negatives=99, seed=42,
):
    """Sampled-negatives ranking eval (the BPR / NCF-style protocol).

    For each user with at least one test rating >= threshold, we take those
    as positives, sample n_negatives items the user hasn't seen, and rank
    the positives + negatives together. Gives much higher P@10 / R@10
    numbers than the full-catalog version because the negatives are
    "easy" - 99 random items vs. ~1500 real candidates.
    """
    rng = np.random.default_rng(seed)

    # Anything in train OR test counts as "seen" so we don't sample a
    # negative that the user actually rated.
    seen = {}
    for u, i in zip(train["user"].to_numpy(), train["item"].to_numpy()):
        seen.setdefault(int(u), set()).add(int(i))
    for u, i in zip(test["user"].to_numpy(), test["item"].to_numpy()):
        seen.setdefault(int(u), set()).add(int(i))

    positives = test[test["rating"] >= threshold]
    user_positives = {}
    for u, i in zip(positives["user"].to_numpy(), positives["item"].to_numpy()):
        user_positives.setdefault(int(u), []).append(int(i))

    precisions, recalls = [], []
    for u, pos_items in user_positives.items():
        s = seen.get(u, set())
        sampled = set()
        # Reject-sample until we have enough negatives. Slightly over-sample
        # to cut down on retries.
        while len(sampled) < n_negatives:
            cand = rng.integers(0, n_items, size=n_negatives * 2)
            for c in cand:
                c = int(c)
                if c in s or c in sampled:
                    continue
                sampled.add(c)
                if len(sampled) == n_negatives:
                    break

        candidates = np.array(list(pos_items) + list(sampled), dtype=np.int64)
        users_arr = np.full(candidates.shape, u, dtype=np.int64)
        scores = model.predict(users_arr, candidates)

        kk = min(k, len(candidates))
        topk_idx = np.argpartition(-scores, kth=kk - 1)[:kk]
        topk_items = set(candidates[topk_idx].tolist())
        pos_set = set(pos_items)
        hits = len(topk_items & pos_set)
        precisions.append(hits / kk)
        recalls.append(hits / len(pos_set))

    return RankingResult(
        precision_at_k=float(np.mean(precisions)) if precisions else 0.0,
        recall_at_k=float(np.mean(recalls)) if recalls else 0.0,
    )
